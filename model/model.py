import gradio as gr
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
import torch
import os
import traceback
import time
import psutil
from torchvision.transforms.functional import resize
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import AutoConfig, AutoModel, CONFIG_MAPPING, MODEL_MAPPING
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel

# Import from our model_configs module
#from model.teachers.MViT import VideoProcessingConfig, MViTConfig, MViTForVideoClassification
from model.students.S3D import VideoProcessingConfig, S3DConfig, S3DForVideoClassification

def load_and_transform_video(video_path, video_config):
    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(0, video.duration)
    video_frames = video_data['video']

    transform = UniformTemporalSubsample(video_config.num_frames)
    video_frames = transform(video_frames)
    video_frames = video_frames / 255.0
    video_frames = video_frames.permute(1, 0, 2, 3)

    common_size = video_config.resize_to
    video_frames = torch.stack([
        resize(frame, common_size, antialias=True)
        for frame in video_frames
    ])

    video_frames = video_frames.permute(1, 0, 2, 3)
    video_frames = video_frames.unsqueeze(0)

    mean = torch.tensor(video_config.mean, dtype=video_frames.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(video_config.std, dtype=video_frames.dtype).view(1, 3, 1, 1, 1)
    video_frames = (video_frames - mean) / std

    return video_frames

CONFIG_MAPPING.register("S3D", S3DConfig)
MODEL_MAPPING.register(S3DConfig, S3DForVideoClassification)
# Initialize model and configs
# model_config = MViTConfig(
#     num_classes=2,
#     num_frames=16,
#     model="MViT",
#     is_pretrained=True,
#     reinitialize_head=True,
#     label2id={"NonViolence": 0, "Violence": 1},
#     id2label={0: "NonViolence", 1: "Violence"}
# )

model_config = AutoConfig.from_pretrained(f"DanJoshua/estudiante_S3D_profesor_MViT_kl_VIOPERU")
model = AutoModel.from_pretrained(f"DanJoshua/estudiante_S3D_profesor_MViT_kl_VIOPERU", config=model_config)

# Download and load weights
weights_path = hf_hub_download(
    repo_id="DanJoshua/estudiante_S3D_profesor_MViT_kl_VIOPERU",
    filename="model.safetensors"
)

state_dict = load_file(weights_path)
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

video_config = VideoProcessingConfig()

def predict_violence(video_path):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    
    try:
        if video_path is None:
            return None, {
                "Predicción": "No se ha subido ningún video",
                "Métricas del Modelo": {
                    "Nombre del Modelo": model_config.model,
                    "Número de Parámetros": "N/A",
                    "Tiempo de Inferencia": "N/A",
                    "Tiempo de Procesamiento": "N/A",
                    "Consumo de RAM": "N/A"
                }
            }

        # Measure initial RAM usage
        initial_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Measure processing time
        start_process = time.time()
        video_frames = load_and_transform_video(video_path, video_config)
        video_frames = video_frames.to(device)
        process_time = time.time() - start_process

        # Measure inference time
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(video_frames)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
        inference_time = time.time() - start_inference

        # Measure final RAM usage
        final_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        ram_used = final_ram - initial_ram

        prob_no_violence = probabilities[0][0].item()
        prob_violence = probabilities[0][1].item()

        if prob_violence < 0.60:
            prediction = 0
            confidence = prob_no_violence
        else:
            prediction = 1
            confidence = prob_violence

        label = model_config.id2label[prediction]

        return video_path, {
            "Predicción": label,
            "Confianza": f"{confidence:.2%}",
            "Métricas del Modelo": {
                "Nombre del Modelo": model_config.model,
                "Número de Parámetros": f"{model.get_num_parameters():,}",
                "Tiempo de Inferencia": f"{inference_time:.3f} segundos",
                "Tiempo de Procesamiento": f"{process_time:.3f} segundos",
                "Consumo de RAM": f"{ram_used:.2f} MB"
            }
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        return None, {
            "Error": f"Error durante el procesamiento: {str(e)}",
            "Detalle": error_trace,
            "Predicción": "Error",
            "Métricas del Modelo": {
                "Nombre del Modelo": model_config.model,
                "Número de Parámetros": "Error",
                "Tiempo de Inferencia": "Error",
                "Tiempo de Procesamiento": "Error",
                "Consumo de RAM": "Error"
            }
        }

def create_interface():
    with gr.Blocks(title="Detector de Violencia en Videos") as interface:
        gr.Markdown("""
        # 🎥 Detector de Violencia en Videos

        Sube un video para analizar si contiene contenido violento.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(label="Video de entrada")
            with gr.Column(scale=1):
                submit_btn = gr.Button("Analizar Video", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                video_output = gr.Video(label="Video analizado")
            with gr.Column(scale=1):
                output = gr.JSON(label="Resultados")

        submit_btn.click(
            fn=predict_violence,
            inputs=video_input,
            outputs=[video_output, output]
        )

    return interface