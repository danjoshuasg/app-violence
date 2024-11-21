import gradio as gr
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
import torch
import os
import traceback
import time
import psutil
import gc
from torchvision.transforms.functional import resize
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import AutoConfig, AutoModel, CONFIG_MAPPING, MODEL_MAPPING
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel

# Import from our model_configs module
#from model.teachers.MViT import VideoProcessingConfig, MViTConfig, MViTForVideoClassification
from model.students.S3D import VideoProcessingConfig, S3DConfig, S3DForVideoClassification
from model.teachers.MViT import VideoProcessingConfig, MViTConfig, MViTForVideoClassification

def cleanup_memory():
    """Clean up memory for both CPU and GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Print memory stats
    process = psutil.Process()
    print(f"CPU Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def unregister_models():
    """Unregister models from configuration mappings."""
    try:
        # Eliminar registros antiguos si existen
        CONFIG_MAPPING.pop("S3D", None)
        CONFIG_MAPPING.pop("MViT", None)
        
        # Limpiar MODEL_MAPPING
        keys_to_remove = []
        for key in MODEL_MAPPING._registry.keys():
            if hasattr(key, '__name__') and key.__name__ in ["S3DConfig", "MViTConfig"]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            MODEL_MAPPING._registry.pop(key, None)
            
    except Exception as e:
        print(f"Advertencia al limpiar registros: {str(e)}")
        # Continuamos incluso si hay error, ya que no es crítico

def get_model(model_type):
    CONFIG_MAPPING.register("S3D", S3DConfig)
    CONFIG_MAPPING.register("MViT", MViTConfig)
    MODEL_MAPPING.register(S3DConfig, S3DForVideoClassification)
    MODEL_MAPPING.register(MViTConfig, MViTForVideoClassification)
    """Obtiene el modelo con seguimiento de memoria para CPU y GPU."""
    dict_models = {
        "estudiante": "DanJoshua/estudiante_S3D_profesor_MViT_kl_RWF2000",
        "maestro": "DanJoshua/profesor_MViT_S_RWF2000"
    }
    
    print(f"\n{'='*50}")
    print(f"Iniciando carga del modelo {model_type}")
    print(f"{'='*50}")
    
    # Limpiar memoria
    cleanup_memory()
    
    # Imprimir estado de memoria inicial
    print("\nMemoria antes de cargar el modelo:")
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    # Cargar el modelo apropiado
    if model_type == "estudiante":
        print("\nCargando configuración del modelo estudiante (S3D)...")
        model_config = S3DConfig.from_pretrained(
            dict_models[model_type],
            num_classes=2,
            label2id={"NonViolence": 0, "Violence": 1},
            id2label={0: "NonViolence", 1: "Violence"}
        )
        print("Cargando modelo...")
        model = S3DForVideoClassification(model_config)
    elif model_type == "maestro":
        print("\nCargando configuración del modelo maestro (MViT)...")
        model_config = MViTConfig.from_pretrained(
            dict_models[model_type],
            num_classes=2,
            label2id={"NonViolence": 0, "Violence": 1},
            id2label={0: "NonViolence", 1: "Violence"}
        )
        print("Cargando modelo...")
        model = MViTForVideoClassification(model_config)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    print(f"\nDescargando pesos desde {dict_models[model_type]}...")
    weights_path = hf_hub_download(
        repo_id=dict_models[model_type],
        filename="model.safetensors"
    )

    print("Cargando pesos del modelo...")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)

    # Imprimir estado de memoria final
    print("\nMemoria después de cargar el modelo:")
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    print(f"\nNúmero de parámetros: {model.get_num_parameters():,}")
    print(f"{'='*50}\n")
        
    return model_config, model


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




def predict_violence(video_path, model_type):
    """Modified prediction function with proper cleanup for both CPU and GPU."""
    try:
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
        
        print(f"\nIniciando predicción con modelo {model_type}...")
        model_config, model = get_model(model_type)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")
        
        model = model.to(device)
        model.eval()
        
        # Track initial memory
        process = psutil.Process()
        initial_cpu_memory = process.memory_info().rss
        initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        video_config = VideoProcessingConfig()
        
        result = process_video(video_path, model, model_config, device, video_config)
        
        # Cleanup
        model.cpu()
        del model
        del model_config
        cleanup_memory()
        
        # Track final memory
        final_cpu_memory = process.memory_info().rss
        final_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print("\nEstadísticas de memoria:")
        print(f"CPU Memory diff: {(final_cpu_memory - initial_cpu_memory)/1024/1024:.2f} MB")
        if torch.cuda.is_available():
            print(f"GPU Memory diff: {(final_gpu_memory - initial_gpu_memory)/1024/1024:.2f} MB")
        
        return result
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        cleanup_memory()
        return None, {
            "Error": f"Error durante el procesamiento: {str(e)}",
            "Detalle": traceback.format_exc()
        }

def process_video(video_path, model, model_config, device, video_config):
    """Video processing with memory tracking."""
    if video_path is None:
        return None, {"Predicción": "No se ha subido ningún video"}

    try:
        # Measure initial RAM usage
        process = psutil.Process()
        initial_ram = process.memory_info().rss / 1024 / 1024

        # Process video
        start_process = time.time()
        video_frames = load_and_transform_video(video_path, video_config)
        video_frames = video_frames.to(device)
        process_time = time.time() - start_process

        # Inference
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(video_frames)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
        inference_time = time.time() - start_inference

        # Cleanup processed video data
        del video_frames
        cleanup_memory()

        # Calculate metrics
        final_ram = process.memory_info().rss / 1024 / 1024
        ram_used = final_ram - initial_ram

        # Get prediction
        prob_violence = probabilities[0][1].item()
        prediction = 1 if prob_violence >= 0.60 else 0
        confidence = prob_violence if prediction == 1 else probabilities[0][0].item()

        return video_path, {
            "Predicción": model_config.id2label.get(prediction, "Unknown"),
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
        print(f"Error en el procesamiento del video: {str(e)}")
        traceback.print_exc()
        cleanup_memory()
        raise

# def predict_violence(video_path, model_type):
#     login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)

#     model_config, model = get_model(model_type)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
#     video_config = VideoProcessingConfig()
#     try:
#         if video_path is None:
#             return None, {
#                 "Predicción": "No se ha subido ningún video",
#                 "Métricas del Modelo": {
#                     "Nombre del Modelo": model_config.model,
#                     "Número de Parámetros": "N/A",
#                     "Tiempo de Inferencia": "N/A",
#                     "Tiempo de Procesamiento": "N/A",
#                     "Consumo de RAM": "N/A"
#                 }
#             }

#         # Measure initial RAM usage
#         initial_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

#         # Measure processing time
#         start_process = time.time()
#         video_frames = load_and_transform_video(video_path, video_config)
#         video_frames = video_frames.to(device)
#         process_time = time.time() - start_process

#         # Measure inference time
#         start_inference = time.time()
#         with torch.no_grad():
#             outputs = model(video_frames)
#             logits = outputs['logits']
#             probabilities = torch.softmax(logits, dim=1)
#         inference_time = time.time() - start_inference

#         # Measure final RAM usage
#         final_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB
#         ram_used = final_ram - initial_ram

#         prob_no_violence = probabilities[0][0].item()
#         prob_violence = probabilities[0][1].item()

#         if prob_violence < 0.60:
#             prediction = 0  # NonViolence
#             confidence = prob_no_violence
#         else:
#             prediction = 1  # Violence
#             confidence = prob_violence

#         # Safely get the label with a fallback
#         label = model_config.id2label.get(prediction, "Unknown")

#         return video_path, {
#             "Predicción": label,
#             "Confianza": f"{confidence:.2%}",
#             "Métricas del Modelo": {
#                 "Nombre del Modelo": model_config.model,
#                 "Número de Parámetros": f"{model.get_num_parameters():,}",
#                 "Tiempo de Inferencia": f"{inference_time:.3f} segundos",
#                 "Tiempo de Procesamiento": f"{process_time:.3f} segundos",
#                 "Consumo de RAM": f"{ram_used:.2f} MB"
#             }
#         }

#     except Exception as e:
#         error_trace = traceback.format_exc()
#         return None, {
#             "Error": f"Error durante el procesamiento: {str(e)}",
#             "Detalle": error_trace,
#             "Predicción": "Error",
#             "Métricas del Modelo": {
#                 "Nombre del Modelo": model_config.model,
#                 "Número de Parámetros": "Error",
#                 "Tiempo de Inferencia": "Error",
#                 "Tiempo de Procesamiento": "Error",
#                 "Consumo de RAM": "Error"
#             }
#         }

# def create_interface():
#     with gr.Blocks(title="Detector de Violencia en Videos") as interface:
#         gr.Markdown("""
#         # 🎥 Detector de Violencia en Videos

#         Sube un video para analizar si contiene contenido violento.
#         """)

#         with gr.Row():
#             with gr.Column(scale=2):
#                 video_input = gr.Video(label="Video de entrada")
#             with gr.Column(scale=1):
#                 submit_btn = gr.Button("Analizar Video", variant="primary")

#         with gr.Row():
#             with gr.Column(scale=2):
#                 video_output = gr.Video(label="Video analizado")
#             with gr.Column(scale=1):
#                 output = gr.JSON(label="Resultados")

#         submit_btn.click(
#             fn=predict_violence,
#             inputs=video_input,
#             outputs=[video_output, output]
#         )

#     return interface