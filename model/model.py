import gradio as gr
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
import torch
import os
import traceback
import time
import psutil
import gc
import cv2
import numpy as np
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
            label2id={"No Violencia": 0, "Violencia": 1},
            id2label={0: "No Violencia", 1: "Violencia"}
        )
        print("Cargando modelo...")
        model = S3DForVideoClassification(model_config)
    elif model_type == "maestro":
        print("\nCargando configuración del modelo maestro (MViT)...")
        model_config = MViTConfig.from_pretrained(
            dict_models[model_type],
            num_classes=2,
            label2id={"No Violencia": 0, "Violencia": 1},
            id2label={0: "No Violencia", 1: "Violencia"}
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


def process_frames(frames, video_config):
    """Procesa una lista de frames para el modelo."""
    # Convertir frames a tensor
    frames = np.stack(frames)  # (N, H, W, C)
    frames = torch.from_numpy(frames).float()  # Convertir a tensor
    
    # Cambiar orden de dimensiones a (C, T, H, W)
    frames = frames.permute(3, 0, 1, 2)
    
    # Submuestreo temporal uniforme
    transform = UniformTemporalSubsample(video_config.num_frames)
    frames = transform(frames)
    
    # Normalizar a [0, 1]
    frames = frames / 255.0
    
    # Redimensionar frames
    common_size = video_config.resize_to
    frames = torch.stack([
        resize(frame, common_size, antialias=True)
        for frame in frames.permute(1, 0, 2, 3)  # (T, C, H, W)
    ]).permute(1, 0, 2, 3)  # Volver a (C, T, H, W)
    
    # Normalizar con media y desviación estándar
    mean = torch.tensor(video_config.mean, dtype=frames.dtype).view(3, 1, 1, 1)
    std = torch.tensor(video_config.std, dtype=frames.dtype).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    # Agregar dimensión de batch
    frames = frames.unsqueeze(0)  # (1, C, T, H, W)
    
    return frames

def analyze_video_segments(video_path, model_type, segment_duration=2):
    """Analiza el video en segmentos y genera un nuevo video con los resultados."""
    try:
        # Inicializar el modelo
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
        model_config, model = get_model(model_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        video_config = VideoProcessingConfig()

        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Preparar el video de salida
        output_filename = f"processed_{os.path.basename(video_path)}"
        output_path = os.path.join(os.path.dirname(video_path), output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frames_per_segment = int(fps * segment_duration)
        current_frame = 0
        segment_frames = []
        current_time = 0.0

        while current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            segment_frames.append(frame)
            current_frame += 1

            # Procesar segmento cuando se alcanza el tamaño deseado
            if len(segment_frames) == frames_per_segment or current_frame == total_frames:
                try:
                    # Procesar frames del segmento
                    video_tensor = process_frames(segment_frames, video_config)
                    video_tensor = video_tensor.to(device)

                    # Realizar inferencia
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(video_tensor)
                        logits = outputs['logits']
                        probabilities = torch.softmax(logits, dim=1)
                    inference_time = time.time() - start_time

                    # Obtener predicción
                    prob_violence = probabilities[0][1].item()
                    prob_no_violence = probabilities[0][0].item()
                    prediction = model_config.id2label.get(1 if prob_violence >= 0.60 else 0)
                    
                    # Definir colores (BGR format)
                    RED = (0, 0, 255)
                    GREEN = (0, 255, 0)
                    BLACK = (0, 0, 0)
                    WHITE = (255, 255, 255)

                    for frame in segment_frames:
                        frame_width = frame.shape[1]
                        frame_height = frame.shape[0]
                        
                        # Definir dimensiones y posición del recuadro blanco
                        box_width = 300
                        box_height = 230  # Aumentado para dar más espacio en la parte inferior
                        box_x = frame_width - box_width - 20
                        box_y = (frame_height // 2) - (box_height // 2)
                        
                        # Dibujar recuadro blanco con borde negro
                        cv2.rectangle(frame, (box_x, box_y), 
                                    (box_x + box_width, box_y + box_height), 
                                    WHITE, -1)
                        cv2.rectangle(frame, (box_x, box_y), 
                                    (box_x + box_width, box_y + box_height), 
                                    BLACK, 1)

                        # Configuración de barras
                        bar_height = 20
                        bar_width = 250
                        bar_x = box_x + 25
                        bar_y = box_y + 45
                        
                        # Barra de violencia (roja)
                        cv2.rectangle(frame, (bar_x, bar_y), 
                                    (bar_x + int(bar_width * prob_violence), bar_y + bar_height), 
                                    RED, -1)
                        cv2.rectangle(frame, (bar_x, bar_y), 
                                    (bar_x + bar_width, bar_y + bar_height), 
                                    BLACK, 1)
                        
                        # Barra de no violencia (verde)
                        bar_y += bar_height + 20
                        cv2.rectangle(frame, (bar_x, bar_y), 
                                    (bar_x + int(bar_width * prob_no_violence), bar_y + bar_height), 
                                    GREEN, -1)
                        cv2.rectangle(frame, (bar_x, bar_y), 
                                    (bar_x + bar_width, bar_y + bar_height), 
                                    BLACK, 1)

                        # Texto de predicción con color según resultado
                        text_color = RED if prediction == "Violencia" else GREEN
                        prediction_text = f"{prediction}"
                        cv2.putText(frame, prediction_text, 
                                (bar_x, bar_y + bar_height + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                        # Textos de tiempo en negro
                        time_text = f"Time: {current_time:.1f}s"
                        cv2.putText(frame, time_text, 
                                (bar_x, bar_y + bar_height + 75),  # Aumentado el espaciado
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 1)
                        
                        inf_text = f"Inference: {inference_time:.3f}s"
                        cv2.putText(frame, inf_text, 
                                (bar_x, bar_y + bar_height + 110),  # Aumentado el espaciado
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 1)
                        
                        out.write(frame)

                except Exception as e:
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                    return None, {
                        "Error": f"Error durante Procesar frames del segmento: {str(e)}",
                        "Detalle": traceback.format_exc()
                    }
                # Actualizar tiempo y limpiar frames
                current_time += segment_duration
                segment_frames = []
                cleanup_memory()

        # Liberar recursos
        cap.release()
        out.release()
        model.cpu()
        del model
        cleanup_memory()

        return output_path

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None, {
            "Error": f"Error durante el análisis de segmentos de video: {str(e)}",
            "Detalle": traceback.format_exc()
        }
    
def process_frames(frames, video_config):
    """Procesa una lista de frames para el modelo."""
    # Convertir frames a tensor
    frames = np.stack(frames)  # (N, H, W, C)
    frames = torch.from_numpy(frames).float()  # Convertir a tensor
    
    # Cambiar orden de dimensiones a (C, T, H, W)
    frames = frames.permute(3, 0, 1, 2)
    
    # Submuestreo temporal uniforme
    transform = UniformTemporalSubsample(video_config.num_frames)
    frames = transform(frames)
    
    # Normalizar a [0, 1]
    frames = frames / 255.0
    
    # Redimensionar frames
    common_size = video_config.resize_to
    frames = torch.stack([
        resize(frame, common_size, antialias=True)
        for frame in frames.permute(1, 0, 2, 3)  # (T, C, H, W)
    ]).permute(1, 0, 2, 3)  # Volver a (C, T, H, W)
    
    # Normalizar con media y desviación estándar
    mean = torch.tensor(video_config.mean, dtype=frames.dtype).view(3, 1, 1, 1)
    std = torch.tensor(video_config.std, dtype=frames.dtype).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    # Agregar dimensión de batch
    frames = frames.unsqueeze(0)  # (1, C, T, H, W)
    
    return frames


def predict_violence(video_path, model_type):
    """Función principal modificada para analizar el video por segmentos."""
    try:
        # Analizar el video y generar salida
        output_path = analyze_video_segments(video_path, model_type)
        return output_path, {
            "Mensaje": "Video procesado exitosamente",
            "Ruta del video": output_path
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None, {
            "Error": f"Error durante el procesamiento: {str(e)}",
            "Detalle": traceback.format_exc()
        }