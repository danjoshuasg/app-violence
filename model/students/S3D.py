from torchvision.models.video import s3d, S3D_Weights
from transformers import PreTrainedModel, PretrainedConfig
from torch import nn
import torch
import torch.nn.functional as F
from typing import Tuple

class VideoProcessingConfig:
    name_dataset: str = "RWF2000"
    strategy: str = "N"
    frac_dataset: float = 1.0
    val_fraction: float = 0.08
    seed: int = 30
    num_frames: int = 16
    sample_rate: int = 4
    fps: int = 30
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
    resize_to: Tuple[int, int] = (224, 224)

class S3DConfig(PretrainedConfig):
    """Clase de configuración para el modelo S3D."""
    def __init__(self,
                 num_classes=2,
                 num_frames=16,
                 model="S3D",
                 is_pretrained=True,
                 reinitialize_head =True,
                 distillation_type="kl",  # Tipo de destilación
                 label2id={"NonViolence": 0, "Violence": 1},
                 id2label={0: "NonViolence", 1: "Violence"},
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.model = model
        self.is_pretrained = is_pretrained
        self.reinitialize_head = reinitialize_head
        self.distillation_type = distillation_type
        self.label2id = label2id
        self.id2label = id2label


class S3DForVideoClassification(PreTrainedModel):
    config_class = S3DConfig

    def __init__(self, config: S3DConfig):
        super().__init__(config)
        self.config = config

        weights = S3D_Weights.DEFAULT if config.is_pretrained else None
        self.model = s3d(weights=weights)

        if config.reinitialize_head:
            self._modify_classification_head()

        self.loss_fn = nn.CrossEntropyLoss()

    def _modify_classification_head(self):
        in_features = self.model.classifier[1].in_channels
        self.model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_features, self.config.num_classes, kernel_size=1),
            nn.Flatten(start_dim=1)
        )

    def forward(self, pixel_values, labels=None, teacher_logits=None, temperature=1.0, alpha=0.5, mu=0.5):
        if pixel_values.dim() == 5:
            pass
        elif pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)
        else:
            raise ValueError(f"Unexpected input shape: {pixel_values.shape}")

        features = self.model.features(pixel_values)
        logits = self.model.classifier(features)

        loss = None
        if labels is not None:
            if teacher_logits is not None and self.training:
                #print("Calculando pérdida de distilación")
                # Calcular la pérdida de distilación
                loss = self.distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=temperature,
                    alpha=alpha,
                    mu = mu
                )
            else:
                #print("Calculando pérdida estándar de clasificación")
                # Calcular la pérdida estándar de clasificación
                loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

    def _kl_distillation_loss(self, student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5, mu=None):
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_soft_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_soft_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        task_loss = self.loss_fn(student_logits, labels)
        return alpha * kl_loss + (1 - alpha) * task_loss


    def _akl_distillation_loss(self, student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5, mu=0.5):
        # Compute probabilities
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        # Compute the gap function
        gap = torch.abs(teacher_probs - student_probs)

        # Compute the mask M
        M = (teacher_probs >= mu).float()

        # Compute g_head and g_tail
        g_head = (M * gap).sum(dim=-1)
        g_tail = ((1 - M) * gap).sum(dim=-1)

        # Compute FKL and RKL
        fkl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')*(temperature ** 2)
        rkl = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean')*(temperature ** 2)

        # Compute AKL
        w_head = g_head / (g_head + g_tail + 1e-6)
        w_tail = g_tail / (g_head + g_tail + 1e-6)
        akl = (w_head * fkl + w_tail * rkl).mean()

        # Compute task loss
        task_loss = F.cross_entropy(student_logits, labels)

        # Compute final loss
        total_loss = alpha * akl * (temperature**2) + (1 - alpha) * task_loss

        return total_loss

    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5, mu=0.5):
        if self.config.distillation_type == "kl":
            return self._kl_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
        elif self.config.distillation_type == "akl":
            return self._akl_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, mu)
        elif self.config.distillation_type == "response":
            return self._response_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
        else:
            raise ValueError(f"Unsupported distillation type: {self.config.distillation_type}")


    def get_labels(self):
        return self.config.id2label

    def set_labels(self, labels):
        self.config.id2label = {i: label for i, label in enumerate(labels)}
        self.config.label2id = {label: i for i, label in enumerate(labels)}
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())        