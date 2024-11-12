from dataclasses import dataclass
from typing import Tuple
from transformers import PreTrainedModel, PretrainedConfig
from torch import nn
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

@dataclass
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

class MViTConfig(PretrainedConfig):
    def __init__(self,
                 num_classes=2,
                 num_frames=16,
                 model="MViT",
                 is_pretrained=True,
                 reinitialize_head=True,
                 label2id={"NonViolence": 0, "Violence": 1},
                 id2label={0: "NonViolence", 1: "Violence"},
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.model = model
        self.is_pretrained = is_pretrained
        self.reinitialize_head = reinitialize_head
        self.label2id = label2id
        self.id2label = id2label

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class MViTForVideoClassification(PreTrainedModel):
    config_class = MViTConfig
    
    def __init__(self, config: MViTConfig):
        super().__init__(config)
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MViT_V2_S_Weights.DEFAULT if config.is_pretrained else None
        self.model = mvit_v2_s(weights=weights)
        if config.reinitialize_head:
            self._modify_classification_head()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.to(device)
    
    def _modify_classification_head(self):
        if hasattr(self.model, 'head'):
            last_linear = None
            for module in self.model.head.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                in_features = last_linear.in_features
            else:
                raise ValueError("No linear layer found in model head")
            
        else:
            raise ValueError("Model has no 'head' attribute")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, self.config.num_classes)
        )
        self.model.head = nn.Identity()
    
    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())