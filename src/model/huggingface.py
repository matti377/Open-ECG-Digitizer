from typing import Any, Dict, Optional
from src.utils import import_class_from_path
import torch
import torch.nn as nn


class HuggingFaceSegmentation(nn.Module):
    def __init__(
        self,
        model_class_path: str,
        processor_class_path: str,
        model_path: str,
        processor_attributes: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        unfreeze_last_n_layers: int = 1,
    ):
        super(HuggingFaceSegmentation, self).__init__()

        self.processor_class_obj = import_class_from_path(processor_class_path)
        self.processor = self.processor_class_obj.from_pretrained(model_path)

        if processor_attributes is not None:
            for processor_key, processor_value in processor_attributes.items():
                setattr(self.processor, processor_key, processor_value)

        self.model_class_obj = import_class_from_path(model_class_path)
        self.pretrained_model = self.model_class_obj.from_pretrained(model_path, **model_kwargs)

        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        for param in [*self.pretrained_model.parameters()][:-unfreeze_last_n_layers]:
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.processor(x, return_tensors="pt").to(x.device)
        p: torch.Tensor = self.pretrained_model(**t).logits
        p = nn.functional.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return p
