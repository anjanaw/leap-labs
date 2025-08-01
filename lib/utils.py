import torch
from torch import Tensor

def apply_imagenet_norm(image: Tensor) -> Tensor:
    device = image.device
    
    # image shape should be [B,C,H,W]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    return (image - mean) / std


