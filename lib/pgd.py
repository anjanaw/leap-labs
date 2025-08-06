
import os
import warnings
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch import device as Device
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .utils import apply_imagenet_norm

class PGD:
    def __init__(self, use_l2: bool, device_str: Optional[str] = None):
        """
        Initialize

        Args:
            device_str (Optional[str]): 'cpu', 'cuda' or None
        """
        # Select device (CPU or GPU) based on user input and availability
        self.device = self._get_device(device_str)

        # Load pretrained ConvNeXt-Tiny model and set to eval mode
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        self.use_l2 = use_l2

        if self.use_l2:
            self.epsilon = 2.0
            self.alpha = 0.02
            self.max_steps = 100
        else:
            # pgd l_inf parameters
            self.epsilon = 4/255
            self.alpha = 1/255
            self.max_steps = 40

        # preprocessing transforms for ImageNet pretrained convNeXt-tiny: resize min h/w to 224 and convert to tensor
        # skipping ImageNet normalisation - added only for inference
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def _get_device(self, device: Optional[str] = None) -> Device:
        if device == 'cuda':
            if torch.cuda.is_available():
                return Device('cuda')
            else:
                warnings.warn("CUDA requested but not available. Using CPU instead.")
                return Device('cpu')
        elif device == 'cpu':
            return Device('cpu')
        
        return Device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_from_file(self, image_path: str, target_class: int, output_path: str) -> bool:
        """
        Generates a targeted adversarial example for a given input image using pgd

        Args:
            image_path (str): Path to the input image.
            target_class (int): Target ImageNet class (0–999).
            output_path (str): Folder to save the adversarial image.

        Returns:
            bool: True if image generated successfully, False otherwise.
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # run generate on PIL image
        adverse_image = self.generate_from_pil(image, target_class)

        # If successful, save adversarial image in orginal size.
        if adverse_image:
            org_name, org_extension = os.path.basename(image_path).split(".")
            output_file = os.path.join(output_path, f"{org_name}_{target_class}.{org_extension}")
            adverse_image.save(output_file)

        return adverse_image is not None
    
    def generate_from_pil(self, image: Image.Image, target_class: int) -> Optional[Image.Image]:
        """
        Generates a targeted adversarial example for a given PIL image using pgd

        Args:
            image (Image): PIL Image.
            target_class (int): Target ImageNet class (0–999).

        Returns:
            adverse_image (Image): If successful returns generated PIL image, None otherwise
        """
        source_image = self.transforms(image)
        source_image = source_image.unsqueeze(0)    
        source_image = source_image.to(self.device)

        target_tensor = torch.tensor(target_class, dtype=torch.long).unsqueeze(0).to(self.device)

        # Optional: get model prediction for the orginal image to further validate target class
        with torch.no_grad():
            # apply normalisation before inference
            source_image_norm = apply_imagenet_norm(source_image)
            logits = self.model(source_image_norm)
            predicted_class = torch.argmax(logits).item()

        if predicted_class == target_class:
            warnings.warn("Target class is the same as the model's initial prediction. No adversarial perturbation will be applied.")
            return None

        # Clone the original image for adversarial modification
        adverse_image = source_image.clone().detach()

        adverse_class = None
        step = 0
        # Perform iterative pgd updates with early stop
        while step < self.max_steps:
            adverse_image.requires_grad = True

            # apply normalisation before inference
            adverse_image_norm = apply_imagenet_norm(adverse_image)
            adverse_logits = self.model(adverse_image_norm)
            adverse_class = torch.argmax(adverse_logits).item()
            
            # check for early stop
            if adverse_class == target_class:
                break

            # Loss computation to maximize target class probability
            adverse_loss = F.cross_entropy(adverse_logits, target_tensor)
            grad = torch.autograd.grad(adverse_loss, adverse_image, retain_graph=False, create_graph=False)[0]

            if self.use_l2:
                # Update image with gradient normalised 
                g_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                grad_norm = grad / (g_norm + 1e-8)
                adverse_image = adverse_image - self.alpha * grad_norm

                delta = adverse_image - source_image
                d_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                delta = delta * (self.epsilon / (d_norm + 1e-8)).clamp(max=1.0)

                adverse_image = source_image + delta
                adverse_image = torch.clamp(adverse_image, 0, 1).detach()
            else:
                # Update image with gradient sign, constrained by epsilon
                adverse_image = adverse_image - self.alpha * grad.sign()
                adverse_image = torch.max(torch.min(adverse_image, source_image + self.epsilon), source_image - self.epsilon)
                adverse_image = torch.clamp(adverse_image, 0, 1).detach()
            step += 1
        
        # If successful, return adversarial image in PIL format in orginal size.
        if adverse_class == target_class:
            adverse_image = adverse_image.squeeze(0).cpu().detach()
            return to_pil_image(adverse_image).resize(image.size, resample=Image.LANCZOS)  
        
        return None
        

