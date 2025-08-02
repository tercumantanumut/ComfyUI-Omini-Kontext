import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the omini-kontext source to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'omini_kontext_src'))

from pipeline_flux_omini_kontext import FluxOminiKontextPipeline
from pipeline_tools import encode_images, prepare_text_input


class OminiKontextImageEncoderNode:
    """
    ComfyUI node for encoding images using the Omini Kontext pipeline
    """
    
    def __init__(self):
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE_IDS")
    FUNCTION = "encode"
    CATEGORY = "OminiKontext"
    
    def encode(self, pipeline, image):
        # Convert ComfyUI image to tensor format expected by pipeline
        # ComfyUI: [B, H, W, C] in range [0, 1]
        # Pipeline expects: [B, C, H, W] in range [0, 1]
        image_tensor = image.permute(0, 3, 1, 2).to(self.device).to(self.dtype)
        
        # Encode the image
        with torch.no_grad():
            image_tokens, image_ids = encode_images(pipeline, image_tensor)
        
        # Return as separate outputs
        return (image_tokens, image_ids)


class OminiKontextTextEncoderNode:
    """
    ComfyUI node for encoding text prompts using the Omini Kontext pipeline
    """
    
    def __init__(self):
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_sequence_length": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("PROMPT_EMBEDS", "POOLED_EMBEDS", "TEXT_IDS")
    FUNCTION = "encode"
    CATEGORY = "OminiKontext"
    
    def encode(self, pipeline, prompt, max_sequence_length=512):
        # Prepare text input
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                pipeline, 
                [prompt],  # Wrap in list as pipeline expects batch
                max_sequence_length=max_sequence_length
            )
        
        return (prompt_embeds, pooled_prompt_embeds, text_ids)


class OminiKontextReferenceEncoderNode:
    """
    ComfyUI node for encoding reference images with position delta
    """
    
    def __init__(self):
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "reference_image": ("IMAGE",),
                "delta_x": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "delta_y": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "delta_z": ("INT", {"default": 96, "min": -1000, "max": 1000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("REF_LATENT", "REF_IMAGE_IDS")
    FUNCTION = "encode"
    CATEGORY = "OminiKontext"
    
    def encode(self, pipeline, reference_image, delta_x, delta_y, delta_z):
        # Convert ComfyUI image to tensor format
        ref_tensor = reference_image.permute(0, 3, 1, 2).to(self.device).to(self.dtype)
        
        # Encode the reference image
        with torch.no_grad():
            ref_tokens, ref_ids = encode_images(pipeline, ref_tensor)
            
            # Apply position delta to reference image IDs
            ref_ids = ref_ids.clone()
            ref_ids[:, 0] += delta_x
            ref_ids[:, 1] += delta_y
            ref_ids[:, 2] += delta_z
        
        return (ref_tokens, ref_ids)


class OminiKontextLatentCombinerNode:
    """
    ComfyUI node for combining input and reference latents
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_latent": ("LATENT",),
                "input_ids": ("IMAGE_IDS",),
                "reference_latent": ("REF_LATENT",),
                "reference_ids": ("REF_IMAGE_IDS",),
            }
        }
    
    RETURN_TYPES = ("COMBINED_LATENT", "COMBINED_IDS")
    FUNCTION = "combine"
    CATEGORY = "OminiKontext"
    
    def combine(self, input_latent, input_ids, reference_latent, reference_ids):
        # Combine latents along sequence dimension
        combined_latent = torch.cat([input_latent, reference_latent], dim=1)
        
        # Combine IDs
        combined_ids = torch.cat([input_ids, reference_ids], dim=0)
        
        return (combined_latent, combined_ids)


NODE_CLASS_MAPPINGS = {
    "OminiKontextImageEncoder": OminiKontextImageEncoderNode,
    "OminiKontextTextEncoder": OminiKontextTextEncoderNode,
    "OminiKontextReferenceEncoder": OminiKontextReferenceEncoderNode,
    "OminiKontextLatentCombiner": OminiKontextLatentCombinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextImageEncoder": "Omini Kontext Image Encoder",
    "OminiKontextTextEncoder": "Omini Kontext Text Encoder",
    "OminiKontextReferenceEncoder": "Omini Kontext Reference Encoder",
    "OminiKontextLatentCombiner": "Omini Kontext Latent Combiner",
}