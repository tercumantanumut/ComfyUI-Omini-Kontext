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


class OminiKontextLatentDecoderNode:
    """
    ComfyUI node for decoding Omini Kontext latents back to images
    """
    
    def __init__(self):
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "latent": ("LATENT",),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "OminiKontext"
    
    def decode(self, pipeline, latent, height, width):
        # Unpack the latents to the correct shape
        vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
        
        # The latent is in packed format, need to unpack it
        batch_size = latent.shape[0]
        latent_channels = pipeline.vae.config.latent_channels
        
        # Unpack using the pipeline's method
        unpacked = pipeline._unpack_latents(latent, height, width, vae_scale_factor)
        
        # Decode through VAE
        with torch.no_grad():
            # Adjust for VAE's expected input
            unpacked = (unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            image = pipeline.vae.decode(unpacked, return_dict=False)[0]
        
        # Convert to ComfyUI format [B, H, W, C] in range [0, 1]
        image = image.clamp(-1, 1)
        image = (image + 1) / 2
        image = image.permute(0, 2, 3, 1).cpu().float()
        
        return (image,)


class OminiKontextLatentVisualizerNode:
    """
    ComfyUI node for visualizing the structure of Omini Kontext latents
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "image_ids": ("IMAGE_IDS",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "visualize"
    CATEGORY = "OminiKontext"
    OUTPUT_NODE = True
    
    def visualize(self, latent, image_ids=None):
        info = []
        info.append("=== Omini Kontext Latent Info ===")
        info.append(f"Latent shape: {latent.shape}")
        info.append(f"Latent dtype: {latent.dtype}")
        info.append(f"Latent device: {latent.device}")
        info.append(f"Latent min/max: {latent.min().item():.4f} / {latent.max().item():.4f}")
        
        if image_ids is not None:
            info.append(f"\nImage IDs shape: {image_ids.shape}")
            info.append(f"Image IDs dtype: {image_ids.dtype}")
            info.append(f"First few IDs: {image_ids[:5].tolist()}")
        
        return ("\n".join(info),)


NODE_CLASS_MAPPINGS = {
    "OminiKontextImageEncoder": OminiKontextImageEncoderNode,
    "OminiKontextTextEncoder": OminiKontextTextEncoderNode,
    "OminiKontextReferenceEncoder": OminiKontextReferenceEncoderNode,
    "OminiKontextLatentCombiner": OminiKontextLatentCombinerNode,
    "OminiKontextLatentDecoder": OminiKontextLatentDecoderNode,
    "OminiKontextLatentVisualizer": OminiKontextLatentVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextImageEncoder": "Omini Kontext Image Encoder",
    "OminiKontextTextEncoder": "Omini Kontext Text Encoder",
    "OminiKontextReferenceEncoder": "Omini Kontext Reference Encoder",
    "OminiKontextLatentCombiner": "Omini Kontext Latent Combiner",
    "OminiKontextLatentDecoder": "Omini Kontext Latent Decoder",
    "OminiKontextLatentVisualizer": "Omini Kontext Latent Visualizer",
}