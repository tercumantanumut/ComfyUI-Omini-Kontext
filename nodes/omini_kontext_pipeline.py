import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.utils
import os
import sys
import json
import folder_paths
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions
from contextlib import contextmanager
from safetensors.torch import load_file
from glob import glob

import zipfile

folder_names_and_paths["unet"] = ([os.path.join(models_dir, "unet")], supported_pt_extensions)

# Add the omini-kontext source to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'omini_kontext_src'))

from transformers import (
    CLIPTextModel, CLIPTextConfig,
    CLIPTokenizer, 
    T5EncoderModel, T5Config,
    T5TokenizerFast
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL, FluxTransformer2DModel

from pipeline_flux_omini_kontext import FluxOminiKontextPipeline
from diffusers.utils import load_image


current_dir = os.path.dirname(os.path.abspath(__file__))


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


class OminiKontextPipelineLoaderNode:
    """
    ComfyUI node for loading the Flux Omini Kontext pipeline
    """
    
    def __init__(self):
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-Kontext-dev", "multiline": False}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": "", "multiline": False}),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("OMINI_KONTEXT_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "OminiKontext"
    
    def load_pipeline(self, model_path, lora_path="", hf_token=""):
        # Load pipeline
        print(f"Loading Flux Omini Kontext pipeline from: {model_path}")
        
        # Prepare kwargs for from_pretrained
        kwargs = {"torch_dtype": self.dtype}
        if hf_token:
            kwargs["token"] = hf_token
            print("Using provided Hugging Face token for authentication")
        
        pipeline = FluxOminiKontextPipeline.from_pretrained(
            model_path,
            **kwargs
        ).to(self.device)
        
        # Enable memory optimizations
        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        
        # Load LoRA if provided
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            pipeline.load_lora_weights(lora_path, adapter_name="omini_kontext")
        
        return (pipeline,)


class OminiKontextSplitPipelineLoaderNode:

    def __init__(self):
        if not os.path.exists(os.path.join(current_dir, "configs")):
            with zipfile.ZipFile(os.path.join(current_dir, "configs.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(current_dir))

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        transformer_path = [os.path.basename(f) for f in glob(os.path.join(models_dir, "unet/*")) if f.endswith(('.safetensors', '.sft', '.gguf'))]
        clip_path = [f for f in folder_paths.get_filename_list("clip") if f.endswith(('.safetensors', '.sft'))]
        t5_path = [f for f in folder_paths.get_filename_list("clip") if f.endswith(('.safetensors', '.sft'))]
        vae_path = [f for f in folder_paths.get_filename_list("vae") if f.endswith(('.safetensors', '.sft'))]

        return {
            "required": {
                "transformer_path": (transformer_path,),
                "clip_path": (clip_path,),
                "t5_path": (t5_path,),
                "vae_path": (vae_path, )
            },
            "optional": {
                "lora_path": ("STRING", {"default": "", "multiline": False}),
                "auto_t5_gguf": ("BOOLEAN", {"default": False, "tooltip": "Auto load GGUF quantization for T5 model if available."}),
            }
        }
    
    RETURN_TYPES = ("OMINI_KONTEXT_PIPELINE",)
    FUNCTION = "load_safetensor"
    CATEGORY = "OminiKontext"

    def load_safetensor(self, transformer_path, clip_path, t5_path, vae_path, lora_path="", auto_t5_gguf=False, auto_transformer_gguf=False):
        # Load pipeline
        transformer_path = os.path.join(models_dir, "unet", transformer_path)
        clip_path = os.path.join(models_dir, "clip", clip_path)
        t5_path = os.path.join(models_dir, "clip", t5_path)
        vae_path = os.path.join(models_dir, "vae", vae_path)
        
        # Prepare kwargs for from_pretrained
        kwargs = {"torch_dtype": self.dtype}
        
        scheduler = FlowMatchEulerDiscreteScheduler(**read_json(os.path.join(current_dir, "configs/scheduler/scheduler_config.json")))
        clip_model = CLIPTextModel(
            CLIPTextConfig(**read_json(os.path.join(current_dir, "configs/text_encoder/config.json")))
        )
        clip_model.load_state_dict(load_file(clip_path))
        clip_model = clip_model.to(dtype=self.dtype)
        clip_tokenizer = CLIPTokenizer.from_pretrained(os.path.join(current_dir, "configs/tokenizer"))
        
        if auto_t5_gguf:
            t5_model = T5EncoderModel.from_pretrained(
                "calcuis/kontext-gguf",
                gguf_file="t5xxl_fp16-q4_0.gguf",
                torch_dtype=self.dtype,
            )
        else:
            t5_model = T5EncoderModel(
                T5Config(**read_json(os.path.join(current_dir, "configs/text_encoder_2/config.json")))
            )
            t5_model.load_state_dict(load_file(t5_path), strict=False)
            t5_model = t5_model.to(dtype=self.dtype)
        t5_tokenizer = T5TokenizerFast.from_pretrained(os.path.join(current_dir, "configs/tokenizer_2"))

        vae_model = AutoencoderKL(
            **read_json(os.path.join(current_dir, "configs/vae/config.json"))
        )
        vae_model.load_state_dict(load_file(vae_path))
        vae_model = vae_model.to(dtype=self.dtype)

        if transformer_path.endswith('.safetensors') or transformer_path.endswith('.sft'):
            transformer_model = FluxTransformer2DModel(
                **read_json(os.path.join(current_dir, "configs/transformer/config.json"))
            )
            transformer_model.load_state_dict(
                load_file(transformer_path)
            )
            transformer_model = transformer_model.to(dtype=self.dtype)
        elif transformer_path.endswith('.gguf'):
            from diffusers import GGUFQuantizationConfig

            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype)
            transformer_model = FluxTransformer2DModel.from_single_file(
                transformer_path,
                quantization_config=quantization_config,
                torch_dtype=self.dtype,
            )
        else:
            raise NotImplementedError
        
        pipeline = FluxOminiKontextPipeline(
            scheduler=scheduler,
            vae=vae_model,
            text_encoder=clip_model,
            tokenizer=clip_tokenizer,
            text_encoder_2=t5_model,
            tokenizer_2=t5_tokenizer,
            transformer=transformer_model,
            image_encoder=None,
            feature_extractor=None
        ).to(self.device)

        # Enable memory optimizations
        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        
        # Load LoRA if provided
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            pipeline.load_lora_weights(lora_path, adapter_name="omini_kontext")
        
        return (pipeline,)


class OminiKontextPipelineNode:
    """
    ComfyUI node for the Flux Omini Kontext pipeline generation
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
                "reference_image": ("IMAGE",),
                "reference_delta_x": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "reference_delta_y": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "reference_delta_z": ("INT", {"default": 96, "min": -1000, "max": 1000, "step": 1}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "OminiKontext"
    
    def generate(self, pipeline, prompt, reference_image, reference_delta_x, reference_delta_y, 
                 reference_delta_z, steps, guidance_scale, width, height, seed,
                 input_image=None, negative_prompt=None, true_cfg_scale=1.0):
        
        # Convert ComfyUI images to PIL
        # ComfyUI images are [B, H, W, C] in range [0, 1]
        reference_pil = Image.fromarray((reference_image[0].cpu().numpy() * 255).astype(np.uint8))
        
        input_pil = None
        if input_image is not None:
            input_pil = Image.fromarray((input_image[0].cpu().numpy() * 255).astype(np.uint8))
        
        # Set up generator
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare reference delta
        reference_delta = [reference_delta_x, reference_delta_y, reference_delta_z]
        
        # Create ComfyUI progress bar for the UI
        pbar = comfy.utils.ProgressBar(steps)
        
        # Create a callback to update ComfyUI progress during generation
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            # Update ComfyUI progress bar
            pbar.update_absolute(step_index + 1, steps)
            return callback_kwargs
        
        # Generate with both console progress (from diffusers) and UI progress (ComfyUI)
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                image=input_pil,
                reference=reference_pil,
                reference_delta=reference_delta,
                negative_prompt=negative_prompt if negative_prompt else None,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
                callback_on_step_end=progress_callback
            )
        
        # Convert result back to ComfyUI format
        result_np = np.array(result.images[0]).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)


class OminiKontextImageScaleNode:
    """
    Scale images to optimal Kontext resolutions
    """
    
    KONTEXT_RESOLUTIONS = [
        (672, 1568), (688, 1504), (720, 1456), (752, 1392),
        (800, 1328), (832, 1248), (880, 1184), (944, 1104),
        (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
        (1328, 800), (1392, 752), (1456, 720), (1504, 688),
        (1568, 672)
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"
    CATEGORY = "OminiKontext"
    
    def scale(self, image):
        # Get image dimensions
        batch_size, height, width, channels = image.shape
        aspect_ratio = width / height
        
        # Find best matching resolution
        _, best_width, best_height = min(
            (abs(aspect_ratio - w / h), w, h) 
            for w, h in self.KONTEXT_RESOLUTIONS
        )
        
        # Use ComfyUI's built-in upscale function
        scaled = comfy.utils.common_upscale(
            image.movedim(-1, 1),  # [B, H, W, C] -> [B, C, H, W]
            best_width, 
            best_height, 
            "lanczos", 
            "center"
        ).movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]
        
        return (scaled,)


NODE_CLASS_MAPPINGS = {
    "OminiKontextPipelineLoader": OminiKontextPipelineLoaderNode,
    "OminiKontextPipeline": OminiKontextPipelineNode,
    "OminiKontextImageScale": OminiKontextImageScaleNode,
    "OminiKontextSplitPipelineLoader": OminiKontextSplitPipelineLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextPipelineLoader": "Omini Kontext Pipeline Loader",
    "OminiKontextPipeline": "Omini Kontext Pipeline",
    "OminiKontextImageScale": "Omini Kontext Image Scale",
    "OminiKontextSplitPipelineLoader": "Omini Kontext Split Pipeline Loader",
}
