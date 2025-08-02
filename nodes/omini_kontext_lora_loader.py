import torch
import os
import folder_paths
import comfy.utils
import sys

# Add the omini-kontext source to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'omini_kontext_src'))

from pipeline_flux_omini_kontext import FluxOminiKontextPipeline


class OminiKontextLoRALoaderNode:
    """
    ComfyUI node for loading Omini Kontext LoRA weights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get LoRA files from ComfyUI's loras folder
        lora_files = folder_paths.get_filename_list("loras")
        lora_files = [f for f in lora_files if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt'))]
        
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "lora_name": (lora_files + ["custom_path"],),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "custom_lora_path": ("STRING", {"default": "", "multiline": False}),
                "adapter_name": ("STRING", {"default": "omini_kontext", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("OMINI_KONTEXT_PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "OminiKontext"
    
    def load_lora(self, pipeline, lora_name, strength, custom_lora_path="", adapter_name="omini_kontext"):
        # Determine the LoRA path
        if lora_name == "custom_path" and custom_lora_path:
            lora_path = custom_lora_path
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        print(f"Loading LoRA weights from: {lora_path}")
        print(f"Adapter name: {adapter_name}, Strength: {strength}")
        
        # Load LoRA weights
        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        
        # Set LoRA scale
        pipeline.set_adapters([adapter_name], adapter_weights=[strength])
        
        return (pipeline,)


class OminiKontextLoRAUnloadNode:
    """
    ComfyUI node for unloading Omini Kontext LoRA weights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "adapter_name": ("STRING", {"default": "omini_kontext", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("OMINI_KONTEXT_PIPELINE",)
    FUNCTION = "unload_lora"
    CATEGORY = "OminiKontext"
    
    def unload_lora(self, pipeline, adapter_name="omini_kontext"):
        print(f"Unloading LoRA adapter: {adapter_name}")
        
        try:
            # Delete the adapter
            pipeline.delete_adapters([adapter_name])
        except Exception as e:
            print(f"Warning: Could not unload adapter {adapter_name}: {e}")
        
        return (pipeline,)


class OminiKontextLoRAMergeNode:
    """
    ComfyUI node for merging multiple LoRA adapters
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("OMINI_KONTEXT_PIPELINE",),
                "adapter_names": ("STRING", {"default": "adapter1,adapter2", "multiline": False}),
                "adapter_weights": ("STRING", {"default": "1.0,1.0", "multiline": False}),
                "combination_type": (["linear", "cat"], {"default": "linear"}),
            }
        }
    
    RETURN_TYPES = ("OMINI_KONTEXT_PIPELINE",)
    FUNCTION = "merge_lora"
    CATEGORY = "OminiKontext"
    
    def merge_lora(self, pipeline, adapter_names, adapter_weights, combination_type="linear"):
        # Parse adapter names and weights
        names = [name.strip() for name in adapter_names.split(",")]
        weights = [float(w.strip()) for w in adapter_weights.split(",")]
        
        if len(names) != len(weights):
            raise ValueError("Number of adapter names must match number of weights")
        
        print(f"Merging LoRA adapters: {names} with weights: {weights}")
        print(f"Combination type: {combination_type}")
        
        # Set multiple adapters
        pipeline.set_adapters(names, adapter_weights=weights)
        
        return (pipeline,)


NODE_CLASS_MAPPINGS = {
    "OminiKontextLoRALoader": OminiKontextLoRALoaderNode,
    "OminiKontextLoRAUnload": OminiKontextLoRAUnloadNode,
    "OminiKontextLoRAMerge": OminiKontextLoRAMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextLoRALoader": "Omini Kontext LoRA Loader",
    "OminiKontextLoRAUnload": "Omini Kontext LoRA Unload",
    "OminiKontextLoRAMerge": "Omini Kontext LoRA Merge",
}