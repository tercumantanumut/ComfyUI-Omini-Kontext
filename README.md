# ComfyUI-Omini-Kontext

Wrapper ComfyUI integration for the [Flux Omini Kontext](https://github.com/Saquib764/omini-kontext) pipeline, enabling seamless character/object insertion into scenes using FLUX.1-Kontext-dev with LoRA adaptation.

## Features

- **Character/Object Insertion**: Insert reference images into scenes with precise spatial control
- **LoRA Support**: Load and use pre-trained LoRA weights for specific insertion tasks
- **Memory Optimization**: Built-in VAE slicing and tiling for efficient VRAM usage
- **Flexible Pipeline**: Support for both text-to-image and image-to-image workflows
- **Position Control**: Fine-tune object placement with reference_delta parameters

## Installation

1. **Clone the repository into your ComfyUI custom_nodes folder:**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/tercumantanumut/ComfyUI-Omini-Kontext.git
   ```

2. **Install dependencies:**
   ```bash
   cd ComfyUI-Omini-Kontext
   pip install -r requirements.txt
   ```

3. **Download the base model (optional - will auto-download on first use):**
   - The pipeline uses `black-forest-labs/FLUX.1-Kontext-dev` by default
   - Requires HuggingFace login: `huggingface-cli login`

4. **Download pre-trained LoRA weights (optional):**
   ```bash
   # Example: Character insertion LoRA
   wget https://huggingface.co/saquiboye/omini-kontext-character/resolve/main/character_5000.safetensors \
        -O ComfyUI/models/loras/omini_kontext_character_5000.safetensors
   ```

## Available Nodes

### 1. Omini Kontext Pipeline Loader
Loads the Flux Omini Kontext pipeline with optional LoRA weights.
- **Inputs:**
  - `model_path`: HuggingFace model ID or local path
  - `lora_path`: Optional path to LoRA weights
- **Output:** `OMINI_KONTEXT_PIPELINE`

### 2. Omini Kontext Pipeline
Main generation node for character/object insertion.
- **Required Inputs:**
  - `pipeline`: Loaded pipeline from loader node
  - `prompt`: Text description
  - `reference_image`: Character/object to insert
  - `reference_delta_x/y/z`: Position control (default: 0, 0, 96)
  - Generation parameters (steps, guidance_scale, width, height, seed)
- **Optional Inputs:**
  - `input_image`: Base image for img2img mode
  - `negative_prompt`: Negative text prompt
  - `true_cfg_scale`: Additional CFG control
- **Output:** Generated image

### 3. Omini Kontext Image Scale
Scales images to optimal Kontext resolutions.
- **Input:** Any image
- **Output:** Scaled image at optimal resolution

### 4. Omini Kontext LoRA Loader
Load LoRA weights into an existing pipeline.
- **Inputs:**
  - `pipeline`: Pipeline to add LoRA to
  - `lora_name`: LoRA file from models/loras folder
  - `strength`: LoRA strength multiplier
  - `adapter_name`: Name for the adapter

### 5. Advanced Encoder Nodes
For advanced workflows:
- **Image Encoder**: Encode images to latents
- **Text Encoder**: Encode prompts to embeddings
- **Reference Encoder**: Encode reference with position delta
- **Latent Combiner**: Combine input and reference latents

## Basic Workflow

1. **Load Pipeline**: Use "Omini Kontext Pipeline Loader" with model path
2. **Load LoRA** (optional): Use "Omini Kontext LoRA Loader" 
3. **Prepare Images**: 
   - Load your input image (optional)
   - Load your reference character/object image
   - Optionally scale with "Omini Kontext Image Scale"
4. **Generate**: Connect everything to "Omini Kontext Pipeline" node
5. **Save Result**: Use standard ComfyUI save image node

## Example Use Cases

### Character Insertion
Insert a specific character into various scenes:
```
reference_delta = [0, 0, 96]  # Standard positioning
prompt = "A boy playing in a sunny park"
```

### Object Placement
Place objects with spatial control:
```
reference_delta = [50, 0, 96]  # Shift right
prompt = "A vintage car parked on a city street"
```

### Style Transfer
Combine reference style with scene:
```
reference_delta = [0, 0, 48]  # Closer integration
prompt = "In the style of the reference"
```

## Tips

1. **Reference Delta Values**:
   - X: Horizontal position (-100 to 100 typical)
   - Y: Vertical position (-100 to 100 typical)  
   - Z: Depth/integration (48-144 typical, 96 default)

2. **Memory Management**:
   - Pipeline automatically enables VAE slicing/tiling
   - For 24GB VRAM: up to 1024x1024 generation
   - For 16GB VRAM: recommended 768x768 or lower

3. **LoRA Strength**:
   - 1.0 = full strength (default)
   - 0.5-0.8 = subtle effect
   - 1.2-1.5 = stronger effect

## Troubleshooting

### "No module named 'diffusers'"
Run: `pip install git+https://github.com/huggingface/diffusers.git`

### "CUDA out of memory"
- Reduce generation resolution
- Close other GPU applications
- Enable CPU offloading (future feature)

### "401 Unauthorized" when loading model
Run: `huggingface-cli login` and enter your HuggingFace token

## Credits

- Original Omini-Kontext implementation: [Saquib764/omini-kontext](https://github.com/Saquib764/omini-kontext)
- Based on FLUX.1-Kontext-dev by Black Forest Labs
- ComfyUI integration by ogkai (github: tercumantanumut)

## License

This project follows the same license as the original omini-kontext repository.
