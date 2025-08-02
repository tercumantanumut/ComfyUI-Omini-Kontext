"""
ComfyUI-OminiKontext
Custom nodes for using the Flux Omini Kontext pipeline in ComfyUI
"""

import os
import sys

# Add nodes directory to path
nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
if nodes_dir not in sys.path:
    sys.path.append(nodes_dir)

# Import all node modules with proper package handling
from .nodes.omini_kontext_pipeline import NODE_CLASS_MAPPINGS as pipeline_mappings
from .nodes.omini_kontext_pipeline import NODE_DISPLAY_NAME_MAPPINGS as pipeline_display_mappings
from .nodes.omini_kontext_lora_loader import NODE_CLASS_MAPPINGS as lora_mappings
from .nodes.omini_kontext_lora_loader import NODE_DISPLAY_NAME_MAPPINGS as lora_display_mappings
from .nodes.omini_kontext_encoder import NODE_CLASS_MAPPINGS as encoder_mappings
from .nodes.omini_kontext_encoder import NODE_DISPLAY_NAME_MAPPINGS as encoder_display_mappings

# Combine all mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add all node mappings
NODE_CLASS_MAPPINGS.update(pipeline_mappings)
NODE_CLASS_MAPPINGS.update(lora_mappings)
NODE_CLASS_MAPPINGS.update(encoder_mappings)

NODE_DISPLAY_NAME_MAPPINGS.update(pipeline_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(lora_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(encoder_display_mappings)

# Set web directory for custom UI elements (if any)
WEB_DIRECTORY = "./web"

# Metadata
__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("=" * 60)
print("ComfyUI-OminiKontext loaded successfully!")
print(f"Version: {__version__}")
print(f"Available nodes: {len(NODE_CLASS_MAPPINGS)}")
for node_name in NODE_CLASS_MAPPINGS:
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)}")
print("=" * 60)