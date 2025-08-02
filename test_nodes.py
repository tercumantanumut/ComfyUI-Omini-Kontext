#!/usr/bin/env python3
"""
Test script to validate ComfyUI-OminiKontext nodes
"""

import sys
import os

# Add ComfyUI to path
sys.path.append('/home/ubuntu/ComfyUI')

# Import the nodes
try:
    from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("✓ Successfully imported node mappings")
    print(f"\nAvailable nodes ({len(NODE_CLASS_MAPPINGS)}):")
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        print(f"  - {display_name} ({node_name})")
        
    # Test instantiation
    print("\nTesting node instantiation:")
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            instance = node_class()
            print(f"  ✓ {node_name}")
        except Exception as e:
            print(f"  ✗ {node_name}: {e}")
            
    # Test INPUT_TYPES
    print("\nTesting INPUT_TYPES:")
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            input_types = node_class.INPUT_TYPES()
            print(f"  ✓ {node_name}")
            if "required" in input_types:
                print(f"    Required inputs: {list(input_types['required'].keys())}")
            if "optional" in input_types:
                print(f"    Optional inputs: {list(input_types['optional'].keys())}")
        except Exception as e:
            print(f"  ✗ {node_name}: {e}")
            
except Exception as e:
    print(f"✗ Failed to import nodes: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")