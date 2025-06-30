#!/usr/bin/env python3
"""
Test script for ArctangentSurrogateFeature
"""

import torch
import sys
import os

# Add the bindsnet directory to path
sys.path.insert(0, '/Users/kevinchang/Downloads/bindsnet')

try:
    from bindsnet.network.topology_features import ArctangentSurrogateFeature
    print("✅ ArctangentSurrogateFeature imported successfully")
    
    # Test creating the feature
    feature = ArctangentSurrogateFeature(
        spike_threshold=1.0,
        alpha=2.0,
        dt=1.0,
        reset_mechanism="subtract"
    )
    print("✅ ArctangentSurrogateFeature created successfully")
    print(f"Feature info: {feature}")
    
    # Test basic functionality
    print(f"Spike threshold: {feature.spike_threshold}")
    print(f"Alpha: {feature.alpha}")
    print(f"Reset mechanism: {feature.reset_mechanism}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
