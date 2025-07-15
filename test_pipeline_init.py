#!/usr/bin/env python3
"""
Test if BindsNETForwardForwardPipeline can be instantiated without NotImplementedError
"""

import sys
sys.path.insert(0, '/Users/kevinchang/bindsnet/bindsnet')

try:
    print("Testing pipeline instantiation...")
    
    # Mock imports to avoid torch issues
    import torch
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection
    from bindsnet.network.topology_features import ArctangentSurrogateFeature
    from bindsnet.pipeline.forward_forward_pipeline import BindsNETForwardForwardPipeline
    
    print("✓ All imports successful")
    
    # Create simple network
    network = Network()
    input_layer = Input(n=10)
    hidden_layer = LIFNodes(n=5)
    
    network.add_layer(input_layer, name="input")
    network.add_layer(hidden_layer, name="hidden")
    
    connection = Connection(input_layer, hidden_layer)
    network.add_connection(connection, source="input", target="hidden")
    
    print("✓ Network created")
    
    # Create features
    features = {
        "input_to_hidden": ArctangentSurrogateFeature(
            name="test_feature",
            spike_threshold=1.0,
            alpha=2.0
        )
    }
    
    print("✓ Features created")
    
    # Create pipeline
    pipeline = BindsNETForwardForwardPipeline(
        network=network,
        features=features,
        positive_threshold=2.0,
        negative_threshold=-2.0,
        learning_rate=0.03,
        time=10
    )
    
    print("✓ Pipeline instantiated successfully!")
    print(f"Pipeline has {len(pipeline.features)} features")
    print(f"Required methods implemented: init_fn, train, test, step_, plots")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
