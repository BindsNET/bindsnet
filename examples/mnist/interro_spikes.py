import torch

def load_and_inspect_spike_history(file_path):
    # Load the spike_history from the .pt file
    spike_history = torch.load(file_path)
    
    # Print general information about the spike_history object
    print("Type of spike_history:", type(spike_history))
    print("Total steps recorded:", len(spike_history))
    
    # Check if spike_history is a dictionary and print details
    if isinstance(spike_history, dict):
        # Print information about a few steps to see the structure
        sample_keys = list(spike_history.keys())[:5]  # Get the first 5 keys as samples
        print("Sample keys (steps):", sample_keys)
        
        # Inspect the content of the first few steps
        for key in sample_keys:
            step_data = spike_history[key]
            print(f"\nData for step {key}:")
            print("  Type of data:", type(step_data))
            
            # Assuming step_data is a dictionary {layer_name: spike_data_tensor}
            if isinstance(step_data, dict):
                for layer, spikes in step_data.items():
                    print(f"    Layer: {layer}")
                    print("    Type of spikes:", type(spikes))
                    if isinstance(spikes, torch.Tensor):
                        print("    Shape of spike tensor:", spikes.shape)
                        print("    Dtype of spike tensor:", spikes.dtype)
                        print("    Device of spike tensor:", spikes.device)
                    else:
                        print("    Spike data is not a tensor")
            else:
                print("  Step data is not organized as expected (dict of tensors)")
    else:
        print("spike_history is not a dictionary as expected")

# Path to the spike_history file
file_path = 'spike_history.pt'

# Call the function to load and inspect the spike_history
load_and_inspect_spike_history(file_path)

