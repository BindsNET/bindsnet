import torch
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys

# Add the parent directory of 'bindsnet' to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import ForwardForwardConnection
from bindsnet.pipeline.forward_forward_pipeline import BindsNETForwardForwardPipeline
from bindsnet.encoding.encodings import repeat as repeat_encoder


def create_bindsnet_ff_network(input_size: int, hidden_sizes: list, device: torch.device, 
                               alpha: float = 2.0, spike_threshold: float = 1.0) -> Network:
    """
    Create a BindsNET network with ForwardForwardConnections for Forward-Forward training.
    
    :param input_size: Size of input layer (e.g., 784 for MNIST)
    :param hidden_sizes: List of hidden layer sizes [500, 500]
    :param device: Device to run on
    :param alpha: Arctangent surrogate gradient parameter
    :param spike_threshold: Spike threshold for neurons
    :return: BindsNET Network with FF connections
    """
    network = Network(dt=1.0)
    
    # Create input layer
    input_layer = Input(n=input_size, name="input")
    network.add_layer(input_layer, name="input")
    
    # Create hidden layers
    layer_sizes = [input_size] + hidden_sizes
    layer_names = ["input"] + [f"hidden_{i}" for i in range(len(hidden_sizes))]
    
    for i in range(len(hidden_sizes)):
        # Create LIF hidden layer
        hidden_layer = LIFNodes(
            n=hidden_sizes[i], 
            decay=0.99,  # β = 0.99
            thresh=spike_threshold,
            reset=0.0,
            name=f"hidden_{i}"
        )
        network.add_layer(hidden_layer, name=f"hidden_{i}")
        
        # Create ForwardForwardConnection between layers
        source_layer_name = layer_names[i]
        target_layer_name = layer_names[i + 1]
        source_layer = network.layers[source_layer_name]
        target_layer = network.layers[target_layer_name]
        
        ff_connection = ForwardForwardConnection(
            source=source_layer,
            target=target_layer,
            spike_threshold=spike_threshold,
            alpha=alpha,
            w=torch.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1,  # Small random weights
            wmin=-torch.inf,
            wmax=torch.inf
        )
        
        connection_name = f"{source_layer_name}_to_{target_layer_name}"
        network.add_connection(ff_connection, source=source_layer_name, target=target_layer_name)
    
    return network
def filter_dataset_by_labels(dataset, target_labels):
    """
    Filter dataset to only include samples with specified labels.
    
    :param dataset: Original dataset
    :param target_labels: List of labels to keep (e.g., [0, 1, 2, 3, 4])
    :return: Filtered dataset
    """
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in target_labels:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

def main():
    """
    BindsNET Forward-Forward MNIST training example with surrogate gradients.
    """
    print("BindsNET Forward-Forward MNIST Example")
    print("=====================================")

    #Modifying the Dataset to only include classes 0-4
    target_labels = [0, 1, 2, 3, 4,5,6,7,8,9]
    num_classes = len(target_labels)  # Now 5 classes instead of 10

    
    # Hyperparameters based on Forward-Forward paper for MNIST
    image_feature_size = 784      # MNIST flattened image size
    hidden_sizes = [500, 500]     # Two hidden layers with 500 neurons each
    
    #num_classes = 10              # MNIST classes
    
    # SNN parameters
    snn_beta = 0.99              # Neuron decay rate
    snn_threshold = 1.0          # Spike threshold
    alpha_surrogate = 2.0        # Arctangent surrogate gradient parameter
    
    # Training parameters
    simulation_time = 10         # Time steps T
    dt = 1.0                     # Time step size
    learning_rate = 0.001        # Learning rate
    alpha_ff_loss = 0.6          # α in Forward-Forward loss
    batch_size = 64             # Smaller batch for debugging
    num_epochs = 10               # Few epochs for quick test
    
    # Dataset limits for quick testing
    max_train_samples = 2024      # Small dataset for testing
    max_test_samples = 512        # Small test set
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    print("\nPreparing MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
        transforms.Lambda(lambda x: x.view(-1))       # Flatten to 784
    ])
    
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    filtered_train = filter_dataset_by_labels(full_train, target_labels)
    filtered_test = filter_dataset_by_labels(full_test, target_labels)

    print(f"Filtered train dataset size: {len(filtered_train)}")
    print(f"Filtered test dataset size: {len(filtered_test)}")

    # Take subset of filtered data
    train_subset_size = min(max_train_samples, len(filtered_train))
    test_subset_size = min(max_test_samples, len(filtered_test))
    
    train_dataset = torch.utils.data.Subset(filtered_train, range(train_subset_size))
    test_dataset = torch.utils.data.Subset(filtered_test, range(test_subset_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Expected batches per epoch: {len(train_dataset) // batch_size}")
    
    # Create BindsNET network with ForwardForwardConnections
    print("\nCreating BindsNET network with ForwardForwardConnections...")
    network = create_bindsnet_ff_network(
        input_size=image_feature_size,
        hidden_sizes=hidden_sizes,
        device=device,
        alpha=alpha_surrogate,
        spike_threshold=snn_threshold
    )
    
    print(f"Network layers: {list(network.layers.keys())}")
    print(f"Network connections: {list(network.connections.keys())}")
    
    # Verify connections are ForwardForwardConnection
    for conn_name, connection in network.connections.items():
        print(f"Connection '{conn_name}': {type(connection).__name__}")
        if hasattr(connection, 'get_surrogate_info'):
            info = connection.get_surrogate_info()
            print(f"  Surrogate: {info['surrogate_type']}, α={info['alpha']}, threshold={info['spike_threshold']}")
    
    # Define FF layer pairs for training
    ff_pairs_specs = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            source_name = "input"
        else:
            source_name = f"hidden_{i-1}"
        target_name = f"hidden_{i}"
        
        # FIX: Use tuple keys to match how BindsNET actually stores connections
        connection_key = (source_name, target_name)  # ('input', 'hidden_0'), ('hidden_0', 'hidden_1')
        ff_pairs_specs.append((connection_key, target_name))
    
    print(f"FF layer pairs: {ff_pairs_specs}")
    
    # Create encoder function
    def wrapped_repeat_encoder(datum: torch.Tensor) -> torch.Tensor:
        """Encode input data for SNN simulation."""
        return repeat_encoder(datum=datum, time=simulation_time, dt=dt)
    
    # Create BindsNET Forward-Forward pipeline
    print("\nCreating BindsNET Forward-Forward pipeline with surrogate gradients...")
    pipeline = BindsNETForwardForwardPipeline(
        network=network,
        train_ds=train_dataset,
        num_classes=num_classes,
        encoder=wrapped_repeat_encoder,
        time=simulation_time,
        ff_pairs_specs=ff_pairs_specs,
        input_layer_name="input",
        lr=learning_rate,
        alpha_loss=alpha_ff_loss,
        alpha=alpha_surrogate,
        spike_threshold=snn_threshold,
        optimizer_cls=optim.Adam,
        device=device,
        dt=dt,
        batch_size=batch_size,
        num_epochs=num_epochs,
        test_ds=test_dataset
    )
    
    # Display pipeline information
    surrogate_info = pipeline.get_surrogate_info()
    print(f"Pipeline surrogate info: {surrogate_info}")
    
    # Training
    print("\n" + "="*50)
    print("STARTING BINDSNET FORWARD-FORWARD TRAINING")
    print("="*50)
    try:
        pipeline.train()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Testing
    print("\n" + "="*30)
    print("STARTING TESTING")
    print("="*30)
    try:
        pipeline.test_epoch()
        print("\nTesting completed successfully!")
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*50)
    print("BINDSNET FORWARD-FORWARD MNIST EXAMPLE FINISHED")
    print("="*50)
    
    # Display final network statistics
    print("\nFinal Network Statistics:")
    for conn_name, connection in network.connections.items():
        if hasattr(connection, 'w'):
            weight_stats = {
                'mean': connection.w.mean().item(),
                'std': connection.w.std().item(),
                'min': connection.w.min().item(),
                'max': connection.w.max().item()
            }
            print(f"Connection '{conn_name}' weights: {weight_stats}")

if __name__ == "__main__":
    main()