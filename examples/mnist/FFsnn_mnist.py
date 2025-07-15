'''
Think about other users and think about features. add layer with ff and layer without.
'''
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
from bindsnet.network.topology import Connection
from bindsnet.network.topology_features import ArctangentSurrogateFeature, GoodnessScore
from bindsnet.pipeline.forward_forward_pipeline import BindsNETForwardForwardPipeline
from bindsnet.encoding.encodings import repeat as repeat_encoder
from bindsnet.datasets.contrastive_transforms import prepend_label_to_image


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
        
        # Create regular connection (features handled in pipeline)
        source_layer_name = layer_names[i]
        target_layer_name = layer_names[i + 1]
        source_layer = network.layers[source_layer_name]
        target_layer = network.layers[target_layer_name]
        connection = Connection(
            source=source_layer,
            target=target_layer,
            w=torch.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1,  # Small random weights
            wmin=-torch.inf,
            wmax=torch.inf
        )
        connection_name = f"{source_layer_name}_to_{target_layer_name}"
        network.add_connection(connection, source=source_layer_name, target=target_layer_name)
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
    target_labels = [0, 1, 2, 3, 4]
    num_classes = len(target_labels)  # Now 5 classes instead of 10

    
    # Hyperparameters based on Forward-Forward paper for MNIST
    image_feature_size = 784      # MNIST flattened image size
    hidden_sizes = [500, 500]     # Two hidden layers with 500 neurons each
    
    #num_classes = 10              # MNIST classes
    
    # `SNN` parameters
    snn_beta = 0.99              # Neuron decay rate
    snn_threshold = 1.0          # Spike threshold
    alpha_surrogate = 2.0        # Arctangent surrogate gradient parameter
    
    # Training parameters
    simulation_time = 10         # Time steps T
    dt = 1.0                     # Time step size
    learning_rate = 0.001        # Learning rate
    alpha_ff_loss = 0.6          # α in Forward-Forward loss
    batch_size = 64             # Smaller batch for debugging
    num_epochs = 3               # Few epochs for quick test
    
    # Dataset limits for quick testing
    max_train_samples = 10000      # Small dataset for testing
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
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))      
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
    print("\nCreating BindsNET network with topology features...")
    network = create_bindsnet_ff_network(
        input_size=image_feature_size,
        hidden_sizes=hidden_sizes,
        device=device,
        alpha=alpha_surrogate,
        spike_threshold=snn_threshold
    )
    
    print(f"Network layers: {list(network.layers.keys())}")
    print(f"Network connections: {list(network.connections.keys())}")
    
    # Verify connections are regular Connection objects
    for conn_name, connection in network.connections.items():
        print(f"Connection '{conn_name}': {type(connection).__name__}")
        if hasattr(connection, 'w'):
            print(f"  Weight shape: {connection.w.shape}")
    
    # Create features dictionary for the pipeline
    features = {}
    for i in range(len(hidden_sizes)):
        if i == 0:
            source_name = "input"
        else:
            source_name = f"hidden_{i-1}"
        target_name = f"hidden_{i}"
        
        connection_key = f"{source_name}_to_{target_name}"
        features[connection_key] = ArctangentSurrogateFeature(
            name=f"feature_{connection_key}",
            spike_threshold=snn_threshold,
            alpha=alpha_surrogate,
            dt=dt,
            reset_mechanism="subtract"
        )
    
    print(f"Created features: {list(features.keys())}")

    # Create BindsNET Forward-Forward pipeline with topology features
    print("\nCreating BindsNET Forward-Forward pipeline with topology features...")
    pipeline = BindsNETForwardForwardPipeline(
        network=network,
        features=features,
        positive_threshold=2.0,
        negative_threshold=-2.0,
        learning_rate=learning_rate,
        time=simulation_time,
        dt=dt
    )
    
    # Display pipeline information
    print(f"Pipeline features: {list(pipeline.features.keys())}")
    print(f"Pipeline time: {pipeline.time}")
    print(f"Pipeline learning rate: {pipeline.learning_rate}")

    # Prepare positive and negative data for Forward-Forward training
    print("\nPreparing positive and negative data for Forward-Forward training...")
    positive_data = []
    negative_data = []
    goodness_score = GoodnessScore(
    name="goodness",
    parent_feature=features["input_to_hidden_0"],  # Use the feature instance for the first connection
    network=network,  # Pass your network directly
    time=simulation_time,  #You need to provide this
    input_layer="input"
    )
    pipeline.goodness_score = goodness_score  # Attach to pipeline

    # Generate positive and negative data using the pipeline method
    positive_data, negative_data = pipeline.generate_positive_negative_data(
        train_dataset, num_classes
    )

    print(f"Training with {len(positive_data)} positive and {len(negative_data)} negative samples")

    # Train using Forward-Forward algorithm
    print("\nStarting Forward-Forward training...")


    metrics = pipeline.train_ff(
        positive_data=positive_data,
        negative_data=negative_data,
        n_epochs=num_epochs
    )
    print(f"Training metrics: {metrics}")
    print("Training complete.")

    # Predict on test set using label scoring
    print("\nStarting prediction on test set...")

    predicted_labels = pipeline.predict_label_scoring(
        test_dataset,
        num_classes,
    )
    # Calculate percentage correct
    true_labels = torch.tensor([target for _, target in test_dataset])
    num_correct = (predicted_labels == true_labels).sum().item()
    percent_correct = 100.0 * num_correct / len(true_labels) if len(true_labels) > 0 else 0.0
    print(f"Percentage correct: {percent_correct:.2f}% ({num_correct}/{len(true_labels)})")
    print("Prediction complete.")

if __name__ == "__main__":
    main()