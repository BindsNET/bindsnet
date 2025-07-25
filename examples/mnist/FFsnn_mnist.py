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
from bindsnet.network.topology_features import ArctangentSurrogateFeature, GoodnessScore, BatchNormalization
from bindsnet.pipeline.forward_forward_pipeline import BindsNETForwardForwardPipeline
from bindsnet.encoding.encodings import bernoulli
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
        # Create LIF hidden layer with BindsNET standard voltage scale
        hidden_layer = LIFNodes(
            n=hidden_sizes[i], 
            decay=0.99,        # β = 0.99 (standard decay)
            thresh=-52.0,      # Standard BindsNET threshold
            rest=-65.0,        # Standard BindsNET rest potential  
            reset=-60.0,       # Reset slightly above rest
            refrac=5,          # 5ms refractory period
            tc_decay=100.0,    # Standard membrane time constant
            name=f"hidden_{i}"
        )
        network.add_layer(hidden_layer, name=f"hidden_{i}")
        
        # Create connection with larger weights to encourage spiking
        source_layer_name = layer_names[i]
        target_layer_name = layer_names[i + 1]
        source_layer = network.layers[source_layer_name]
        target_layer = network.layers[target_layer_name]
        connection = Connection(
            source=source_layer,
            target=target_layer,
            w=torch.randn(layer_sizes[i], layer_sizes[i + 1]) * 5.0,  # Larger weights for BindsNET voltage scale
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
    max_train_samples = 500      # Small dataset for testing
    max_test_samples = 512        # Small test set
    
    # Device setup
    device = torch.device("cpu")  # Use CPU to avoid MPS issues during debugging

    # Intensity scaling for Poisson encoding (like other BindsNET examples)
    intensity = 128.0  # Standard BindsNET intensity for MNIST

    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    print("\nPreparing MNIST dataset...")
    
    # Use BindsNET-compatible transform (no normalization, intensity scaling)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * intensity),  # Scale for Poisson encoding
        transforms.Lambda(lambda x: x.view(-1))      # Flatten to [784]
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

    print(f"Actual network connection keys: {list(network.connections.keys())}")
    # Explicitly create features dict for each connection
    features = {}
    for connection_key, connection in network.connections.items():
        # Create a simple feature that just holds layer activations
        feature = ArctangentSurrogateFeature(
            name=f"feature_{connection_key}",
            spike_threshold=snn_threshold,
            alpha=alpha_surrogate,
            dt=dt,
            reset_mechanism="subtract"
        )
        feature.prime_feature(connection, device)
        feature.initialize_value()
        
        # DISABLED: Batch normalization interferes with sparse spike data in Forward-Forward learning
        # The normalization to zero mean makes goodness scores (sum of squared activations) near zero
        # Re-enable batch normalization now that the network produces spikes
        # Add batch normalization - determine size from target layer
        # target_layer_size = connection.target.n
        # feature.batch_norm = BatchNormalization(
        #     name=f"batch_norm_{connection_key}",
        #     parent_feature=feature,
        #     eps=1e-5,
        #     affine=True,
        #     per_timestep=True  # Enable per-timestep normalization for Forward-Forward
        # )
        # # Manually initialize BatchNorm with known size
        # feature.batch_norm._init_bn(target_layer_size)
        # feature.batch_norm.bn.to(device)
        # # Enable gradients for batch norm parameters
        # for param in feature.batch_norm.bn.parameters():
        #     param.requires_grad_(True)
        
        features[str(connection_key)] = feature
    print(f"Created features for: {list(features.keys())}")
    print("\nCreating BindsNET Forward-Forward pipeline with topology features...")
    pipeline = BindsNETForwardForwardPipeline(
        network=network,
        features=features,
        positive_threshold=2.0,
        negative_threshold=-2.0,
        learning_rate=learning_rate,
        time=simulation_time,
        dt=dt,
        alpha_ff_loss=alpha_ff_loss
    )

    # Use the first feature as the parent_feature for GoodnessScore
    first_feature = next(iter(features.values()))
    pipeline.goodness_score = GoodnessScore(
        name="goodness",
        parent_feature=first_feature,
        network=network,
        time=simulation_time,
        input_layer="input"
    )
    print("GoodnessScore attached to pipeline.")

    # Display pipeline information
    print(f"Pipeline time: {pipeline.time}")
    print(f"Pipeline learning rate: {pipeline.learning_rate}")

    # Prepare positive and negative data for Forward-Forward training
    print("\nPreparing positive and negative data for Forward-Forward training...")

    # Generate positive and negative data using the pipeline method
    positive_data, negative_data = pipeline.generate_positive_negative_data(
        train_dataset, num_classes
    )

    print(f"Training with {len(positive_data)} positive and {len(negative_data)} negative samples")

    # Train using Forward-Forward algorithm
    print("\nStarting Forward-Forward training...")

    # Let the pipeline handle the optimizer creation
    metrics = pipeline.train_ff(
        positive_data=positive_data,
        negative_data=negative_data,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=None  # Let pipeline create optimizer
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