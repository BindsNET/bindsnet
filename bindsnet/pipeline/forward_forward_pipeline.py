import torch
from typing import Dict, Optional, Tuple, Union, Sequence
from abc import ABC, abstractmethod

from bindsnet.network import Network
from bindsnet.network.topology_features import AbstractFeature
from bindsnet.pipeline.base_pipeline import BasePipeline


class BindsNETForwardForwardPipeline(BasePipeline):
    """
    Forward-Forward learning pipeline compatible with topology features.
    
    This pipeline implements the Forward-Forward algorithm using BindsNET's
    topology features framework for feature extraction and learning.
    """
    
    def __init__(
        self,
        network: Network,
        features: Dict[str, AbstractFeature],
        positive_threshold: float = 2.0,
        negative_threshold: float = -2.0,
        learning_rate: float = 0.03,
        time: int = 250,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initialize the Forward-Forward pipeline.
        
        Args:
            network: BindsNET network instance
            features: Dictionary mapping connection names to feature instances
            positive_threshold: Threshold for positive examples
            negative_threshold: Threshold for negative examples
            learning_rate: Learning rate for feature updates
            time: Simulation time
            dt: Time step
        """
        
        self.features = features
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.learning_rate = learning_rate
        self.time = time
        self.dt = dt
        super().__init__(network, **kwargs)
        
    def _initialize_features(self):
        """Initialize all features with the network connections."""
        device = getattr(self.network, 'device', torch.device('cpu'))
        for conn_name, feature in self.features.items():
            if hasattr(self.network, conn_name):
                connection = getattr(self.network, conn_name)
                # Prime the feature with the connection (this sets up the connection reference)
                if hasattr(feature, 'prime_feature'):
                    feature.prime_feature(connection, device)
                
                # Initialize feature value
                if hasattr(feature, 'initialize_value'):
                    feature.initialize_value()
                
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        is_positive: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one step of Forward-Forward learning.
        
        Args:
            inputs: Dictionary of input tensors
            labels: Optional labels for supervised learning
            is_positive: Whether this is a positive or negative example
            
        Returns:
            Dictionary containing network outputs and goodness values
        """
        # Reset network state
        self.network.reset_state_variables()
        
        # Reset feature states
        for feature in self.features.values():
            feature.reset_state_variables()
            
        # Run simulation
        outputs = {}
        goodness_per_layer = {}
        
        for t in range(self.time):
            # Forward pass through network
            self.network.run(inputs, time=1)
            
            # Compute features and goodness for each connection
            for conn_name, feature in self.features.items():
                if hasattr(self.network, conn_name):
                    connection = getattr(self.network, conn_name)
                    
                    # Get connection spikes
                    source_spikes = connection.source.s.float()
                    target_spikes = connection.target.s.float()
                    
                    # Create conn_spikes tensor for feature computation
                    batch_size = source_spikes.shape[0] if source_spikes.dim() > 1 else 1
                    conn_spikes = torch.outer(
                        source_spikes.flatten(), 
                        target_spikes.flatten()
                    ).reshape(batch_size, -1)
                    
                    # Compute feature
                    feature_output = feature.compute(conn_spikes)
                    
                    # Compute goodness (sum of squares of feature activations)
                    goodness = torch.sum(feature_output ** 2, dim=-1)
                    
                    # Store goodness values
                    if conn_name not in goodness_per_layer:
                        goodness_per_layer[conn_name] = []
                    goodness_per_layer[conn_name].append(goodness)
                    
                    # Update connection weights based on Forward-Forward rule
                    self._update_weights(connection, feature_output, goodness, is_positive)
                    
        # Average goodness values over time
        for conn_name in goodness_per_layer:
            goodness_per_layer[conn_name] = torch.stack(goodness_per_layer[conn_name]).mean(0)
            
        self.goodness_values = goodness_per_layer
        
        # Get final network outputs
        for pop_name, population in self.network.layers.items():
            outputs[pop_name] = population.s.clone()
            
        return {
            'outputs': outputs,
            'goodness': goodness_per_layer,
            'total_goodness': sum(goodness_per_layer.values())
        }
        
    def _update_weights(
        self,
        connection,
        feature_output: torch.Tensor,
        goodness: torch.Tensor,
        is_positive: bool
    ):
        """
        Update connection weights using Forward-Forward learning rule.
        
        Args:
            connection: Network connection to update
            feature_output: Output from feature computation
            goodness: Computed goodness values
            is_positive: Whether this is a positive example
        """
        if not hasattr(connection, 'w'):
            return
            
        # Determine target goodness based on example type
        if is_positive:
            target_goodness = self.positive_threshold
        else:
            target_goodness = self.negative_threshold
            
        # Compute goodness error
        goodness_error = goodness - target_goodness
        
        # Update weights to minimize goodness error
        # This is a simplified version - in practice, you'd use more sophisticated updates
        if hasattr(connection, 'update'):
            # Use connection's built-in update mechanism if available
            connection.update()
        else:
            # Manual weight update
            with torch.no_grad():
                weight_update = self.learning_rate * goodness_error.unsqueeze(-1) * feature_output
                connection.w += weight_update.mean(0)  # Average over batch
                
    def train_ff(
        self,
        positive_data: torch.Tensor,
        negative_data: torch.Tensor,
        n_epochs: int = 1,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the network using Forward-Forward learning.
        
        Args:
            positive_data: Positive training examples
            negative_data: Negative training examples
            n_epochs: Number of training epochs
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'positive_goodness': [],
            'negative_goodness': [],
            'goodness_separation': []
        }
        
        for epoch in range(n_epochs):
            epoch_pos_goodness = 0
            epoch_neg_goodness = 0
            
            # Train on positive examples
            for batch in positive_data:
                inputs = {'X': batch}
                result = self.step(inputs, is_positive=True)
                epoch_pos_goodness += result['total_goodness'].item()
                
            # Train on negative examples
            for batch in negative_data:
                inputs = {'X': batch}
                result = self.step(inputs, is_positive=False)
                epoch_neg_goodness += result['total_goodness'].item()
                
            # Calculate average goodness
            avg_pos_goodness = epoch_pos_goodness / len(positive_data)
            avg_neg_goodness = epoch_neg_goodness / len(negative_data)
            goodness_separation = avg_pos_goodness - avg_neg_goodness
            
            metrics['positive_goodness'].append(avg_pos_goodness)
            metrics['negative_goodness'].append(avg_neg_goodness)
            metrics['goodness_separation'].append(goodness_separation)
            
        return metrics
    
    def predict_label_scoring(
        self,
        test_dataset,
        num_classes,
    ) -> torch.Tensor:
        """
        Predict labels for each sample in test_dataset using label scoring.
        For each sample, embed every possible label, run through the network,
        and select the label with the highest total goodness.

        Args:
            test_dataset: Iterable of (data, target) pairs (targets not used for prediction)
            num_classes: Number of possible class labels
            prepend_label_to_image: Function to embed label into input

        Returns:
            Tensor of predicted labels for each sample
        """
        from bindsnet.datasets.contrastive_transforms import prepend_label_to_image

        predictions = []
        for data, _ in test_dataset:
            goodness_scores = []
            for label in range(num_classes):
                sample = prepend_label_to_image(data, label, num_classes)
                goodness = self.compute_goodness(sample.unsqueeze(0))["total_goodness"]
                goodness_scores.append(goodness.item())
            predicted_label = int(torch.tensor(goodness_scores).argmax())
            predictions.append(predicted_label)
        return torch.tensor(predictions)

        
    def reset_state_variables(self):
        """Reset all state variables."""
        super().reset_state_variables()
        for feature in self.features.values():
            feature.reset_state_variables()
        self.goodness_values = {}
    
    def init_fn(self):
        """
        Initialize the pipeline. This method is called by the base pipeline.
        Sets up the features and prepares the network for Forward-Forward learning.
        """
        print("Initializing Forward-Forward pipeline...")
        
        # Initialize features with network connections
        self._initialize_features()
        
        # Set up any additional pipeline-specific initialization
        if hasattr(self.network, 'dt'):
            self.network.dt = self.dt
            
        print(f"Pipeline initialized with {len(self.features)} features")
        print(f"Features: {list(self.features.keys())}")
        
        # Track goodness values for each layer
        self.goodness_values = {}

    def step_(self, batch):
        """
        Run a single step of the pipeline.
        
        Args:
            batch: Input batch data
            
        Returns:
            Step output
        """
        # Extract inputs from batch
        if isinstance(batch, dict):
            inputs = batch
        else:
            # Assume batch is a tensor for input layer
            inputs = {'X': batch}
            
        # Run forward pass
        result = self.step(inputs, is_positive=True)
        return result
    
    def train(self):
        """
        Train the network using Forward-Forward learning.
        This is a placeholder that implements the BasePipeline interface.
        Use train_ff(positive_data, negative_data) for actual Forward-Forward training.
        """
        print("BasePipeline train() method called.")
        print("For Forward-Forward training, use train_ff(positive_data, negative_data) method.")
        return {}
    
    def test(self):
        """
        Test the network.
        This is a placeholder - actual testing should use predict() method with data.
        """
        print("Test method called. Use predict(test_data) for actual testing.")
        return {}
    
    def plots(self, batch, step_out):
        """
        Create plots and logs for a step.
        
        Args:
            batch: Input batch
            step_out: Step output
        """
        # Placeholder implementation
        pass

    def compute_goodness(self, sample: torch.Tensor) -> dict:
        """
        Computes the overall goodness score across the entire network for a given sample,
        using an AbstractSubFeature (e.g., GoodnessScore).

        Args:
            sample: Input tensor for a single sample (shape should match input layer).

        Returns:
            goodness_per_layer: Dictionary of goodness scores per layer, including "total_goodness".
        """
        if not hasattr(self, "goodness_score") or self.goodness_score is None:
            raise RuntimeError("Pipeline must have a 'goodness_score' AbstractSubFeature instance attached as self.goodness_score.")

        # Use the GoodnessScore subfeature to compute goodness
        return self.goodness_score.compute(sample)

    def generate_positive_negative_data(self, train_dataset, num_classes):
        """
        Generate positive and hard negative samples for Forward-Forward training.
        """
        from bindsnet.datasets.contrastive_transforms import prepend_label_to_image

        positive_data = []
        negative_data = []
        for data, target in train_dataset:
            pos_sample = prepend_label_to_image(data, target, num_classes)
            positive_data.append(pos_sample)

            candidate_samples = []
            candidate_goodness = []
            for neg_label in range(num_classes):
                if neg_label == target:
                    continue
                neg_sample = prepend_label_to_image(data, neg_label, num_classes)
                candidate_samples.append(neg_sample)
                goodness = self.goodness_score.compute(sample=neg_sample.unsqueeze(0))["total_goodness"]
                candidate_goodness.append(goodness)
            
            best_idx = int(torch.tensor(candidate_goodness).argmax())
            negative_data.append(candidate_samples[best_idx])
        return positive_data, negative_data


# Example usage and helper functions
def create_ff_pipeline_with_features(
    network: Network,
    feature_configs: Dict[str, Dict],
    **pipeline_kwargs
) -> BindsNETForwardForwardPipeline:
    """
    Helper function to create a Forward-Forward pipeline with specified features.
    
    Args:
        network: BindsNET network
        feature_configs: Dictionary mapping connection names to feature configurations
        **pipeline_kwargs: Additional pipeline arguments
        
    Returns:
        Configured Forward-Forward pipeline
    """
    from ..network.topology_features import (
        ArctangentSurrogateFeature,
    )
    
    # Feature factory
    feature_classes = {
        'arctangent': ArctangentSurrogateFeature,
    }
    
    features = {}
    for conn_name, config in feature_configs.items():
        feature_type = config.pop('type', 'arctangent')
        feature_class = feature_classes[feature_type]
        features[conn_name] = feature_class(**config)
        
    return BindsNETForwardForwardPipeline(
        network=network,
        features=features,
        **pipeline_kwargs
    )