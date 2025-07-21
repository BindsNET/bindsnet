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
        
    def _update_weights(
        self,
        connection,
        feature_output: torch.Tensor,
        goodness: torch.Tensor,
        is_positive: bool
    ):
        """
        Update connection weights using Forward-Forward learning rule.
        Delegates to the ForwardForwardUpdate subfeature if available, otherwise uses feature's update_weights.
        Uses self.alpha_ff_loss if provided, otherwise falls back to feature's alpha.
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

        # Find the feature for this connection
        feature = None
        for conn_name, f in self.features.items():
            if hasattr(self.network, conn_name) and getattr(self.network, conn_name) is connection:
                feature = f
                break

        # Try to delegate to ForwardForwardUpdate subfeature if available
        ff_update = getattr(feature, 'ff_update', None)
        # Use alpha_ff_loss if set, otherwise fall back to feature's alpha or default
        alpha = self.alpha_ff_loss if self.alpha_ff_loss is not None else getattr(feature, 'alpha', 2.0)
        if ff_update is not None and hasattr(ff_update, 'update_weights'):
            ff_update.update_weights(
                connection,
                feature_output,
                goodness,
                goodness_error,
                is_positive,
                learning_rate=self.learning_rate,
                alpha=alpha
            )
    
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

    def step_(self, batch, **kwargs):
        """
        Run a single step of the pipeline: forward pass, feature/goodness computation, and weight update.
        At every time step t, normalization is performed for every layer (if the feature has a BatchNormalization subfeature).
        Args:
            batch: Input batch data
            **kwargs: Additional arguments (e.g., is_positive)
        Returns:
            dict with outputs, goodness, and total_goodness
        """
        is_positive = kwargs.get('is_positive', True)
        # Prepare inputs - encode as spikes using bernoulli
        if isinstance(batch, dict):
            if 'input' in batch:
                input_data = batch['input']
                encoded_batch = {'input': input_data}
            else:
                encoded_batch = batch
        else:
            # Convert batch to spikes using bernoulli encoding
            from bindsnet.encoding.encodings import bernoulli
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
            encoded_batch = {'input': batch}

        # Reset state variables for network and features
        self.network.reset_state_variables()
        for feature in self.features.values():
            feature.reset_state_variables()

        # Run the network for self.time steps and extract features
        outputs = {pop_name: [] for pop_name in self.network.layers}
        layer_activities = {layer_name: [] for layer_name in self.network.layers}
        
        for t in range(self.time):
            # For each timestep, generate spikes from the batch using bernoulli sampling
            if 'input' in encoded_batch:
                # Generate spikes for this timestep
                input_data = encoded_batch['input']
                
                # Ensure input data is in [0, 1] range for bernoulli sampling
                input_probs = torch.clamp(input_data, 0.0, 1.0)
                spike_input = torch.bernoulli(input_probs)
                timestep_inputs = {'input': spike_input}
            else:
                timestep_inputs = encoded_batch
            
            self.network.run(timestep_inputs, time=1)
            
            # Collect layer activities for feature extraction
            for layer_name, layer in self.network.layers.items():
                layer_activities[layer_name].append(layer.s.clone())
            
            # Collect outputs for each layer
            for pop_name, population in self.network.layers.items():
                outputs[pop_name].append(population.s.clone())

        # After simulation, extract features and apply batch normalization
        # Also compute goodness from normalized features
        total_goodness = torch.tensor(0.0, requires_grad=True)
        goodness_dict = {}
        
        for conn_key, feature in self.features.items():
            # Get connection to determine source and target layers
            connection = None
            for name, conn in self.network.connections.items():
                if str(name) == str(conn_key):
                    connection = conn
                    break
            
            if connection is not None:
                # Get target layer activity (sum over time)
                target_layer_name = None
                for layer_name, layer in self.network.layers.items():
                    if layer is connection.target:
                        target_layer_name = layer_name
                        break
                
                if target_layer_name is not None:
                    # Sum target layer activity over time to get feature representation
                    target_activity = torch.stack(layer_activities[target_layer_name], dim=0)  # [time, batch, neurons]
                    target_activity_sum = target_activity.sum(dim=0).float()  # [batch, neurons] - convert to float
                    
                    # Ensure proper batch dimension
                    if target_activity_sum.dim() == 1:
                        target_activity_sum = target_activity_sum.unsqueeze(0)
                    
                    # Set feature value for batch normalization (with gradients)
                    feature.value = target_activity_sum.requires_grad_(True)
                    
                    # Apply batch normalization if present
                    batch_norm = getattr(feature, 'batch_norm', None)
                    if batch_norm is not None:
                        normalized_activity = batch_norm.batch_normalize()
                        # Update feature value with normalized activity
                        feature.value = normalized_activity
                    
                    # Compute goodness from the (possibly normalized) feature values
                    feature_goodness = (feature.value ** 2).sum()
                    goodness_dict[f"layer_{target_layer_name}"] = feature_goodness
                    total_goodness = total_goodness + feature_goodness

        # Add total goodness
        goodness_dict["total_goodness"] = total_goodness

        # Stack outputs for each layer
        outputs_stacked = {pop_name: torch.stack(s_list, dim=0) for pop_name, s_list in outputs.items()}
        return {
            'outputs': outputs_stacked,
            'goodness': {k: v for k, v in goodness_dict.items() if k != 'total_goodness'},
            'total_goodness': goodness_dict['total_goodness']
        }

    def get_learnable_parameters(self):
        """
        Get all learnable parameters from features (e.g., BatchNorm1d parameters).
        
        Returns:
            List of torch.nn.Parameter objects that can be optimized
        """
        params = []
        for feature in self.features.values():
            if hasattr(feature, 'batch_norm') and hasattr(feature.batch_norm, 'bn'):
                bn_params = list(feature.batch_norm.bn.parameters())
                params.extend(bn_params)
        return params

    def train_ff(
        self,
        positive_data: torch.Tensor,
        negative_data: torch.Tensor,
        n_epochs: int = 1,
        batch_size: int = 64,
        optimizer: torch.optim.Optimizer = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the network using Forward-Forward learning and update learnable parameters (e.g., gamma, beta).
        If optimizer is not provided, creates Adam optimizer for learnable parameters.
        Uses MSE loss between total_goodness and target threshold as a simple example.
        """
        if optimizer is None:
            params = self.get_learnable_parameters()
            if params:
                optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            else:
                optimizer = None
        loss_fn = torch.nn.MSELoss()
        metrics = {
            'positive_goodness': [],
            'negative_goodness': [],
            'goodness_separation': []
        }
        for epoch in range(n_epochs):
            epoch_pos_goodness = 0
            epoch_neg_goodness = 0
            # Train on positive examples in batches
            for i in range(0, len(positive_data), batch_size):
                batch_list = positive_data[i:i+batch_size]
                if len(batch_list) < 2:
                    continue  # Skip batches too small for BatchNorm1d

                batch = torch.stack(batch_list)
                if optimizer:
                    optimizer.zero_grad()
                inputs = {'input': batch}
                result = self.step(inputs, is_positive=True)
                # result['total_goodness'] should be a tensor of batch_size or scalar; ensure it's meaned if needed
                if isinstance(result['total_goodness'], torch.Tensor) and result['total_goodness'].numel() > 1:
                    batch_goodness = result['total_goodness'].mean()
                else:
                    batch_goodness = result['total_goodness']
                epoch_pos_goodness += batch_goodness.item() * batch.size(0)
                if optimizer:
                    target = torch.full_like(result['total_goodness'], self.positive_threshold, dtype=torch.float32)
                    output = result['total_goodness']
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
            # Train on negative examples in batches
            for i in range(0, len(negative_data), batch_size):
                batch_list = negative_data[i:i+batch_size]
                if len(batch_list) < 2:
                    continue  # Skip batches too small for BatchNorm1d
                batch = torch.stack(batch_list)
                if optimizer:
                    optimizer.zero_grad()
                inputs = {'input': batch}
                result = self.step(inputs, is_positive=False)
                if isinstance(result['total_goodness'], torch.Tensor) and result['total_goodness'].numel() > 1:
                    batch_goodness = result['total_goodness'].mean()
                else:
                    batch_goodness = result['total_goodness']
                epoch_neg_goodness += batch_goodness.item() * batch.size(0)
                if optimizer:
                    target = torch.full_like(result['total_goodness'], self.negative_threshold, dtype=torch.float32)
                    output = result['total_goodness']
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
            # Calculate average goodness
            avg_pos_goodness = epoch_pos_goodness / len(positive_data)
            avg_neg_goodness = epoch_neg_goodness / len(negative_data)
            goodness_separation = avg_pos_goodness - avg_neg_goodness
            metrics['positive_goodness'].append(avg_pos_goodness)
            metrics['negative_goodness'].append(avg_neg_goodness)
            metrics['goodness_separation'].append(goodness_separation)
        return metrics


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