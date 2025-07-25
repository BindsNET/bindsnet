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
        
        # Initialize adaptive threshold tracking
        self._recent_positive_goodness = {}
        self._recent_negative_goodness = {}
        self.min_separation = 5.0  # Minimum required separation between pos/neg
        
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

        # Determine target goodness based on example type with adaptive thresholds
        layer_name = getattr(connection.target, 'name', 'unknown')
        
        if is_positive:
            # For positive samples, ensure target maintains separation above negative
            if layer_name in self._recent_negative_goodness:
                recent_neg = self._recent_negative_goodness[layer_name]
                adaptive_threshold = recent_neg + self.min_separation
                target_goodness = max(adaptive_threshold, self.positive_threshold)
            else:
                target_goodness = self.positive_threshold
        else:
            # For negative samples, ensure target maintains separation below positive
            if layer_name in self._recent_positive_goodness:
                recent_pos = self._recent_positive_goodness[layer_name]
                adaptive_threshold = recent_pos - self.min_separation
                target_goodness = min(adaptive_threshold, self.negative_threshold)
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

        Returns:
            Tensor of predicted labels for each sample
        """
        from bindsnet.datasets.contrastive_transforms import prepend_label_to_image

        predictions = []
        print(f"\n🔍 DEBUG: Recent goodness tracking:")
        print(f"   Positive: {self._recent_positive_goodness}")
        print(f"   Negative: {self._recent_negative_goodness}")
        
        for i, (data, _) in enumerate(test_dataset):
            goodness_scores = []
            
            for label in range(num_classes):
                sample = prepend_label_to_image(data, label, num_classes)
                
                # Use the same step_ method as training for consistency
                step_result = self.step_(sample.unsqueeze(0), is_positive=True)
                total_goodness = step_result['total_goodness']
                
                if isinstance(total_goodness, torch.Tensor):
                    goodness_scores.append(total_goodness.item())
                else:
                    goodness_scores.append(total_goodness)
                
                # DEBUG: Print scores for first sample
                if i == 0:
                    print(f"   Label {label}: goodness={goodness_scores[-1]:.3f}")
            
            # The label with highest goodness should be the prediction
            # (network should give high goodness to positive/correct samples)
            predicted_label = int(torch.tensor(goodness_scores).argmax())
            predictions.append(predicted_label)
            
            # DEBUG: Print prediction for first few samples
            if i < 3:
                print(f"\nSample {i}: goodness_scores={[f'{g:.3f}' for g in goodness_scores]}, predicted={predicted_label}")
                
        return torch.tensor(predictions)

    def compute_goodness_per_layer(self, sample: torch.Tensor) -> dict:
        """
        Compute goodness for each layer separately to identify which layers have learned.
        
        Args:
            sample: Input tensor for a single sample
            
        Returns:
            Dictionary of goodness scores per layer
        """
        # Reset network state
        self.network.reset_state_variables()
        for feature in self.features.values():
            feature.reset_state_variables()

        # Prepare inputs
        encoded_batch = {'input': sample}
        layer_activities = {layer_name: [] for layer_name in self.network.layers}
        
        # Run network simulation
        for t in range(self.time):
            if 'input' in encoded_batch:
                input_data = encoded_batch['input']
                input_probs = torch.clamp(input_data, 0.0, 1.0)
                spike_input = torch.bernoulli(input_probs)
                timestep_inputs = {'input': spike_input}
            else:
                timestep_inputs = encoded_batch
            
            self.network.run(timestep_inputs, time=1)
            
            # Collect layer activities
            for layer_name, layer in self.network.layers.items():
                layer_activities[layer_name].append(layer.s.clone())

        # Compute goodness for each layer
        goodness_dict = {}
        total_goodness = torch.tensor(0.0, requires_grad=True)
        
        for conn_key, feature in self.features.items():
            # Find the target layer for this connection
            target_layer_name = None
            for name, conn in self.network.connections.items():
                if str(name) == str(conn_key):
                    for layer_name, layer in self.network.layers.items():
                        if layer is conn.target:
                            target_layer_name = layer_name
                            break
                    break
            
            if target_layer_name and target_layer_name in layer_activities:
                # Sum activity over time
                target_activity = torch.stack(layer_activities[target_layer_name], dim=0)
                target_activity_sum = target_activity.sum(dim=0).float()
                
                if target_activity_sum.dim() == 1:
                    target_activity_sum = target_activity_sum.unsqueeze(0)
                
                # Set feature value and apply batch normalization
                feature.value = target_activity_sum.requires_grad_(True)
                
                batch_norm = getattr(feature, 'batch_norm', None)
                if batch_norm is not None:
                    try:
                        normalized_activity = batch_norm.batch_normalize()
                        feature.value = normalized_activity
                    except:
                        # Skip batch norm if it fails (e.g., single sample issues)
                        pass
                
                # Compute goodness
                layer_goodness = (feature.value ** 2).sum()
                goodness_dict[f"layer_{target_layer_name}"] = layer_goodness
                total_goodness = total_goodness + layer_goodness

        goodness_dict["total_goodness"] = total_goodness
        return goodness_dict

        
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
        Uses hard labeling with square root transformation and probabilistic sampling
        as described in the Forward-Forward paper.
        """
        from bindsnet.datasets.contrastive_transforms import prepend_label_to_image

        positive_data = []
        negative_data = []
        
        print("Generating hard negative samples with square root transformation...")
        hard_label_distribution = [0] * num_classes  # Track distribution of hard labels
        
        for i, (data, target) in enumerate(train_dataset):
            # Create positive sample with true label
            pos_sample = prepend_label_to_image(data, target, num_classes)
            positive_data.append(pos_sample)

            # Compute goodness scores for all possible class labels
            goodness_scores = []
            candidate_samples = []
            
            for label in range(num_classes):
                candidate_sample = prepend_label_to_image(data, label, num_classes)
                candidate_samples.append(candidate_sample)
                
                # Compute goodness score for this label
                goodness = self.goodness_score.compute(sample=candidate_sample.unsqueeze(0))["total_goodness"]
                goodness_scores.append(goodness.item() if isinstance(goodness, torch.Tensor) else goodness)
            
            # Set goodness score for true class label to zero (exclude from hard labeling)
            goodness_scores[target] = 0.0
            
            # Apply square root transformation to flatten the distribution
            # This makes the distribution less peaked and more uniform
            transformed_scores = [max(0.0, score) ** 0.5 for score in goodness_scores]
            
            # Convert to probability distribution (normalize)
            total_score = sum(transformed_scores)
            if total_score > 0:
                probabilities = [score / total_score for score in transformed_scores]
            else:
                # Fallback: uniform distribution over non-true labels
                probabilities = [1.0 / (num_classes - 1) if i != target else 0.0 for i in range(num_classes)]
            
            # Sample hard label using the transformed probability distribution
            # This chooses labels that the network finds relatively difficult to distinguish
            hard_label = torch.multinomial(torch.tensor(probabilities), 1).item()
            hard_label_distribution[hard_label] += 1
            
            # Add the hard negative sample
            negative_data.append(candidate_samples[hard_label])
            
        #     # Print progress for first few samples
        #     if i < 5:
        #         print(f"  Sample {i}: true={target}, hard_negative={hard_label}")
        #         print(f"    Raw goodness: {[f'{g:.2f}' for g in goodness_scores]}")
        #         print(f"    Transformed:  {[f'{t:.2f}' for t in transformed_scores]}")
        #         print(f"    Probabilities: {[f'{p:.3f}' for p in probabilities]}")
        
        # print(f"Hard label distribution: {hard_label_distribution}")
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
            # Ensure batch normalization is in training mode during training
            batch_norm = getattr(feature, 'batch_norm', None)
            if batch_norm is not None:
                batch_norm.set_training_mode(training=True)

        # Run the network for self.time steps and extract features
        outputs = {pop_name: [] for pop_name in self.network.layers}
        layer_activities = {layer_name: [] for layer_name in self.network.layers}
        
        # Track normalized activities per timestep for proper Forward-Forward learning
        normalized_activities = {layer_name: [] for layer_name in self.network.layers}
        
        for t in range(self.time):
            # For each timestep, generate spikes from the batch
            if 'input' in encoded_batch:
                # Generate spikes for this timestep
                input_data = encoded_batch['input']
                
                # Use Poisson encoding for intensity-scaled data (MNIST with intensity scaling)
                # Check if data is intensity-scaled (values > 1.0) or probability-scaled (values <= 1.0)
                if input_data.max() > 1.0:
                    # Intensity-scaled data: use Poisson encoding
                    from bindsnet.encoding.encodings import poisson
                    spike_input = poisson(input_data, time=1, dt=self.dt).squeeze(0)  # Remove time dimension
                else:
                    # Probability data: use Bernoulli encoding
                    input_probs = torch.clamp(input_data, 0.0, 1.0)
                    spike_input = torch.bernoulli(input_probs)
                
                timestep_inputs = {'input': spike_input}
            else:
                timestep_inputs = encoded_batch
            
            self.network.run(timestep_inputs, time=1)
            
            # Apply per-timestep batch normalization and collect activities
            for conn_key, feature in self.features.items():
                # Get connection to determine target layer
                connection = None
                for name, conn in self.network.connections.items():
                    if str(name) == str(conn_key):
                        connection = conn
                        break
                
                if connection is not None:
                    # Get target layer name
                    target_layer_name = None
                    for layer_name, layer in self.network.layers.items():
                        if layer is connection.target:
                            target_layer_name = layer_name
                            break
                    
                    if target_layer_name is not None:
                        # Get current timestep activity
                        current_activity = self.network.layers[target_layer_name].s.clone().float()
                        
                        # Ensure proper batch dimension
                        if current_activity.dim() == 1:
                            current_activity = current_activity.unsqueeze(0)
                        
                        # Apply per-timestep batch normalization if present
                        batch_norm = getattr(feature, 'batch_norm', None)
                        if batch_norm is not None:
                            # Set the current activity as feature value for normalization
                            feature.value = current_activity.requires_grad_(True)
                            
                            # Apply batch normalization at this timestep
                            normalized_activity = batch_norm.batch_normalize()
                            
                            # Store normalized activity for this timestep
                            normalized_activities[target_layer_name].append(normalized_activity)
                        else:
                            # No batch norm, just store the raw activity
                            normalized_activities[target_layer_name].append(current_activity)
            
            # Collect layer activities for all layers (store original for outputs)
            for layer_name, layer in self.network.layers.items():
                layer_activities[layer_name].append(layer.s.clone())
            
            # Collect outputs for each layer (original spike outputs)
            for pop_name, population in self.network.layers.items():
                outputs[pop_name].append(population.s.clone())

        # Compute goodness from normalized activities accumulated over time
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
                # Get target layer name
                target_layer_name = None
                for layer_name, layer in self.network.layers.items():
                    if layer is connection.target:
                        target_layer_name = layer_name
                        break
                
                if target_layer_name is not None and target_layer_name in normalized_activities:
                    # Sum normalized activities over time to get feature representation
                    if len(normalized_activities[target_layer_name]) > 0:
                        target_activity_sum = torch.stack(normalized_activities[target_layer_name], dim=0).sum(dim=0)
                        
                        # Set final feature value 
                        feature.value = target_activity_sum.requires_grad_(True)
                        
                        # Compute goodness from the normalized feature values
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
        Train the network using Forward-Forward learning with sequential layer-wise training.
        Each sample goes through layer 1 and is trained on it BEFORE going to the next layer.
        If optimizer is not provided, creates Adam optimizer for learnable parameters.
        """
        if optimizer is None:
            params = self.get_learnable_parameters()
            if params:
                optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            else:
                optimizer = None
        
        loss_fn = torch.nn.MSELoss()
        
        # Determine layer order from network connections
        layer_order = self._get_layer_training_order()
        print(f"Training layers in order: {layer_order}")
        
        metrics = {
            'positive_goodness': [],
            'negative_goodness': [],
            'goodness_separation': [],
            'layer_goodness': {layer_info['target_layer']: [] for layer_info in layer_order},
            'layer_losses': {layer_info['target_layer']: {'positive': [], 'negative': [], 'total': []} for layer_info in layer_order},
            'epoch_total_loss': []
        }
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print("=" * 50)
            epoch_metrics = {}
            epoch_total_loss = 0.0
            
            # Train layer by layer, sample by sample
            for layer_idx, layer_info in enumerate(layer_order):
                layer_name = layer_info['target_layer']
                connection_name = layer_info['connection']
                print(f"\nTraining layer: {layer_name} via connection: {connection_name}")
                
                # Train on positive examples for this layer
                layer_pos_goodness, layer_pos_loss = self._train_layer_on_samples(
                    positive_data, layer_name, connection_name, 
                    is_positive=True, batch_size=batch_size, 
                    optimizer=optimizer, loss_fn=loss_fn
                )
                
                # Train on negative examples for this layer
                layer_neg_goodness, layer_neg_loss = self._train_layer_on_samples(
                    negative_data, layer_name, connection_name,
                    is_positive=False, batch_size=batch_size,
                    optimizer=optimizer, loss_fn=loss_fn
                )
                
                layer_separation = layer_pos_goodness - layer_neg_goodness
                layer_total_loss = layer_pos_loss + layer_neg_loss
                epoch_total_loss += layer_total_loss
                
                # Store layer metrics
                epoch_metrics[layer_name] = {
                    'pos': layer_pos_goodness,
                    'neg': layer_neg_goodness,
                    'separation': layer_separation,
                    'pos_loss': layer_pos_loss,
                    'neg_loss': layer_neg_loss,
                    'total_loss': layer_total_loss
                }
                
                metrics['layer_goodness'][layer_name].append(layer_separation)
                metrics['layer_losses'][layer_name]['positive'].append(layer_pos_loss)
                metrics['layer_losses'][layer_name]['negative'].append(layer_neg_loss)
                metrics['layer_losses'][layer_name]['total'].append(layer_total_loss)
                
                print(f"  Goodness - Pos: {layer_pos_goodness:.3f}, Neg: {layer_neg_goodness:.3f}, Sep: {layer_separation:.3f}")
                print(f"  Loss     - Pos: {layer_pos_loss:.4f}, Neg: {layer_neg_loss:.4f}, Total: {layer_total_loss:.4f}")
                
                # Check if layer is learning (loss should decrease over time)
                if len(metrics['layer_losses'][layer_name]['total']) >= 2:
                    prev_loss = metrics['layer_losses'][layer_name]['total'][-2]
                    curr_loss = metrics['layer_losses'][layer_name]['total'][-1]
                    loss_change = curr_loss - prev_loss
                    trend = "↓" if loss_change < 0 else "↑"
                    print(f"  Loss trend: {trend} ({loss_change:+.4f}) {'[LEARNING]' if loss_change < 0 else '[NOT LEARNING]'}")
                
                # If this layer shows good separation, we can start training the next layer
                # Otherwise, spend more time on this layer
                if abs(layer_separation) < 0.5 and layer_idx == 0:  # First layer needs to learn well
                    print(f"  ⚠️  Layer {layer_name} separation too low ({layer_separation:.3f}), may need more training")
            
            # Store epoch-level metrics
            metrics['epoch_total_loss'].append(epoch_total_loss)
            
            # Calculate overall metrics (focus on layers that are actually learning)
            learning_layers = []
            total_pos = 0
            total_neg = 0
            
            for layer_name, layer_metrics in epoch_metrics.items():
                if abs(layer_metrics['separation']) > 0.1:  # Layer is showing some learning
                    learning_layers.append(layer_name)
                    total_pos += layer_metrics['pos']
                    total_neg += layer_metrics['neg']
            
            if learning_layers:
                avg_pos_goodness = total_pos / len(learning_layers)
                avg_neg_goodness = total_neg / len(learning_layers)
                goodness_separation = avg_pos_goodness - avg_neg_goodness
                
                print(f"\n📊 Epoch {epoch + 1} Summary:")
                print(f"   Learning layers: {learning_layers}")
                print(f"   Overall separation: {goodness_separation:.3f}")
                print(f"   Total loss: {epoch_total_loss:.4f}")
                
                # Check epoch-level learning trend
                if len(metrics['epoch_total_loss']) >= 2:
                    prev_epoch_loss = metrics['epoch_total_loss'][-2]
                    curr_epoch_loss = metrics['epoch_total_loss'][-1]
                    epoch_loss_change = curr_epoch_loss - prev_epoch_loss
                    epoch_trend = "↓" if epoch_loss_change < 0 else "↑"
                    print(f"   Epoch loss trend: {epoch_trend} ({epoch_loss_change:+.4f}) {'[IMPROVING]' if epoch_loss_change < 0 else '[NOT IMPROVING]'}")
            else:
                avg_pos_goodness = 0
                avg_neg_goodness = 0
                goodness_separation = 0
                print(f"\n⚠️  Epoch {epoch + 1}: No layers showing meaningful learning yet")
                print(f"   Total loss: {epoch_total_loss:.4f}")
            
            metrics['positive_goodness'].append(avg_pos_goodness)
            metrics['negative_goodness'].append(avg_neg_goodness)
            metrics['goodness_separation'].append(goodness_separation)
            
        
        print(f"\n{'='*60}")
        print(f"🎯 FINAL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Final goodness separation: {metrics['goodness_separation'][-1]:.3f}")
        print(f"Final total loss: {metrics['epoch_total_loss'][-1]:.4f}")
        
        print(f"\n📈 Layer-wise Learning Analysis:")
        for layer_name, separations in metrics['layer_goodness'].items():
            if len(separations) > 0:
                final_sep = separations[-1] if isinstance(separations[-1], (int, float)) else separations[-1].item()
                initial_sep = separations[0] if isinstance(separations[0], (int, float)) else separations[0].item()
                sep_change = final_sep - initial_sep
                
                final_loss = metrics['layer_losses'][layer_name]['total'][-1]
                initial_loss = metrics['layer_losses'][layer_name]['total'][0]
                loss_change = final_loss - initial_loss
                
                print(f"  {layer_name}:")
                print(f"    Goodness separation: {initial_sep:.3f} → {final_sep:.3f} ({sep_change:+.3f})")
                print(f"    Total loss: {initial_loss:.4f} → {final_loss:.4f} ({loss_change:+.4f})")
                
                # Determine if layer learned
                learned = loss_change < -0.01  # Significant loss decrease
                status = "✅ LEARNED" if learned else "❌ NO LEARNING"
                print(f"    Status: {status}")
        
        # Overall learning assessment
        overall_loss_change = metrics['epoch_total_loss'][-1] - metrics['epoch_total_loss'][0]
        overall_learned = overall_loss_change < -0.1
        print(f"\n🏆 Overall Assessment:")
        print(f"   Total loss change: {overall_loss_change:+.4f}")
        print(f"   Network status: {'✅ LEARNING' if overall_learned else '❌ NOT LEARNING'}")
        
        if not overall_learned:
            print(f"\n💡 Possible Issues:")
            print(f"   - Learning rate too high/low")
            print(f"   - Insufficient training data")
            print(f"   - Network architecture problems")
            print(f"   - Threshold mismatch")
        
        return metrics

    def _get_layer_training_order(self) -> list:
        """
        Determine the order in which layers should be trained based on network topology.
        Returns list of dicts with 'target_layer', 'source_layer', 'connection' keys.
        """
        layer_order = []
        
        # Identify layer connections and their order
        for conn_name, connection in self.network.connections.items():
            # Find source and target layer names
            source_layer_name = None
            target_layer_name = None
            
            for layer_name, layer in self.network.layers.items():
                if layer is connection.source:
                    source_layer_name = layer_name
                if layer is connection.target:
                    target_layer_name = layer_name
            
            if source_layer_name and target_layer_name:
                layer_order.append({
                    'source_layer': source_layer_name,
                    'target_layer': target_layer_name,
                    'connection': str(conn_name)
                })
        
        # Sort by layer depth (input -> hidden_0 -> hidden_1, etc.)
        def layer_depth(layer_info):
            target = layer_info['target_layer']
            if 'input' in target:
                return -1  # Input layer
            elif 'hidden_0' in target:
                return 0
            elif 'hidden_1' in target:
                return 1
            elif 'hidden_2' in target:
                return 2
            else:
                return 999  # Unknown, put at end
        
        layer_order.sort(key=layer_depth)
        return layer_order

    def _train_layer_on_samples(
        self,
        data: torch.Tensor,
        target_layer_name: str,
        connection_name: str,
        is_positive: bool,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module
    ) -> Tuple[float, float]:
        """
        Train a specific layer on all samples before moving to the next layer.
        Each sample completes training on this layer before proceeding.
        
        Returns:
            Tuple of (average_goodness, average_loss)
        """
        total_goodness = 0.0
        total_loss = 0.0
        sample_count = 0
        
        # Process samples in batches for this specific layer
        for i in range(0, len(data), batch_size):
            batch_list = data[i:i+batch_size]
            if len(batch_list) < 1:  # Allow single samples
                continue
                
            batch = torch.stack(batch_list) if len(batch_list) > 1 else batch_list[0].unsqueeze(0)
            
            # Forward pass for this specific layer only
            layer_goodness = self._forward_through_layer(
                batch, target_layer_name, connection_name, is_positive
            )
            
            total_goodness += layer_goodness * batch.size(0)
            sample_count += batch.size(0)
            
            # Backward pass and optimization for this layer
            batch_loss = 0.0
            if optimizer is not None:
                optimizer.zero_grad()
                
                # Compute loss for this layer
                target_goodness = self.positive_threshold if is_positive else self.negative_threshold
                target = torch.tensor(target_goodness, dtype=torch.float32)
                
                if isinstance(layer_goodness, torch.Tensor):
                    loss = loss_fn(layer_goodness, target)
                else:
                    loss = loss_fn(torch.tensor(layer_goodness), target)
                
                batch_loss = loss.item()
                total_loss += batch_loss * batch.size(0)
                
                loss.backward()
                optimizer.step()
            else:
                # If no optimizer, still compute loss for tracking
                target_goodness = self.positive_threshold if is_positive else self.negative_threshold
                target = torch.tensor(target_goodness, dtype=torch.float32)
                
                if isinstance(layer_goodness, torch.Tensor):
                    loss = loss_fn(layer_goodness, target)
                else:
                    loss = loss_fn(torch.tensor(layer_goodness), target)
                
                batch_loss = loss.item()
                total_loss += batch_loss * batch.size(0)
        
        avg_goodness = total_goodness / max(sample_count, 1)
        avg_loss = total_loss / max(sample_count, 1)
        
        # Update recent goodness tracking for adaptive thresholds
        if is_positive:
            self._recent_positive_goodness[target_layer_name] = avg_goodness
        else:
            self._recent_negative_goodness[target_layer_name] = avg_goodness
        
        return avg_goodness, avg_loss

    def _forward_through_layer(
        self,
        batch: torch.Tensor,
        target_layer_name: str,
        connection_name: str,
        is_positive: bool
    ) -> torch.Tensor:
        """
        Forward pass through a specific layer and compute its goodness.
        Accumulates activations from input through the target layer only.
        """
        # Reset network state
        self.network.reset_state_variables()
        for feature in self.features.values():
            feature.reset_state_variables()
        
        # Prepare inputs
        encoded_batch = {'input': batch}
        layer_activities = {layer_name: [] for layer_name in self.network.layers}
        
        # Run network through target layer only
        for t in range(self.time):
            # Generate spikes for this timestep
            if 'input' in encoded_batch:
                input_data = encoded_batch['input']
                input_probs = torch.clamp(input_data, 0.0, 1.0)
                spike_input = torch.bernoulli(input_probs)
                timestep_inputs = {'input': spike_input}
            else:
                timestep_inputs = encoded_batch
            
            # Run only up to the target layer
            self._run_network_to_layer(timestep_inputs, target_layer_name)
            
            # Collect activities for the target layer
            target_layer = self.network.layers[target_layer_name]
            layer_activities[target_layer_name].append(target_layer.s.clone())
        
        # Compute goodness for this specific layer
        layer_goodness = self._compute_layer_goodness(
            layer_activities[target_layer_name], connection_name
        )
        
        return layer_goodness

    def _run_network_to_layer(self, inputs: dict, target_layer_name: str):
        """
        Run network computation only up to the specified target layer.
        Uses BindsNET's standard network.run() but limits propagation.
        """
        # Use BindsNET's standard simulation but only collect results up to target layer
        self.network.run(inputs, time=1)
        
        # Note: BindsNET already handles the full forward pass internally
        # We just need to ensure we only use the target layer's output

    def _is_layer_before_or_equal(self, layer1: str, layer2: str) -> bool:
        """Check if layer1 comes before or is equal to layer2 in network hierarchy."""
        def get_layer_index(layer_name):
            if 'input' in layer_name:
                return -1
            elif 'hidden_0' in layer_name:
                return 0
            elif 'hidden_1' in layer_name:
                return 1
            elif 'hidden_2' in layer_name:
                return 2
            else:
                return 999
        
        return get_layer_index(layer1) <= get_layer_index(layer2)

    def _compute_layer_goodness(
        self,
        layer_activity_over_time: list,
        connection_name: str
    ) -> torch.Tensor:
        """
        Compute goodness for a specific layer from its activity over time.
        """
        # Get the feature for this connection
        feature = self.features.get(connection_name)
        if feature is None:
            # Fallback: compute simple goodness from activity
            if len(layer_activity_over_time) > 0:
                total_activity = torch.stack(layer_activity_over_time, dim=0).sum(dim=0).float()
                return (total_activity ** 2).sum()
            else:
                return torch.tensor(0.0)
        
        # Sum activity over time
        if len(layer_activity_over_time) > 0:
            total_activity = torch.stack(layer_activity_over_time, dim=0).sum(dim=0).float()
            
            # Ensure proper batch dimension
            if total_activity.dim() == 1:
                total_activity = total_activity.unsqueeze(0)
            
            # Set feature value with gradients
            feature.value = total_activity.requires_grad_(True)
            
            # Apply batch normalization if present
            batch_norm = getattr(feature, 'batch_norm', None)
            if batch_norm is not None:
                normalized_activity = batch_norm.batch_normalize()
                feature.value = normalized_activity

            # Compute goodness from normalized features
            goodness = (feature.value ** 2).sum()
            return goodness
        else:
            return torch.tensor(0.0)

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