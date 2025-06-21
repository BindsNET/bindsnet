from typing import Dict, Optional, Callable, List, Tuple, Union # Add Union if batch can be List or Tuple
import torch
from bindsnet.pipeline.dataloader_pipeline import DataLoaderPipeline  # CHANGE: absolute import
from bindsnet.datasets.contrastive_transforms import prepend_label_to_image
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader # Use an alias to avoid confusion if needed
from bindsnet.network import Network as BindsNetwork

class BindsNETForwardForwardPipeline(DataLoaderPipeline):
    
    def __init__(
        self,
        #required parameters
        network: BindsNetwork,
        train_ds: torch.utils.data.Dataset,
        num_classes: int,
        encoder: Callable,
        time: int,
        ff_pairs_specs: List[Tuple[str, str]],
        input_layer_name: str,
        lr: float = 0.001,
        alpha_loss: float = 0.6,
        alpha: float = 2.0,
        spike_threshold: float = 1.0,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        device: Optional[torch.device] = None,
        dt: float = 1.0,
        batch_size: int = 32,  
        num_epochs: int = 1,   
        **kwargs,
    ):
        # STEP 1: Initialize required attributes BEFORE super().__init__()
        self.ff_layer_pairs: List = []  # Empty list to satisfy init_fn()
        self.optimizers: List = []
        
        # STEP 2: Store parameters for init_fn() to use
        self._stored_network = network  # Store network with different name
        self.train_ds = train_ds  # FIX: Store train_ds
        self.num_classes = num_classes
        self.encoder = encoder
        self.sim_time = time
        self.lr = lr
        self.alpha_loss = alpha_loss
        self.alpha = alpha
        self.input_layer_name = input_layer_name
        self.ff_pairs_specs = ff_pairs_specs
        self.optimizer_cls = optimizer_cls
        self.threshold = spike_threshold

        # Store DataLoader parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = kwargs.get('shuffle', True)
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory', False)

        if network is not None:
            network.dt = dt 

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # STEP 3: Call super().__init__() with None network, init_fn will set it up
        super().__init__(network=None, train_ds=train_ds,batch_size=batch_size,num_epochs=num_epochs, **kwargs)

    def init_fn(self) -> None:
        """
        Initialization function called by BasePipeline.
        This is where we do the actual ForwardForwardConnection setup.
        """
        print("init_fn() called - setting up ForwardForwardConnections...")
        
        # Restore the actual network
        self.network = self._stored_network
        
        if self.network is None:
            raise ValueError("Network cannot be None")
        
        # Import here to avoid circular imports
        from bindsnet.network.topology import ForwardForwardConnection

        # Clear and rebuild ff_layer_pairs
        self.ff_layer_pairs.clear()
        self.optimizers.clear()

        print(f"Available connections: {list(self.network.connections.keys())}")
        print(f"Looking for connections: {[pair[0] for pair in self.ff_pairs_specs]}")

        # Setup FF connections with surrogate gradients
        for conn_name, spiking_layer_name in self.ff_pairs_specs:
            if conn_name not in self.network.connections:
                raise ValueError(f"Connection '{conn_name}' not found in network. Available: {list(self.network.connections.keys())}")
            if spiking_layer_name not in self.network.layers:
                raise ValueError(f"Spiking layer '{spiking_layer_name}' not found in network. Available: {list(self.network.layers.keys())}")
            
            connection = self.network.connections[conn_name]
            spiking_layer = self.network.layers[spiking_layer_name]

            # Convert to ForwardForwardConnection if needed
            if not isinstance(connection, ForwardForwardConnection):
                print(f"Converting connection '{conn_name}' to ForwardForwardConnection")
                
                ff_connection = ForwardForwardConnection(
                    source=connection.source,
                    target=connection.target,
                    nu=getattr(connection, 'nu', None),
                    weight_decay=getattr(connection, 'weight_decay', 0.0),
                    spike_threshold=self.threshold,
                    alpha=self.alpha,
                    w=connection.w.data.clone() if hasattr(connection, 'w') else None,
                    wmin=getattr(connection, 'wmin', -torch.inf),
                    wmax=getattr(connection, 'wmax', torch.inf),
                    norm=getattr(connection, 'norm', None),
                )
                
                # Replace in network
                self.network.connections[conn_name] = ff_connection
                connection = ff_connection
            else:
                print(f"Connection '{conn_name}' is already ForwardForwardConnection")
            
            self.ff_layer_pairs.append((connection, spiking_layer))
            
            # FIX: Properly handle Parameter device transfer
            if hasattr(connection, 'w') and connection.w is not None:
                # Move the parameter data to device, keeping it as a Parameter
                if connection.w.device != self.device:
                    connection.w.data = connection.w.data.to(self.device)
                
                # Ensure gradients are enabled
                connection.w.requires_grad = True
                
                # Create optimizer for this connection
                self.optimizers.append(self.optimizer_cls([connection.w], lr=self.lr))
            else:
                raise ValueError(f"Connection '{conn_name}' has no weights!")

        # Initialize other attributes
        self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]

        # Setup dataloaders
        if self.train_ds is not None:
            from torch.utils.data import DataLoader as TorchDataLoader
            self.train_dataloader = TorchDataLoader(
                dataset=self.train_ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            self.train_dataloader = None
            print("Warning: train_ds is None.")
        
        self.test_dataloader = None
        
        print(f"BindsNETForwardForwardPipeline initialized with {len(self.ff_layer_pairs)} FF layer pairs")

    def _run_snn_batch_with_surrogate(
        self,
        batch_encoded_inputs: torch.Tensor, 
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Run SNN simulation with surrogate gradients for Forward-Forward training.
        Uses the ForwardForwardConnection's compute_with_surrogate method.
        """
        batch_size = batch_encoded_inputs.shape[0]
        time_steps = batch_encoded_inputs.shape[1]

        s_traces_batch: Dict[int, torch.Tensor] = {}
        g_scalars_batch: Dict[int, torch.Tensor] = {}

        # Initialize storage for each FF layer
        for ff_pair_idx, (connection, spiking_layer) in enumerate(self.ff_layer_pairs):
            num_neurons = connection.target.n
            s_traces_batch[ff_pair_idx] = torch.zeros(
                batch_size, time_steps, num_neurons, device=self.device
            )

        # Reset membrane potentials for all FF connections
        for connection, _ in self.ff_layer_pairs:
            connection.reset_membrane_potential()

        # Simulate through time with surrogate gradients
        current_input = batch_encoded_inputs  # [batch_size, time, features]
        
        for t in range(time_steps):
            layer_input = current_input[:, t, :]  # [batch_size, features]
            
            for ff_pair_idx, (connection, spiking_layer) in enumerate(self.ff_layer_pairs):
                # Forward through connection with surrogate gradients
                layer_spikes = connection.compute_with_surrogate(layer_input)
                
                # Store spike traces
                s_traces_batch[ff_pair_idx][:, t, :] = layer_spikes
                
                # Output becomes input for next layer
                layer_input = layer_spikes

        # Compute goodness scores for each layer
        for ff_pair_idx in s_traces_batch:
            # Total spike count per neuron across time
            spike_counts = torch.sum(s_traces_batch[ff_pair_idx], dim=1)  # [batch_size, neurons]
            # Goodness score: mean squared spike activity per sample
            g_scalars_batch[ff_pair_idx] = torch.mean(spike_counts ** 2, dim=1)  # [batch_size]
        
        return {
            "s_traces": s_traces_batch,
            "g_scalars": g_scalars_batch
        }

    def _compute_goodness_score(self, original_x: torch.Tensor, label_to_embed: int) -> float:
        """
        Compute goodness score for a single sample with a specific label.
        Now uses surrogate gradients for proper computation.
        """
        temp_labeled_input_single = prepend_label_to_image(
            original_x.cpu(), label_to_embed, self.num_classes
        ).to(self.device)
        
        encoded_input_single = self.encoder(temp_labeled_input_single)
        if encoded_input_single.dim() == 2:
            batch_encoded_input = encoded_input_single.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError("Encoder output for single sample should be [T, Features]")

        with torch.no_grad():
            run_data = self._run_snn_batch_with_surrogate(batch_encoded_input)
        
        total_goodness_for_sample = 0.0
        num_ff_pairs = len(self.ff_layer_pairs)
        if num_ff_pairs == 0: 
            return 0.0

        for ff_pair_idx in range(num_ff_pairs):
            total_goodness_for_sample += run_data["g_scalars"][ff_pair_idx].item()
        
        return total_goodness_for_sample / num_ff_pairs if num_ff_pairs > 0 else 0.0

    def _select_hard_negative_label(self, original_x_sample: torch.Tensor, true_label_sample: int) -> int:
        """
        Select a hard negative label using goodness scores.
        Now works with surrogate gradients for better gradient flow.
        """
        goodness_scores = torch.zeros(self.num_classes, device=self.device)
        
        with torch.no_grad():
            for label_idx in range(self.num_classes):
                if label_idx != true_label_sample:
                    goodness_scores[label_idx] = self._compute_goodness_score(original_x_sample, label_idx)
                else:
                    goodness_scores[label_idx] = 0.0
        
        # Remove true label from consideration
        non_true_labels = list(range(self.num_classes))
        non_true_labels.remove(true_label_sample)
        non_true_goodness = goodness_scores[non_true_labels]
        
        # Apply square root transformation
        epsilon = 1e-8
        transformed_goodness = torch.sqrt(non_true_goodness + epsilon)
        
        # Normalize to create probability distribution
        if transformed_goodness.sum() > 0:
            probabilities = transformed_goodness / transformed_goodness.sum()
        else:
            probabilities = torch.ones(len(non_true_labels), device=self.device) / len(non_true_labels)
        
        # Sample hard negative label
        selected_idx = torch.multinomial(probabilities, 1).item()
        return non_true_labels[selected_idx]

    def _forward_forward_update_batch(
        self,
        batch_original_x: torch.Tensor,
        batch_true_labels: torch.Tensor
    ) -> None:
        """
        Forward-Forward update with surrogate gradients for proper backpropagation.
        Now enables automatic differentiation through the BindsNET simulation.
        """
        print(f"_forward_forward_update_batch called with batch size: {batch_original_x.shape[0]}")
        
        batch_size = batch_original_x.shape[0]
        
        # Process each sample individually
        for i in range(batch_size):
            if i % 32 == 0:  # Progress update
                print(f"Processing sample {i+1}/{batch_size}")
            
            original_x_sample = batch_original_x[i]
            true_label_sample = batch_true_labels[i].item()
            
            # Update each FF layer pair separately
            for pair_idx in range(len(self.ff_layer_pairs)):
                # Zero gradients for this layer
                self.optimizers[pair_idx].zero_grad()
                
                # Ensure network connections are in training mode
                for connection, _ in self.ff_layer_pairs:
                    connection.w.requires_grad = True
                
                # Select hard negative label
                hard_negative_label = self._select_hard_negative_label(original_x_sample, true_label_sample)
                
                # Create positive and negative samples
                x_pos_sample = prepend_label_to_image(
                    original_x_sample, true_label_sample, self.num_classes
                ).to(self.device)
                x_neg_sample = prepend_label_to_image(  # Use positive sample with hard negative label
                    original_x_sample, hard_negative_label, self.num_classes
                ).to(self.device)
                
                # Encode samples
                batch_encoded_x_pos = self.encoder(x_pos_sample).unsqueeze(0)
                batch_encoded_x_neg = self.encoder(x_neg_sample).unsqueeze(0)
                
                # Forward passes with surrogate gradients (separate graphs)
                pos_run_data = self._run_snn_batch_with_surrogate(batch_encoded_x_pos)
                neg_run_data = self._run_snn_batch_with_surrogate(batch_encoded_x_neg)
                
                # Get goodness scores for this layer
                g_pos = pos_run_data["g_scalars"][pair_idx][0]
                g_neg = neg_run_data["g_scalars"][pair_idx][0]
                
                # Compute Forward-Forward loss
                loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))
                loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
                loss = self.alpha_loss * loss_pos + (1 - self.alpha_loss) * loss_neg
                
                # Backward pass with surrogate gradients - THIS NOW WORKS!
                loss.backward()
                self.optimizers[pair_idx].step()
                
                # Store loss
                self.current_epoch_losses[pair_idx].append(loss.item())
                
                # Reset membrane potentials for next sample
                for connection, _ in self.ff_layer_pairs:
                    connection.reset_membrane_potential()
    
    def step_(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], **kwargs) -> None:
        """
        Required step_ method for BasePipeline compatibility.
        This calls our train_step method for Forward-Forward training.
        """
        self.train_step(batch)

    def train_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> None:
        """Train step using ForwardForwardConnection with surrogate gradients."""
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"train_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device)

        # Ensure all FF connections are in training mode
        for connection, _ in self.ff_layer_pairs:
            connection.w.requires_grad = True
        
        self._forward_forward_update_batch(original_images, true_labels)

    def train(self) -> None:
        """Training loop with surrogate gradient support."""
        print(f"Starting Forward-Forward training for {self.num_epochs} epochs using BindsNET with surrogate gradients.")
        
        for epoch in range(self.num_epochs):
            print(f"\n=== Starting Epoch {epoch + 1}/{self.num_epochs} ===")
            self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]
            
            batch_count = 0
            total_loss = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch_data in enumerate(pbar):
                print(f"Processing batch {batch_idx + 1} with {len(batch_data[0])} samples")
                
                self.train_step(batch_data)
                batch_count += 1
                
                # Print loss info
                if len(self.current_epoch_losses[0]) > 0:
                    recent_loss = self.current_epoch_losses[0][-1]
                    total_loss += recent_loss
                    pbar.set_postfix({"loss": f"{recent_loss:.6f}"})
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"=== Epoch {epoch + 1} completed. Avg loss: {avg_loss:.6f} ===")

        print("Training complete with surrogate gradients!")

    def test_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[int, int]:
        """Test step using surrogate gradient goodness computation."""
        correct_predictions = 0
        
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"test_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device)
        total_samples = original_images.shape[0]

        with torch.no_grad(): 
            for i in range(total_samples):
                original_x_sample = original_images[i]
                true_label_sample = true_labels[i].item()
                
                goodness_for_all_classes = torch.tensor(
                    [self._compute_goodness_score(original_x_sample, j) for j in range(self.num_classes)],
                    device=self.device
                )
                predicted_label = torch.argmax(goodness_for_all_classes).item()
                
                if predicted_label == true_label_sample:
                    correct_predictions += 1
                    
        return correct_predictions, total_samples

    def test_epoch(self) -> None:
        """Test epoch using surrogate gradient evaluation."""
        if not self.test_dataloader:
            if hasattr(self, 'test_ds') and self.test_ds is not None:
                self.test_dataloader = TorchDataLoader(
                    self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, 
                    pin_memory=self.pin_memory, shuffle=False
                )
            else:
                print("No test dataloader configured and no test_ds found.")
                return

        total_correct = 0
        total_tested = 0
        pbar = tqdm(self.test_dataloader, desc="Testing")
        for batch_data in pbar:
            correct, num_samples = self.test_step(batch_data)
            total_correct += correct
            total_tested += num_samples
            if total_tested > 0:
                pbar.set_postfix({"acc": f"{(total_correct / total_tested) * 100:.2f}%"})
        
        if total_tested > 0:
            accuracy = (total_correct / total_tested) * 100
            print(f"Test Accuracy: {accuracy:.2f}% ({total_correct}/{total_tested})")
        else:
            print("No samples in test set or test dataloader is empty.")

    def get_surrogate_info(self) -> dict:
        """Get information about surrogate gradient configuration."""
        return {
            'surrogate_type': 'arctangent',
            'alpha': self.alpha,
            'spike_threshold': self.threshold,
            'num_ff_layers': len(self.ff_layer_pairs),
            'formula': '1 / (α * |input - threshold| + 1)'
        }