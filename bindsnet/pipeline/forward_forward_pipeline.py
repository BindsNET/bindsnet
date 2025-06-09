from typing import Dict, Optional, Callable, List, Tuple, Union # Add Union if batch can be List or Tuple
import torch
import torch.nn as nn # Ensure nn is imported if used in type hints
import snntorch as snn # Ensure snn is imported if used in type hints
from bindsnet.pipeline.dataloader_pipeline import DataLoaderPipeline  # CHANGE: absolute import

from bindsnet.datasets.contrastive_transforms import generate_positive_sample, generate_negative_sample
import random
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader # Use an alias to avoid confusion if needed

class ForwardForwardPipeline(DataLoaderPipeline):
    """
    Pipeline for training a Spiking Neural Network using a Forward-Forward
    algorithm variant with hard negative mining, adapted for snntorch.
    """

    def __init__(
        self,
        network: nn.Module,
        train_ds: torch.utils.data.Dataset,
        num_classes: int,
        encoder: Callable,
        time: int,
        ff_layer_pairs: List[Tuple[nn.Linear, snn.SpikingNeuron]],
        lr: float = 0.001,
        alpha_loss: float = 0.6,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        device: Optional[torch.device] = None,
        dt: float = 1.0,
        **kwargs, # This will include batch_size, num_workers, etc.
    ):
        super().__init__(network=network, train_ds=train_ds, **kwargs) # DataLoaderPipeline sets self.train_ds, self.batch_size etc.
        self.network = network
        self.threshold = kwargs.get('threshold', 1.0)  # CHANGE THIS LINE - extract from kwargs with default
        self.num_classes = num_classes
        self.encoder = encoder
        self.sim_time = time
        self.dt = dt
        self.lr = lr
        self.alpha_loss = alpha_loss

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.ff_layer_pairs = ff_layer_pairs
        if not self.ff_layer_pairs:
            print("Warning: No 'ff_layer_pairs' specified. No layers will be updated by FF.")

        self.optimizers = []
        for linear_layer, _ in self.ff_layer_pairs:
            self.optimizers.append(optimizer_cls(linear_layer.parameters(), lr=self.lr))
        
        self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]

        # Initialize train_dataloader using attributes set by DataLoaderPipeline's __init__
        if self.train_ds is not None:
            self.train_dataloader = TorchDataLoader(
                dataset=self.train_ds,
                batch_size=self.batch_size, # This attribute is set by DataLoaderPipeline
                shuffle=self.shuffle,       # This attribute is set by DataLoaderPipeline
                num_workers=self.num_workers, # This attribute is set by DataLoaderPipeline
                pin_memory=self.pin_memory    # This attribute is set by DataLoaderPipeline
            )
        else:
            self.train_dataloader = None
            print("Warning: train_ds is None. self.train_dataloader is not created.")
        
        # test_dataloader is handled in test_epoch, which is fine.
        self.test_dataloader = None 


    def init_fn(self) -> None:
        """
        Initialization function called by the BasePipeline.
        For ForwardForwardPipeline, specific initializations are primarily handled
        in its own __init__ method. This method satisfies the BasePipeline's
        requirement for an init_fn to be implemented.
        """
        # If there's any setup that MUST happen after BasePipeline's __init__
        # and DataLoaderPipeline's __init__ are complete, it would go here.
        # For now, assuming all necessary setup for ForwardForwardPipeline
        # is done in its own __init__.
        pass

    def _run_snn_batch(
        self,
        batch_encoded_inputs: torch.Tensor,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        # language=rst
        """
        Runs the SNN (snntorch based) for a batch of encoded inputs.
        Collects spikes and membrane potentials for specified ff_layer_pairs.
        The forward pass iterates through the (Linear, SpikingNeuron) pairs in self.ff_layer_pairs.

        :param batch_encoded_inputs: Batch of spike-encoded inputs. Shape: [B, T, Features_input]
        :return: A dictionary containing collected data, keyed by ff_pair_idx:
            's_traces': {pair_idx: [B, T, N_spiking_layer]}
            'v_traces': {pair_idx: [B, T, N_spiking_layer]}
            'c_totals': {pair_idx: [B, N_spiking_layer]}
            'g_scalars': {pair_idx: [B]}
        """
        batch_size = batch_encoded_inputs.shape[0]

        s_traces_batch = {}
        v_traces_batch = {}
        # Initialize trace tensors based on ff_layer_pairs
        for idx, (linear_layer, _) in enumerate(self.ff_layer_pairs): # Use spiking_layer from pair for num_neurons if needed
            if hasattr(linear_layer, 'out_features'):
                num_neurons = linear_layer.out_features
            else:
                raise ValueError(
                    f"Cannot determine number of output neurons for pair {idx}. "
                    f"The first element (linear_layer) of the pair does not have 'out_features'."
                )
            s_traces_batch[idx] = torch.zeros(batch_size, self.sim_time, num_neurons, device=self.device)
            v_traces_batch[idx] = torch.zeros(batch_size, self.sim_time, num_neurons, device=self.device)

        # Initialize states for each spiking layer in ff_layer_pairs
        # Key by the spiking_layer object itself for unique state management during the pass
        spiking_layer_states = {spiking_layer_obj: None for _, spiking_layer_obj in self.ff_layer_pairs}

        for t in range(self.sim_time):
            # Input for the first linear layer at this timestep
            current_input_for_ff_sequence = batch_encoded_inputs[:, t, :]

            for ff_pair_idx, (linear_layer, spiking_ff_layer) in enumerate(self.ff_layer_pairs):
                # Pass through linear layer
                linear_output = linear_layer(current_input_for_ff_sequence)

                # Pass through spiking layer
                # Get the state for *this specific* spiking_ff_layer instance
                current_state = spiking_layer_states.get(spiking_ff_layer)
                spk_out, new_mem = spiking_ff_layer(linear_output, current_state)
                spiking_layer_states[spiking_ff_layer] = new_mem # Update state for this layer

                # Store traces, using ff_pair_idx which corresponds to the s_traces_batch/v_traces_batch keys
                s_traces_batch[ff_pair_idx][:, t, :] = spk_out
                v_traces_batch[ff_pair_idx][:, t, :] = new_mem.clone().detach() # Or new_mem if it's already a new tensor

                # Output of this spiking layer becomes input to the next linear layer in the sequence
                current_input_for_ff_sequence = spk_out

        c_totals_batch = {}
        g_scalars_batch = {}
        for ff_pair_idx in s_traces_batch: # s_traces_batch is already keyed by ff_pair_idx
            c_totals_batch[ff_pair_idx] = torch.sum(s_traces_batch[ff_pair_idx].float(), dim=1)
            if c_totals_batch[ff_pair_idx].numel() > 0:
                g_scalars_batch[ff_pair_idx] = torch.mean(c_totals_batch[ff_pair_idx]**2, dim=1)
            else:
                g_scalars_batch[ff_pair_idx] = torch.zeros(batch_size, device=self.device)
        
        return {
            "s_traces": s_traces_batch,
            "v_traces": v_traces_batch,
            "c_totals": c_totals_batch,
            "g_scalars": g_scalars_batch
        }

    def _compute_goodness_score(self, original_x: torch.Tensor, label_to_embed: int) -> float:
        temp_labeled_input_single = generate_positive_sample(
            original_x, label_to_embed, self.num_classes
        ).to(self.device)
        encoded_input_single = self.encoder(temp_labeled_input_single)
        batch_encoded_input = encoded_input_single.unsqueeze(0)

        # Ensure network is in eval mode for goodness score computation if it has dropout/BN
        self.network.eval()
        with torch.no_grad(): # No gradients needed for goodness score
            run_data = self._run_snn_batch(batch_encoded_input)
        
        total_goodness_for_sample = 0.0
        num_ff_pairs = len(self.ff_layer_pairs)
        if num_ff_pairs == 0: return 0.0

        for ff_pair_idx in range(num_ff_pairs):
            total_goodness_for_sample += run_data["g_scalars"][ff_pair_idx].item()
        
        return total_goodness_for_sample / num_ff_pairs if num_ff_pairs > 0 else 0.0

    def _select_hard_negative_label(self, original_x_sample: torch.Tensor, true_label_sample: int) -> int:
        """
        Select a hard negative label using goodness scores as described in the paper.
        
        1. Compute goodness scores for all possible class labels
        2. Set true label goodness to zero (exclude it)
        3. Apply square root transformation to flatten distribution
        4. Sample from the transformed distribution
        """
        # Compute goodness scores for all possible labels
        goodness_scores = torch.zeros(self.num_classes, device=self.device)
        
        self.network.eval()  # Use eval mode for consistent goodness computation
        with torch.no_grad():
            for label_idx in range(self.num_classes):
                if label_idx != true_label_sample:  # Skip true label
                    goodness_scores[label_idx] = self._compute_goodness_score(original_x_sample, label_idx)
                else:
                    goodness_scores[label_idx] = 0.0  # Set true label goodness to zero
        
        # Remove true label from consideration
        non_true_labels = list(range(self.num_classes))
        non_true_labels.remove(true_label_sample)
        non_true_goodness = goodness_scores[non_true_labels]
        
        # Apply square root transformation to flatten the distribution
        # Add small epsilon to avoid sqrt(0) issues
        epsilon = 1e-8
        transformed_goodness = torch.sqrt(non_true_goodness + epsilon)
        
        # Normalize to create probability distribution
        if transformed_goodness.sum() > 0:
            probabilities = transformed_goodness / transformed_goodness.sum()
        else:
            # Fallback to uniform distribution if all goodness scores are zero
            probabilities = torch.ones(len(non_true_labels), device=self.device) / len(non_true_labels)
        
        # Sample a hard negative label based on the probability distribution
        selected_idx = torch.multinomial(probabilities, 1).item()
        hard_negative_label = non_true_labels[selected_idx]
        
        return hard_negative_label

    def _forward_forward_update_batch(self, batch_original_x: torch.Tensor, batch_true_labels: torch.Tensor) -> None:
        print(f"_forward_forward_update_batch called with batch size: {batch_original_x.shape[0]}")
        
        batch_size = batch_original_x.shape[0]
        
        # Process each sample individually
        for i in range(batch_size):
            if i % 100 == 0:  # Print progress every 100 samples to avoid spam
                print(f"Processing sample {i+1}/{batch_size}")
            
            original_x_sample = batch_original_x[i]
            true_label_sample = batch_true_labels[i].item()
            
            # Update each FF layer pair separately with COMPLETELY independent forward passes
            for pair_idx in range(len(self.ff_layer_pairs)):
                if i % 1000 == 0:  # Only print layer updates for every 1000th sample
                    print(f"Updating layer pair {pair_idx}")
                
                # Zero gradients for this layer
                self.optimizers[pair_idx].zero_grad()
                
                # Ensure network is in training mode for gradient computation
                self.network.train()
                
                # HARD NEGATIVE MINING: Select intelligent negative label
                hard_negative_label = self._select_hard_negative_label(original_x_sample, true_label_sample)
                
                # Create positive and negative samples for this single sample
                x_pos_sample = generate_positive_sample(original_x_sample, true_label_sample, self.num_classes).to(self.device)
                
                # Use the hard negative label instead of random selection
                if hasattr(generate_negative_sample, '__code__') and 'false_label_override' in generate_negative_sample.__code__.co_varnames:
                    # If generate_negative_sample supports false_label_override parameter
                    x_neg_sample = generate_negative_sample(
                        original_x_sample, true_label_sample, self.num_classes, 
                        false_label_override=hard_negative_label
                    ).to(self.device)
                else:
                    # Fallback: manually create negative sample with hard label
                    print("Warning: generate_negative_sample does not support false_label_override. Using custom generation.")
                    x_neg_sample = generate_positive_sample(original_x_sample, hard_negative_label, self.num_classes).to(self.device)
                
                # DEBUG: Print hard negative selection info
                if i == 0 and pair_idx == 0:  # Only for first sample/layer to avoid spam
                    print(f"True label: {true_label_sample}, Selected hard negative label: {hard_negative_label}")
                
                # Encode samples (adding batch dimension)
                batch_encoded_x_pos = self.encoder(x_pos_sample).unsqueeze(0)  # [1, T, F]
                batch_encoded_x_neg = self.encoder(x_neg_sample).unsqueeze(0)  # [1, T, F]
                
                # CRITICAL: Run separate forward passes for each layer pair to avoid graph conflicts
                # This ensures each backward() call gets a completely fresh computational graph
                
                # Positive forward pass (fresh graph)
                pos_run_data = self._run_snn_batch(batch_encoded_x_pos)
                g_pos = pos_run_data["g_scalars"][pair_idx][0]  # Single sample
                
                # Negative forward pass (fresh graph) 
                neg_run_data = self._run_snn_batch(batch_encoded_x_neg)
                g_neg = neg_run_data["g_scalars"][pair_idx][0]  # Single sample
                
                # Compute Forward-Forward loss for this layer
                loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))
                loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
                loss = self.alpha_loss * loss_pos + (1 - self.alpha_loss) * loss_neg
                
                # Backward and step for this layer (fresh graph each time)
                loss.backward()
                self.optimizers[pair_idx].step()
                
                # Store loss for monitoring
                self.current_epoch_losses[pair_idx].append(loss.item())
                
                # Optional: Clear any cached states to ensure fresh computation
                if hasattr(self.network, 'reset_state_variables'):
                    self.network.reset_state_variables()

    def train_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> None:
        # Assuming batch is a list/tuple where batch[0] is images and batch[1] is labels
        # This is standard for torchvision datasets like MNIST with torch.utils.data.DataLoader
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"train_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device)

        self.network.train()
        for linear_layer, _ in self.ff_layer_pairs:
            for param in linear_layer.parameters():
                param.requires_grad = True # Ensure grads are enabled for FF layers
        
        self._forward_forward_update_batch(original_images, true_labels)

    def train(self) -> None:
        print(f"Starting Forward-Forward training for {self.num_epochs} epochs using snntorch.")
        
        for epoch in range(self.num_epochs):
            print(f"\n=== Starting Epoch {epoch + 1}/{self.num_epochs} ===")
            self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]
            
            batch_count = 0
            total_loss = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch_data in enumerate(pbar):
                print(f"Processing batch {batch_idx + 1} with {len(batch_data[0])} samples")  # DEBUG
                
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if start_time:
                    start_time.record()
                
                self.train_step(batch_data)
                
                if start_time:
                    end_time = torch.cuda.Event(enable_timing=True)
                    end_time.record()
                    torch.cuda.synchronize()
                    print(f"Batch {batch_idx + 1} took {start_time.elapsed_time(end_time):.2f}ms")
                
                batch_count += 1
                
                # Print some loss info
                if len(self.current_epoch_losses[0]) > 0:
                    recent_loss = self.current_epoch_losses[0][-1]
                    total_loss += recent_loss
                    print(f"Batch {batch_idx + 1} recent loss: {recent_loss:.6f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"=== Epoch {epoch + 1} completed. Avg loss: {avg_loss:.6f}, Processed {batch_count} batches ===")

        # Log average loss for the epoch
        epoch_log_str = "End of Epoch Summary: "
        for pair_idx, losses in enumerate(self.current_epoch_losses):
            if losses: epoch_log_str += f" | Avg L{pair_idx}_Loss: {sum(losses) / len(losses):.4f}"
        print(epoch_log_str)

        # Optional: Save checkpoint, run validation epoch
        # if self.test_dataloader and hasattr(self, 'test_epoch') and callable(getattr(self, 'test_epoch')):
        #      self.test_epoch()
        print("Training complete.")

    # Basic test_step and test_epoch for snntorch (can be expanded)
    def test_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[int, int]:
        self.network.eval() 
        correct_predictions = 0
        
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"test_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device) # Keep as tensor

        total_samples = original_images.shape[0]


        with torch.no_grad(): 
            for i in range(total_samples):
                original_x_sample = original_images[i]
                true_label_sample = true_labels[i].item() # This was in your original test_step
                
                goodness_for_all_classes = torch.tensor(
                    [self._compute_goodness_score(original_x_sample, j) for j in range(self.num_classes)],
                    device=self.device
                )
                predicted_label = torch.argmax(goodness_for_all_classes).item()
                
                if predicted_label == true_label_sample:
                    correct_predictions += 1
        return correct_predictions, total_samples

    def test_epoch(self) -> None:
        if not self.test_dataloader:
            # Try to initialize if test_ds was provided to superclass
            if hasattr(self, 'test_ds') and self.test_ds is not None:
                 self.test_dataloader = torch.utils.data.DataLoader(
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



# BindsNET specific imports
from bindsnet.network import Network as BindsNetwork
from bindsnet.network.nodes import AbstractInput as BindsNETSpikingNode
from bindsnet.network.topology import Connection as BindsNETConnection
from bindsnet.network.monitors import Monitor

from bindsnet.pipeline.dataloader_pipeline import DataLoaderPipeline
from bindsnet.datasets.contrastive_transforms import generate_positive_sample, generate_negative_sample

class BindsNETForwardForwardPipeline(DataLoaderPipeline):
    # language=rst
    """
    Pipeline for training a BindsNET Spiking Neural Network using a Forward-Forward
    algorithm variant with hard negative mining.
    
    Note: The gradient update for BindsNET Connection weights in this pipeline
    requires a manual implementation of the Forward-Forward learning rule,
    as torch.autograd cannot differentiate through the standard BindsNET simulation.
    A placeholder for this manual gradient calculation is included.
    """

    def __init__(
        self,
        network: BindsNetwork,
        train_ds: torch.utils.data.Dataset,
        num_classes: int,
        encoder: Callable,
        time: int,
        ff_pairs_specs: List[Tuple[str, str]],
        input_layer_name: str,
        lr: float = 0.001,
        alpha_loss: float = 0.6,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        device: Optional[torch.device] = None,
        dt: float = 1.0,
        **kwargs,
    ):
        super().__init__(network=None, train_ds=train_ds, **kwargs)
        
        self.network = network
        self.network.dt = dt 

        self.num_classes = num_classes
        self.encoder = encoder
        self.sim_time = time
        self.lr = lr
        self.alpha_loss = alpha_loss
        self.input_layer_name = input_layer_name

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ff_layer_pairs: List[Tuple[BindsNETConnection, BindsNETSpikingNode]] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        
        resolved_spiking_layers_for_monitoring = {}

        for conn_name, spiking_layer_name in ff_pairs_specs:
            if conn_name not in self.network.connections:
                raise ValueError(f"Connection '{conn_name}' not found in network.")
            if spiking_layer_name not in self.network.layers:
                raise ValueError(f"Spiking layer '{spiking_layer_name}' for goodness not found in network.")
            
            connection = self.network.connections[conn_name]
            spiking_layer = self.network.layers[spiking_layer_name]

            self.ff_layer_pairs.append((connection, spiking_layer))
            
            if not isinstance(connection.w, nn.Parameter):
                connection.w = nn.Parameter(connection.w.clone().detach())
            connection.w.to(self.device) # Ensure weights are on the correct device for optimizer

            self.optimizers.append(optimizer_cls([connection.w], lr=self.lr))
            
            if spiking_layer_name not in resolved_spiking_layers_for_monitoring:
                 resolved_spiking_layers_for_monitoring[spiking_layer_name] = spiking_layer
        
        self.monitors: Dict[str, Monitor] = {}
        for layer_name, layer_obj in resolved_spiking_layers_for_monitoring.items():
            monitor = Monitor(obj=layer_obj, state_vars=["s"], time=self.sim_time)
            self.network.add_monitor(monitor, name=f"m_{layer_name}")
            self.monitors[layer_name] = monitor
        
        self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]

    def _run_snn_batch(
        self,
        batch_encoded_inputs: torch.Tensor, 
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        batch_size = batch_encoded_inputs.shape[0]

        s_traces_batch: Dict[int, torch.Tensor] = {}
        c_totals_batch: Dict[int, torch.Tensor] = {}
        g_scalars_batch: Dict[int, torch.Tensor] = {}

        for ff_pair_idx, (_, spiking_layer) in enumerate(self.ff_layer_pairs):
            num_neurons = spiking_layer.n
            s_traces_batch[ff_pair_idx] = torch.zeros(batch_size, self.sim_time, num_neurons, device=self.device)

        for i in range(batch_size):
            single_sample_encoded_input = batch_encoded_inputs[i, :, :]
            inputs = {self.input_layer_name: single_sample_encoded_input.cpu()}

            self.network.reset_state_variables()
            for monitor_name in self.monitors:
                self.monitors[monitor_name].reset_state_variables()
            
            self.network.run(inputs=inputs, time=self.sim_time, one_step=False)

            for ff_pair_idx, (_, spiking_layer_for_goodness) in enumerate(self.ff_layer_pairs):
                monitored_layer_name = None
                for name, layer_obj_in_net in self.network.layers.items():
                    if layer_obj_in_net is spiking_layer_for_goodness:
                        monitored_layer_name = name
                        break
                if monitored_layer_name is None or f"m_{monitored_layer_name}" not in self.network.monitors:
                    raise ValueError(f"Monitor for goodness layer for pair {ff_pair_idx} not found.")

                monitor_data = self.network.monitors[f"m_{monitored_layer_name}"].get("s")
                s_traces_batch[ff_pair_idx][i, :, :] = monitor_data.to(self.device)
        
        for ff_pair_idx in s_traces_batch:
            current_s_traces = s_traces_batch[ff_pair_idx]
            c_totals_batch[ff_pair_idx] = torch.sum(current_s_traces.float(), dim=1)
            if c_totals_batch[ff_pair_idx].numel() > 0:
                g_scalars_batch[ff_pair_idx] = torch.sum(c_totals_batch[ff_pair_idx]**2, dim=1)
            else:
                g_scalars_batch[ff_pair_idx] = torch.zeros(batch_size, device=self.device)
        
        return {
            "s_traces": s_traces_batch,
            "c_totals": c_totals_batch,
            "g_scalars": g_scalars_batch
        }

    def _compute_goodness_score(self, original_x: torch.Tensor, label_to_embed: int) -> float:
        temp_labeled_input_single = generate_positive_sample(
            original_x.cpu(), label_to_embed, self.num_classes
        ).to(self.device)
        
        encoded_input_single = self.encoder(temp_labeled_input_single)
        if encoded_input_single.dim() == 2:
             batch_encoded_input = encoded_input_single.unsqueeze(0)
        else:
            raise ValueError("Encoder output for single sample should be [T, Features]")

        with torch.no_grad():
            run_data = self._run_snn_batch(batch_encoded_input)
        
        total_goodness_for_sample = 0.0
        num_ff_pairs = len(self.ff_layer_pairs)
        if num_ff_pairs == 0: return 0.0

        for ff_pair_idx in range(num_ff_pairs):
            total_goodness_for_sample += run_data["g_scalars"][ff_pair_idx].item()
        
        return total_goodness_for_sample / num_ff_pairs if num_ff_pairs > 0 else 0.0

    def _forward_forward_update_batch(
        self,
        batch_original_x: torch.Tensor,
        batch_true_labels: torch.Tensor
    ) -> None:
        print(f"_forward_forward_update_batch called with batch size: {batch_original_x.shape[0]}")
        
        batch_size = batch_original_x.shape[0]
    
        # Process each sample individually to avoid graph conflicts
        for i in range(batch_size):
            print(f"Processing sample {i+1}/{batch_size}")
            
            original_x_sample = batch_original_x[i]
            true_label_sample = batch_true_labels[i].item()
            
            # Create positive and negative samples for this single sample
            x_pos_sample = generate_positive_sample(original_x_sample, true_label_sample, self.num_classes).to(self.device)
            x_neg_sample = generate_negative_sample(original_x_sample, true_label_sample, self.num_classes).to(self.device)
            
            # Encode samples (adding batch dimension)
            batch_encoded_x_pos = self.encoder(x_pos_sample).unsqueeze(0)  # [1, T, F]
            batch_encoded_x_neg = self.encoder(x_neg_sample).unsqueeze(0)  # [1, T, F]
            
            # Run SNN for positive and negative samples
            pos_run_data = self._run_snn_batch(batch_encoded_x_pos)
            neg_run_data = self._run_snn_batch(batch_encoded_x_neg)
            
            # Update each FF layer pair
            for pair_idx in range(len(self.ff_layer_pairs)):
                print(f"Updating layer pair {pair_idx}")
                
                # Zero gradients for this layer
                self.optimizers[pair_idx].zero_grad()
                
                # Compute loss for this layer and this sample
                g_pos = pos_run_data["g_scalars"][pair_idx][0]  # Single sample
                g_neg = neg_run_data["g_scalars"][pair_idx][0]  # Single sample
                
                # Forward-Forward loss for this layer
                loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))
                loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
                loss = self.alpha_loss * loss_pos + (1 - self.alpha_loss) * loss_neg
                
                # Retain graph for all but the last layer pair
                retain_graph = (pair_idx < len(self.ff_layer_pairs) - 1)
                
                # Backward and step for this layer
                loss.backward(retain_graph=retain_graph)
                self.optimizers[pair_idx].step()
                
                # Store loss for monitoring
                self.current_epoch_losses[pair_idx].append(loss.item())

    def train_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> None:
        # Assuming batch is a list/tuple where batch[0] is images and batch[1] is labels
        # This is standard for torchvision datasets like MNIST with torch.utils.data.DataLoader
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"train_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device)

        self.network.train()
        for linear_layer, _ in self.ff_layer_pairs:
            for param in linear_layer.parameters():
                param.requires_grad = True # Ensure grads are enabled for FF layers
        
        self._forward_forward_update_batch(original_images, true_labels)

    def train(self) -> None:
        print(f"Starting Forward-Forward training for {self.num_epochs} epochs using snntorch.")
        
        for epoch in range(self.num_epochs):
            print(f"\n=== Starting Epoch {epoch + 1}/{self.num_epochs} ===")
            self.current_epoch_losses = [[] for _ in self.ff_layer_pairs]
            
            batch_count = 0
            total_loss = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch_data in enumerate(pbar):
                print(f"Processing batch {batch_idx + 1} with {len(batch_data[0])} samples")  # DEBUG
                
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if start_time:
                    start_time.record()
                
                self.train_step(batch_data)
                
                if start_time:
                    end_time = torch.cuda.Event(enable_timing=True)
                    end_time.record()
                    torch.cuda.synchronize()
                    print(f"Batch {batch_idx + 1} took {start_time.elapsed_time(end_time):.2f}ms")
                
                batch_count += 1
                
                # Print some loss info
                if len(self.current_epoch_losses[0]) > 0:
                    recent_loss = self.current_epoch_losses[0][-1]
                    total_loss += recent_loss
                    print(f"Batch {batch_idx + 1} recent loss: {recent_loss:.6f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"=== Epoch {epoch + 1} completed. Avg loss: {avg_loss:.6f}, Processed {batch_count} batches ===")

        # Log average loss for the epoch
        epoch_log_str = "End of Epoch Summary: "
        for pair_idx, losses in enumerate(self.current_epoch_losses):
            if losses: epoch_log_str += f" | Avg L{pair_idx}_Loss: {sum(losses) / len(losses):.4f}"
        print(epoch_log_str)

        # Optional: Save checkpoint, run validation epoch
        # if self.test_dataloader and hasattr(self, 'test_epoch') and callable(getattr(self, 'test_epoch')):
        #      self.test_epoch()
        print("Training complete.")

    # Basic test_step and test_epoch for snntorch (can be expanded)
    def test_step(self, batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[int, int]:
        self.network.eval() 
        correct_predictions = 0
        
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError(f"test_step expects batch to be a list or tuple of at least two tensors (images, labels). Got: {type(batch)}")

        original_images = batch[0].to(self.device)
        true_labels = batch[1].to(self.device) # Keep as tensor

        total_samples = original_images.shape[0]


        with torch.no_grad(): 
            for i in range(total_samples):
                original_x_sample = original_images[i]
                true_label_sample = true_labels[i].item() # This was in your original test_step
                
                goodness_for_all_classes = torch.tensor(
                    [self._compute_goodness_score(original_x_sample, j) for j in range(self.num_classes)],
                    device=self.device
                )
                predicted_label = torch.argmax(goodness_for_all_classes).item()
                
                if predicted_label == true_label_sample:
                    correct_predictions += 1
        return correct_predictions, total_samples

    def test_epoch(self) -> None:
        if not self.test_dataloader:
            # Try to initialize if test_ds was provided to superclass
            if hasattr(self, 'test_ds') and self.test_ds is not None:
                 self.test_dataloader = torch.utils.data.DataLoader(
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