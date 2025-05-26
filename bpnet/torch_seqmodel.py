import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import torch.nn.functional as F # For wn summary softmax
try:
    from bpnet.utils.dinuc_shuffle import dinuc_shuffle_pytorch
except ImportError:
    # Fallback for environments where utils might not be in path yet, or for isolated testing
    def dinuc_shuffle_pytorch(batch_tensor):
        print("Warning: Using fallback dinuc_shuffle_pytorch. Ensure bpnet.utils is in PYTHONPATH.")
        return torch.flip(batch_tensor, dims=[1]) # Simple shuffle placeholder

try:
    from captum.attr import DeepLift, Saliency, InputXGradient
except ImportError:
    print("Warning: Captum is not installed. `contrib_score_all` will not be usable.")
    DeepLift, Saliency, InputXGradient = None, None, None # Placeholders


class TorchSeqModel(nn.Module):
    def __init__(self, body, heads, tasks, seqlen=None):
        """
        Args:
            body (nn.Module): The shared body of the model.
            heads (nn.ModuleDict): A ModuleDict where keys are task names 
                                   and values are the head nn.Module for that task.
                                   Each head module is expected to output a dictionary 
                                   of tensors (e.g., {'profile': ..., 'counts': ...})
                                   AND set a `self.last_interpretation_layers` dict.
            tasks (list): A list of task names (strings).
            seqlen (int, optional): Input sequence length. Defaults to None.
        """
        super(TorchSeqModel, self).__init__()
        self.body = body
        self.heads = heads
        self.tasks = tasks 
        self.seqlen = seqlen
        self.return_interpretation_layers = False # Flag to control output of forward
        
        if not isinstance(self.heads, nn.ModuleDict):
            raise TypeError("heads must be an nn.ModuleDict mapping task names to head modules.")
        
        # Ensure all tasks listed in `tasks` are present in `heads`
        for task_name in self.tasks:
            if task_name not in self.heads:
                raise ValueError(f"Task '{task_name}' not found in heads ModuleDict.")

    def forward(self, x, bias_inputs=None):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, num_channels).
            bias_inputs (torch.Tensor, optional): Bias inputs. Not currently used.
        Returns:
            If self.return_interpretation_layers is True:
                (predictions_dict, interpretation_layers_dict)
            Else:
                predictions_dict
        """
        # Assuming body and heads (like those in torch_layers.py) handle input permutation
        # (e.g. from N,L,C to N,C,L) internally if needed.
        
        # Body pass
        # Potentially, body could also set its own `last_interpretation_layers`
        # self.body.last_interpretation_layers = {} # If body is modified to do so
        bottleneck = self.body(x)
        
        # Collect interpretation layers from body if available
        # This requires body to have a `last_interpretation_layers` attribute.
        current_interpretation_layers = {}
        if hasattr(self.body, 'last_interpretation_layers') and self.body.last_interpretation_layers:
            for key, val in self.body.last_interpretation_layers.items():
                 current_interpretation_layers[f"body/{key}"] = val


        predictions_dict = OrderedDict()
        for task_name in self.tasks:
            head_module = self.heads[task_name]
            
            # Head pass
            head_pred_outputs = head_module(bottleneck) # Assuming head returns dict of final preds
            
            if not isinstance(head_pred_outputs, dict):
                raise TypeError(f"Head for task '{task_name}' should return a dictionary of final predictions. "
                                f"Got {type(head_pred_outputs)}.")
            
            for output_type, tensor_val in head_pred_outputs.items():
                predictions_dict[f"{task_name}/{output_type}"] = tensor_val
            
            # Collect interpretation layers from this head
            if hasattr(head_module, 'last_interpretation_layers') and head_module.last_interpretation_layers:
                for interp_key, interp_val in head_module.last_interpretation_layers.items():
                    current_interpretation_layers[f"{task_name}/{interp_key}"] = interp_val
            else: # Fallback: store final predictions also as 'final_output' interpretation layer
                for output_type, tensor_val in head_pred_outputs.items():
                     current_interpretation_layers[f"{task_name}/{output_type}/final_output"] = tensor_val.detach().clone()


        if self.return_interpretation_layers:
            # Store on self primarily for _CaptumModelWrapper to access easily in the same forward call context
            # This might be cleared at the start of the next forward pass if needed.
            self._current_interpretation_layers_for_captum = current_interpretation_layers
            return predictions_dict, current_interpretation_layers
        else:
            return predictions_dict

    @torch.no_grad()
    def predict(self, x, batch_size=None, device='cpu', verbose=False):
        """
        Prediction method for the model.
        Args:
            x (np.ndarray or torch.Tensor): Input data.
            batch_size (int, optional): Batch size for prediction. If None, predicts all at once.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
            verbose (bool, optional): If True, prints progress. Defaults to False.
        Returns:
            dict: A dictionary of output NumPy arrays, similar to forward pass output structure.
        """
        self.eval()
        self.to(device)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(device)

        if batch_size is None:
            batch_size = x.shape[0]

        num_batches = (x.shape[0] + batch_size - 1) // batch_size
        
        all_preds_list = [] # Stores prediction dictionaries from each batch

        # Ensure forward returns only predictions for predict method
        original_return_interp_flag = self.return_interpretation_layers
        self.return_interpretation_layers = False

        for i in range(num_batches):
            if verbose:
                print(f"Processing batch {i+1}/{num_batches}")
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, x.shape[0])
            batch_x = x[start_idx:end_idx]
            
            batch_preds_dict = self.forward(batch_x) # Now returns only predictions_dict
            all_preds_list.append(batch_preds_dict)
        
        self.return_interpretation_layers = original_return_interp_flag # Restore flag

        if not all_preds_list:
            return {}

        # Concatenate prediction results from all batches
        final_outputs_np = {}
        # Get keys from the first batch's prediction dictionary
        output_keys = all_preds_list[0].keys() 

        for key in output_keys:
            concatenated_tensor = torch.cat([d[key] for d in all_preds_list], dim=0)
            final_outputs_np[key] = concatenated_tensor.cpu().numpy()
            
        return final_outputs_np

    @classmethod
    def load_model_from_dir(cls, model_dir):
        # TODO: Implement actual model loading logic.
        # This should reconstruct the model architecture (body, heads, tasks, seqlen)
        # from config files in model_dir and then load the state_dict.
        print(f"Placeholder: TorchSeqModel.load_model_from_dir called for {model_dir}")
        print("Actual model loading (architecture reconstruction and state_dict loading) needs to be implemented here.")
        # For now, return None or raise NotImplementedError to indicate it's a placeholder.
        # To allow the flow for TorchBPNetSeqModel.from_mdir, we might need to return
        # a dummy/uninitialized model for now if the caller expects an instance.
        # However, for this subtask, let's make it clear it's not implemented.
        raise NotImplementedError("TorchSeqModel.load_model_from_dir is not yet implemented.")
        # return None # Or a dummy model if required by TorchBPNetSeqModel.from_mdir structure

if __name__ == '__main__':
    # Example Usage (requires dummy body and head modules from bpnet.torch_layers)
    # Assuming bpnet.torch_layers are in python path
    # from bpnet.torch_layers import DilatedConv1D, GlobalAvgPoolFCN # FCN, DeConv1D etc.

    # This example won't run directly without torch_layers being importable
    # and having suitable dummy implementations if real ones are too complex for a quick test.
    print("TorchSeqModel basic structure defined.")

    # Dummy Body Module
    class DummyBody(nn.Module):
        def __init__(self, in_channels, out_channels, length):
            super().__init__()
            # Example: a conv layer that changes channel size but keeps length for simplicity
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) 
            self.target_length = length # Not used if conv1d padding=same or kernel=1

        def forward(self, x_in): # x_in is (N, L, C_in)
            x_perm = x_in.permute(0,2,1) # (N, C_in, L)
            out = self.conv(x_perm) # (N, C_out, L)
            return out.permute(0,2,1) # (N, L, C_out)

    # Dummy Head Module (emulating output for profile and counts)
    class DummyHead(nn.Module):
        def __init__(self, in_features, profile_len, num_profile_channels=1, num_counts_channels=1):
            super().__init__()
            # For profile, let's assume it processes (N, L, C_in_bottleneck) to (N, L_profile, C_profile_out)
            # For counts, let's assume it processes (N, L, C_in_bottleneck) to (N, C_counts_out) via pooling + linear
            self.profile_conv = nn.Conv1d(in_features, num_profile_channels, kernel_size=1) # keeps length
            self.profile_target_len = profile_len
            
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.counts_linear = nn.Linear(in_features, num_counts_channels)

        def forward(self, bottleneck_x): # bottleneck_x is (N, L_bottleneck, C_bottleneck)
            # Profile part
            bottleneck_perm = bottleneck_x.permute(0,2,1) # (N, C_bottleneck, L_bottleneck)
            profile_out_perm = self.profile_conv(bottleneck_perm) # (N, num_profile_channels, L_bottleneck)
            # In a real scenario, profile_out_perm might need length adjustment to profile_target_len
            # For this dummy, assume L_bottleneck is same as profile_target_len
            profile_final = profile_out_perm.permute(0,2,1) # (N, L_bottleneck, num_profile_channels)
            if profile_final.shape[1] != self.profile_target_len :
                 # Simple interpolation if lengths don't match (very basic)
                 profile_final_perm = profile_final.permute(0,2,1) # N, C, L
                 profile_final_resized = F.interpolate(profile_final_perm, size=self.profile_target_len, mode='linear', align_corners=False)
                 profile_final = profile_final_resized.permute(0,2,1)


            # Counts part
            pooled = self.pool(bottleneck_perm) # (N, C_bottleneck, 1)
            squeezed = pooled.squeeze(-1) # (N, C_bottleneck)
            counts_final = self.counts_linear(squeezed) # (N, num_counts_channels)
            # Counts are usually (N, num_tasks) or (N,1) per task head.
            # If num_counts_channels > 1, it means multiple count outputs for this task-head.

            return {'profile': profile_final, 'counts': counts_final}

    # Config
    N_BATCH, SEQ_LEN, IN_CHANNELS = 2, 100, 4
    BODY_OUT_CHANNELS = 16
    PROFILE_LEN = SEQ_LEN # For simplicity in dummy head
    
    tasks_list = ['task1', 'task2']

    # Create model components
    dummy_body_module = DummyBody(IN_CHANNELS, BODY_OUT_CHANNELS, SEQ_LEN)
    
    heads_dict = nn.ModuleDict()
    for task in tasks_list:
        # Each task gets its own head instance
        # Assuming profile output has 1 channel, counts output has 1 channel per task head
        heads_dict[task] = DummyHead(BODY_OUT_CHANNELS, PROFILE_LEN, 
                                     num_profile_channels=1, num_counts_channels=1)

    # Instantiate TorchSeqModel
    torch_seq_model = TorchSeqModel(body=dummy_body_module, heads=heads_dict, tasks=tasks_list, seqlen=SEQ_LEN)

    # Create dummy input
    dummy_x = torch.randn(N_BATCH, SEQ_LEN, IN_CHANNELS)

    # Test forward pass
    print(f"\nTesting TorchSeqModel forward pass with input shape: {dummy_x.shape}")
    outputs_forward = torch_seq_model(dummy_x)
    print("Forward pass output keys:", outputs_forward.keys())
    for key, val in outputs_forward.items():
        print(f" - Output '{key}' shape: {val.shape}")
        assert val.ndim > 0 # Basic check

    # Test predict method
    print(f"\nTesting TorchSeqModel predict method with input shape: {dummy_x.shape}")
    outputs_predict = torch_seq_model.predict(dummy_x.numpy(), batch_size=1, device='cpu', verbose=True)
    print("Predict method output keys:", outputs_predict.keys())
    for key, val_np in outputs_predict.items():
        print(f" - Output '{key}' shape: {val_np.shape} (NumPy array)")
        assert isinstance(val_np, np.ndarray)

    print("\nTorchSeqModel tests completed.")
