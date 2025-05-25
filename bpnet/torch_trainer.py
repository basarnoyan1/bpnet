import torch
import torch.optim as optim
import torch.nn as nn # For dummy model/loss if needed for testing
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

# Assuming bpnet.torch_seqmodel and bpnet.torch_losses are accessible in the python path
# For testing, we might need to define dummy versions or ensure they can be imported.
try:
    from bpnet.torch_seqmodel import TorchSeqModel 
    # For testing, we might use the dummy models from torch_seqmodel's main block.
    # Let's try to import them if they are made accessible or redefine simplified ones here.
    from bpnet.torch_seqmodel import DummyBody as TestDummyBody # Assuming it's importable or defined in __main__
    from bpnet.torch_seqmodel import DummyHead as TestDummyHead
except ImportError:
    print("Warning: Could not import TorchSeqModel or its dummy components for testing. Define local dummies.")
    # Define minimal dummy TorchSeqModel if import fails, for structure.
    class TestDummyBody(nn.Module):
        def __init__(self, in_channels, out_channels, length):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.target_length = length
        def forward(self, x_in): # x_in is (N, L, C_in)
            x_perm = x_in.permute(0,2,1)
            out = self.conv(x_perm)
            return out.permute(0,2,1)

    class TestDummyHead(nn.Module):
        def __init__(self, in_features, profile_len, num_profile_channels=1, num_counts_channels=1):
            super().__init__()
            self.profile_conv = nn.Conv1d(in_features, num_profile_channels, kernel_size=1)
            self.profile_target_len = profile_len
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.counts_linear = nn.Linear(in_features, num_counts_channels)
        def forward(self, bottleneck_x): # (N, L_bottleneck, C_bottleneck)
            bottleneck_perm = bottleneck_x.permute(0,2,1)
            profile_out_perm = self.profile_conv(bottleneck_perm)
            profile_final = profile_out_perm.permute(0,2,1)
            if profile_final.shape[1] != self.profile_target_len:
                 profile_final_perm = profile_final.permute(0,2,1)
                 profile_final_resized = torch.nn.functional.interpolate(profile_final_perm, size=self.profile_target_len, mode='linear', align_corners=False)
                 profile_final = profile_final_resized.permute(0,2,1)
            pooled = self.pool(bottleneck_perm)
            squeezed = pooled.squeeze(-1)
            counts_final = self.counts_linear(squeezed)
            return {'profile': profile_final, 'counts': counts_final}

    class TorchSeqModel(nn.Module): # Minimal version for type hint if main import fails
        def __init__(self, body, heads, tasks, seqlen=None):
            super().__init__()
            self.body = body
            self.heads = heads
            self.tasks = tasks
            self.seqlen = seqlen
        def forward(self, x, bias_inputs=None):
            bottleneck = self.body(x)
            outputs = {}
            for task_name in self.tasks:
                head_task_outputs = self.heads[task_name](bottleneck)
                for output_type, tensor_val in head_task_outputs.items():
                    outputs[f"{task_name}/{output_type}"] = tensor_val
            return outputs


try:
    from bpnet import torch_losses # For actual loss functions
except ImportError:
    print("Warning: Could not import bpnet.torch_losses. Define local dummy losses for testing.")
    # Define minimal dummy loss if import fails
    class DummyLoss(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, pred, target):
            return F.mse_loss(pred, target)
    torch_losses = {'ProfileMultinomialNLL': DummyLoss, 'CountsMultinomialNLL': DummyLoss}


def train_model(model, dataloader, optimizer, loss_configs, num_epochs, device, metrics_configs=None,
                validation_dataloader=None): # Added validation_dataloader
    """
    Trains a PyTorch model.
    Args:
        model (TorchSeqModel): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_configs (dict): Maps target names to (loss_function, weight).
        num_epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu', 'cuda').
        metrics_configs (dict, optional): Maps target names to list of metric functions.
        validation_dataloader (DataLoader, optional): DataLoader for validation data.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in dataloader:
            # Assuming batch_inputs is the sequence tensor. If it's a dict, adjust model call.
            batch_inputs = batch_inputs.to(device)
            
            # Move all target tensors to device
            for target_name in batch_targets:
                batch_targets[target_name] = batch_targets[target_name].to(device)

            optimizer.zero_grad()
            
            # Get model predictions
            # Assuming model.forward takes the sequence tensor directly.
            predictions = model(batch_inputs) 
            
            batch_total_loss = torch.tensor(0.0, device=device)
            for target_name, (loss_fn, loss_weight) in loss_configs.items():
                if target_name not in predictions:
                    print(f"Warning: Target '{target_name}' not found in model predictions. Skipping loss.")
                    continue
                if target_name not in batch_targets:
                    print(f"Warning: Target '{target_name}' not found in batch targets. Skipping loss.")
                    continue
                
                # The loss_fn from torch_losses might be a nn.Module itself (e.g. ProfileMultinomialNLL)
                # or a simple function.
                # If it's a nn.Module, it should be on the correct device.
                if isinstance(loss_fn, nn.Module):
                    loss_fn.to(device) # Ensure loss module is on device
                
                current_loss = loss_fn(batch_targets[target_name], predictions[target_name]) # Standard (true, pred)
                batch_total_loss += loss_weight * current_loss
            
            if batch_total_loss.requires_grad: # Ensure there's something to backprop
                 batch_total_loss.backward()
                 optimizer.step()
            
            epoch_loss += batch_total_loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_epoch_loss:.4f}")

        if validation_dataloader:
            print("Running validation...")
            eval_results = evaluate_model(model, validation_dataloader, loss_configs, metrics_configs, device)
            # eval_loss = eval_results['loss']
            # eval_metrics = eval_results['metrics']
            # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {eval_loss:.4f}")
            # for target_name, metrics in eval_metrics.items():
            #     for metric_name, value in metrics.items():
            #         print(f"  {target_name} - {metric_name}: {value:.4f}")
            # (evaluate_model will print its own summary for now)


def evaluate_model(model, dataloader, loss_configs, metrics_configs, device):
    """
    Evaluates a PyTorch model.
    Args:
        model (TorchSeqModel): The model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        loss_configs (dict): Maps target names to (loss_function, weight).
        metrics_configs (dict): Maps target names to list of metric functions.
        device (str): Device to evaluate on ('cpu', 'cuda').
    Returns:
        dict: Contains 'loss' and 'metrics'.
    """
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Initialize metrics storage
    # epoch_metrics = {target_name: {metric_fn.__name__: 0.0 for metric_fn in fns} 
    #                  for target_name, fns in metrics_configs.items()} if metrics_configs else {}
    # More robust: store list of metric values per batch, then average.
    # Or, if metric functions are like torchmetrics objects, they handle accumulation.
    # For this subtask, assume metric_fn returns a scalar; we average these scalars.
    
    # Using defaultdict for easier accumulation of potentially multiple metrics per target
    # Example: epoch_metrics_sum['taskA/profile']['mse_metric'] will sum values
    epoch_metrics_sum = defaultdict(lambda: defaultdict(float))
    epoch_metrics_count = defaultdict(lambda: defaultdict(int))


    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            for target_name in batch_targets:
                batch_targets[target_name] = batch_targets[target_name].to(device)

            predictions = model(batch_inputs)
            
            batch_total_loss = torch.tensor(0.0, device=device)
            for target_name, (loss_fn, loss_weight) in loss_configs.items():
                if target_name not in predictions or target_name not in batch_targets:
                    continue 
                if isinstance(loss_fn, nn.Module):
                    loss_fn.to(device)
                current_loss = loss_fn(batch_targets[target_name], predictions[target_name])
                batch_total_loss += loss_weight * current_loss
            
            total_loss += batch_total_loss.item()
            
            if metrics_configs:
                for target_name, metric_fn_list in metrics_configs.items():
                    if target_name not in predictions or target_name not in batch_targets:
                        continue
                    for metric_fn in metric_fn_list:
                        metric_name = getattr(metric_fn, '__name__', 'unknown_metric')
                        try:
                            metric_value = metric_fn(predictions[target_name], batch_targets[target_name])
                            # Assuming metric_value is a scalar torch tensor or float
                            if isinstance(metric_value, torch.Tensor):
                                metric_value = metric_value.item()
                            epoch_metrics_sum[target_name][metric_name] += metric_value
                            epoch_metrics_count[target_name][metric_name] += 1
                        except Exception as e:
                            print(f"Error calculating metric {metric_name} for {target_name}: {e}")


            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    avg_metrics = defaultdict(dict)
    if metrics_configs:
        for target_name, metrics_sum_dict in epoch_metrics_sum.items():
            for metric_name, metric_sum in metrics_sum_dict.items():
                count = epoch_metrics_count[target_name][metric_name]
                avg_metrics[target_name][metric_name] = metric_sum / count if count > 0 else 0

    print(f"Evaluation Summary: Average Loss: {avg_loss:.4f}")
    if metrics_configs:
        for target_name, metrics_values in avg_metrics.items():
            print(f"  Metrics for {target_name}:")
            for metric_name, value in metrics_values.items():
                print(f"    {metric_name}: {value:.4f}")
                
    return {'loss': avg_loss, 'metrics': dict(avg_metrics)}


if __name__ == '__main__':
    print("Testing PyTorch Trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Dummy Data
    N_BATCH_TRAIN, N_BATCH_VAL = 64, 32 
    SEQ_LEN, IN_CHANNELS_SEQ = 100, 4
    PROFILE_LEN_MODEL = SEQ_LEN # Assuming model output profile length is same as input
    
    # Dummy sequence input
    dummy_train_seq = torch.randn(N_BATCH_TRAIN * 5, SEQ_LEN, IN_CHANNELS_SEQ) # 5 batches for train
    dummy_val_seq = torch.randn(N_BATCH_VAL * 2, SEQ_LEN, IN_CHANNELS_SEQ)   # 2 batches for val

    # Dummy targets (assuming one task 'task1' with 'profile' and 'counts')
    # Profile targets: (N, L, C_prof_task_head_output) - C_prof_task_head_output usually 1 for profile track
    # Counts targets: (N, C_counts_task_head_output) - C_counts_task_head_output usually 1 for counts value
    
    # For ProfileMultinomialNLL, y_true is (N, L, 2) where last dim is [counts_for_multinomial, true_profile_for_mse]
    # Let's make a dummy loss that uses this structure for 'task1/profile'.
    # And a dummy loss for 'task1/counts' that uses just counts.
    
    # Train targets
    train_targets_task1_profile = torch.rand(N_BATCH_TRAIN * 5, PROFILE_LEN_MODEL, 2) 
    train_targets_task1_profile[..., 0] = torch.randint(0,10, (N_BATCH_TRAIN*5, PROFILE_LEN_MODEL)).float() # Counts part for ProfileMultinomialNLL
    train_targets_task1_counts = torch.rand(N_BATCH_TRAIN * 5, 1) * 10 # Counts part for CountsMultinomialNLL
    
    train_dataset = TensorDataset(dummy_train_seq, train_targets_task1_profile, train_targets_task1_counts)
    # Need to wrap this in a dataloader that yields (inputs, targets_dict)
    
    # Custom collate or dataset to yield dicts
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, seqs, prof_targets, count_targets):
            self.seqs = seqs
            self.prof_targets = prof_targets
            self.count_targets = count_targets
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return self.seqs[idx], {"task1/profile": self.prof_targets[idx], 
                                    "task1/counts": self.count_targets[idx]}

    train_dataset_custom = CustomDataset(dummy_train_seq, train_targets_task1_profile, train_targets_task1_counts)
    train_dataloader = DataLoader(train_dataset_custom, batch_size=N_BATCH_TRAIN, shuffle=True)

    # Validation targets
    val_targets_task1_profile = torch.rand(N_BATCH_VAL * 2, PROFILE_LEN_MODEL, 2)
    val_targets_task1_profile[..., 0] = torch.randint(0,10, (N_BATCH_VAL*2, PROFILE_LEN_MODEL)).float()
    val_targets_task1_counts = torch.rand(N_BATCH_VAL * 2, 1) * 10
    val_dataset_custom = CustomDataset(dummy_val_seq, val_targets_task1_profile, val_targets_task1_counts)
    val_dataloader = DataLoader(val_dataset_custom, batch_size=N_BATCH_VAL, shuffle=False)


    # 2. Dummy Model (using simplified local versions if imports failed)
    BODY_OUT_CHANNELS_TEST = 16
    tasks_list_test = ['task1']
    
    dummy_body = TestDummyBody(IN_CHANNELS_SEQ, BODY_OUT_CHANNELS_TEST, SEQ_LEN)
    dummy_heads = nn.ModuleDict({
        'task1': TestDummyHead(BODY_OUT_CHANNELS_TEST, PROFILE_LEN_MODEL, 
                               num_profile_channels=1, num_counts_channels=1) 
        # ^This TestDummyHead outputs {'profile': (N,L,1), 'counts': (N,1)}
        # ProfileMultinomialNLL expects pred to be (N,L,2) [logits, pred_profile]
        # CountsMultinomialNLL expects pred to be (N,L,C) [usually C=1 for counts head]
        # The current TestDummyHead output structure needs adjustment for these losses,
        # or the losses need to be adapted to the head's output structure.
        # Let's assume the head for 'task1' should output what ProfileMultinomialNLL and CountsMultinomialNLL expect
        # for keys 'task1/profile' and 'task1/counts' respectively.
        # This means the 'task1' head should output a dict like:
        # {'profile': tensor_for_ProfileNLL, 'counts': tensor_for_CountsNLL}
        # TestDummyHead currently outputs {'profile':(N,L,1), 'counts':(N,1)}
        # Let's adjust the dummy head or use specific heads for specific loss outputs.
    })

    # Re-define DummyHead for this test to output what losses expect for 'task1/profile' and 'task1/counts'
    class TestHeadForProfileLoss(nn.Module): # Outputs data for ProfileMultinomialNLL
        def __init__(self, in_features, profile_len):
            super().__init__()
            # For ProfileMultinomialNLL, y_pred is (N, L, 2) [logits, pred_profile]
            self.conv_logits = nn.Conv1d(in_features, 1, kernel_size=1)
            self.conv_pred_prof = nn.Conv1d(in_features, 1, kernel_size=1)
            self.profile_len = profile_len
        def forward(self, x_bottleneck): # (N, L_bottle, C_bottle)
            x_perm = x_bottleneck.permute(0,2,1) # (N, C_bottle, L_bottle)
            logits = self.conv_logits(x_perm).permute(0,2,1) # (N, L_bottle, 1)
            pred_prof = self.conv_pred_prof(x_perm).permute(0,2,1) # (N, L_bottle, 1)
            # Ensure output length matches profile_len (e.g. via interpolation or careful convs)
            if logits.shape[1] != self.profile_len:
                logits = torch.nn.functional.interpolate(logits.permute(0,2,1), size=self.profile_len).permute(0,2,1)
                pred_prof = torch.nn.functional.interpolate(pred_prof.permute(0,2,1), size=self.profile_len).permute(0,2,1)
            return torch.cat([logits, pred_prof], dim=-1) # (N, L_profile, 2)

    class TestHeadForCountsLoss(nn.Module): # Outputs data for CountsMultinomialNLL or Poisson...
        def __init__(self, in_features, profile_len_counts): # profile_len_counts for (N,L,C) output
            super().__init__()
            # For CountsMultinomialNLL, preds are (N, L, C) -> usually C=1 for counts task
            self.conv_counts = nn.Conv1d(in_features, 1, kernel_size=1)
            self.profile_len_counts = profile_len_counts
        def forward(self, x_bottleneck): # (N, L_bottle, C_bottle)
            x_perm = x_bottleneck.permute(0,2,1)
            counts_preds = self.conv_counts(x_perm).permute(0,2,1) # (N, L_bottle, 1)
            if counts_preds.shape[1] != self.profile_len_counts:
                 counts_preds = torch.nn.functional.interpolate(counts_preds.permute(0,2,1), size=self.profile_len_counts).permute(0,2,1)
            return counts_preds # (N, L_counts, 1)

    # Create a model where heads are structured such that their output keys
    # directly match the loss_configs keys.
    # TorchSeqModel's forward will produce:
    # {'task1_profile/output_from_profile_head': tensor_for_ProfileNLL,
    #  'task1_counts/output_from_counts_head': tensor_for_CountsNLL}
    # So, loss_configs keys should match these.
    
    # This requires a slightly different model structure than one head per task.
    # Let's use the original TorchSeqModel assumption: one head per task, head returns dict.
    # And that head's output dict keys are 'profile' and 'counts'.
    # So, predictions dict from model will be {'task1/profile': ..., 'task1/counts': ...}
    # This is what the current trainer expects for loss_configs keys.

    # The TestDummyHead already outputs {'profile':(N,L,1), 'counts':(N,1)}
    # This is NOT what ProfileMultinomialNLL expects for 'profile' (needs (N,L,2))
    # And CountsMultinomialNLL expects (N,L,C) for counts, TestDummyHead gives (N,C)
    # This test setup is getting complicated due to matching head outputs to loss inputs.
    
    # Simpler test model:
    class SimplerDummyHead(nn.Module):
        def __init__(self, in_features, profile_len):
            super().__init__()
            # Output for 'profile' key (used by ProfileMultinomialNLL)
            self.conv_logits = nn.Conv1d(in_features, 1, kernel_size=1)
            self.conv_pred_prof = nn.Conv1d(in_features, 1, kernel_size=1)
            # Output for 'counts' key (used by CountsMultinomialNLL)
            self.conv_counts_preds = nn.Conv1d(in_features, 1, kernel_size=1) # (N,L,1)
            self.profile_len = profile_len

        def forward(self, x_bottleneck):
            x_perm = x_bottleneck.permute(0,2,1)
            
            logits = self.conv_logits(x_perm).permute(0,2,1)
            pred_prof = self.conv_pred_prof(x_perm).permute(0,2,1)
            if logits.shape[1] != self.profile_len: # Ensure length
                logits = torch.nn.functional.interpolate(logits.permute(0,2,1), size=self.profile_len).permute(0,2,1)
                pred_prof = torch.nn.functional.interpolate(pred_prof.permute(0,2,1), size=self.profile_len).permute(0,2,1)
            profile_output_for_loss = torch.cat([logits, pred_prof], dim=-1) # (N,L,2)

            counts_preds_for_loss = self.conv_counts_preds(x_perm).permute(0,2,1) # (N,L,1)
            if counts_preds_for_loss.shape[1] != self.profile_len: # Ensure length
                counts_preds_for_loss = torch.nn.functional.interpolate(counts_preds_for_loss.permute(0,2,1), size=self.profile_len).permute(0,2,1)

            return {'profile': profile_output_for_loss, 'counts': counts_preds_for_loss}

    dummy_heads_revised = nn.ModuleDict({
        'task1': SimplerDummyHead(BODY_OUT_CHANNELS_TEST, PROFILE_LEN_MODEL)
    })
    test_model = TorchSeqModel(body=dummy_body, heads=dummy_heads_revised, tasks=tasks_list_test, seqlen=SEQ_LEN)

    # 3. Optimizer
    optimizer = optim.Adam(test_model.parameters(), lr=1e-3)

    # 4. Loss Configs (using dummy losses for now if bpnet.torch_losses not fully set up for these head outputs)
    loss_configs_test = {
        "task1/profile": (torch_losses.ProfileMultinomialNLL(p_task_weight=0.1), 1.0),
        # CountsMultinomialNLL expects true_counts (N,L,C) and preds (N,L,C) for its multinomial part.
        # Here, true_counts from dataloader for "task1/counts" is (N,1).
        # Preds for "task1/counts" from SimplerDummyHead is (N,L,1).
        # This is a mismatch. CountsMultinomialNLL's multinomial part needs seq-wise counts and logits.
        # The MSE/Poisson part uses total sums.
        # For simplicity, let's use a basic MSE for the "counts" part in this test.
        "task1/counts": (nn.MSELoss(), 0.5) # Target for this is (N,1), pred is (N,L,1). Needs adjustment.
                                            # Let's assume counts head outputs (N,1) and target is (N,1)
                                            # For this, SimplerDummyHead.counts would need to be (N,1) e.g. via pooling.
    }
    
    # Adjust SimplerDummyHead for counts to be (N,1) for MSE with (N,1) target
    class SimplerDummyHeadFinal(nn.Module): # Adjusted for test
        def __init__(self, in_features, profile_len):
            super().__init__()
            self.conv_logits = nn.Conv1d(in_features, 1, kernel_size=1)
            self.conv_pred_prof = nn.Conv1d(in_features, 1, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool1d(1) # For counts
            self.linear_counts = nn.Linear(in_features, 1) # For counts (N,1)
            self.profile_len = profile_len
        def forward(self, x_bottleneck): # (N,L,C_bottle)
            x_perm = x_bottleneck.permute(0,2,1) # (N,C_bottle,L)
            logits = self.conv_logits(x_perm).permute(0,2,1)
            pred_prof = self.conv_pred_prof(x_perm).permute(0,2,1)
            if logits.shape[1] != self.profile_len:
                logits = torch.nn.functional.interpolate(logits.permute(0,2,1),size=self.profile_len).permute(0,2,1)
                pred_prof = torch.nn.functional.interpolate(pred_prof.permute(0,2,1),size=self.profile_len).permute(0,2,1)
            profile_output = torch.cat([logits, pred_prof], dim=-1)
            
            counts_pooled = self.pool(x_perm).squeeze(-1) # (N,C_bottle)
            counts_output = self.linear_counts(counts_pooled) # (N,1)
            return {'profile': profile_output, 'counts': counts_output}

    dummy_heads_final = nn.ModuleDict({'task1': SimplerDummyHeadFinal(BODY_OUT_CHANNELS_TEST, PROFILE_LEN_MODEL)})
    test_model_final = TorchSeqModel(body=dummy_body, heads=dummy_heads_final, tasks=tasks_list_test, seqlen=SEQ_LEN)
    optimizer_final = optim.Adam(test_model_final.parameters(), lr=1e-3)


    # 5. Metrics Configs (dummy)
    def dummy_mse_metric(pred, target): return F.mse_loss(pred, target).item()
    def dummy_mae_metric(pred, target): return F.l1_loss(pred, target).item()
    
    metrics_configs_test = {
        "task1/profile": [dummy_mse_metric, dummy_mae_metric],
        "task1/counts": [dummy_mse_metric]
    }

    # 6. Train
    print("\nStarting dummy training...")
    train_model(test_model_final, train_dataloader, optimizer_final, loss_configs_test, 
                num_epochs=2, device=device, metrics_configs=metrics_configs_test, # Added metrics here
                validation_dataloader=val_dataloader)

    # 7. Evaluate (normally on a separate validation/test set)
    print("\nStarting dummy evaluation...")
    evaluate_model(test_model_final, val_dataloader, loss_configs_test, metrics_configs_test, device)
    
    print("\nPyTorch Trainer tests completed.")
