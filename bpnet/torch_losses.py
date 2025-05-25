import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Poisson

def multinomial_nll_pytorch(true_counts, logits):
    """
    Computes the multinomial negative log-likelihood loss.

    Args:
        true_counts: Tensor of observed counts, shape (batch, seqlen, channels).
        logits: Tensor of predicted logits, shape (batch, seqlen, channels).

    Returns:
        Scalar tensor representing the mean loss over the batch.
    """
    if true_counts.shape != logits.shape:
        raise ValueError(f"Shapes of true_counts {true_counts.shape} and logits {logits.shape} must match.")

    # Permute to (batch, channels, seqlen) to match Keras implementation's distribution setup
    true_counts_perm = true_counts.permute(0, 2, 1)
    logits_perm = logits.permute(0, 2, 1)

    # total_count for Multinomial: sum of true counts over the sequence length for each channel
    # This results in a total_count for each batch entry and each channel.
    # Shape: (batch, channels)
    # Ensure total_count is float for Multinomial distribution, true_counts might be int/long
    total_count = torch.sum(true_counts_perm, dim=2).float() 

    # Create the Multinomial distribution.
    # logits_perm provides the event logits for 'seqlen' classes.
    # total_count provides the number of trials for each of (batch, channel) instance.
    dist = Multinomial(total_count=total_count, logits=logits_perm)

    # Calculate log probability of observing true_counts_perm.
    # true_counts_perm has shape (batch, channels, seqlen), matching dist.sample() shape.
    # log_prob output shape will be (batch, channels)
    log_probs = dist.log_prob(true_counts_perm)

    # Sum log_probs and normalize by batch size.
    # The negative sign makes it a negative log-likelihood.
    batch_size = true_counts.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=true_counts.device) # Handle empty batch
    
    loss = -torch.sum(log_probs) / batch_size
    
    return loss

if __name__ == '__main__':
    # Test multinomial_nll_pytorch
    print("Testing multinomial_nll_pytorch...")
    batch_size, seq_len, num_channels = 2, 10, 3

    # Example true counts (integers)
    true_counts_example = torch.randint(0, 5, (batch_size, seq_len, num_channels)).float()
    # Ensure some counts are non-zero for total_count
    true_counts_example[0,0,0] = 1.0 
    true_counts_example[1,0,0] = 1.0


    # Example logits (floats)
    logits_example = torch.randn(batch_size, seq_len, num_channels)

    loss_val = multinomial_nll_pytorch(true_counts_example, logits_example)
    print(f"Calculated multinomial_nll loss: {loss_val.item()}")

    # Test case with all zero total_count for one batch item (should be handled by Multinomial, but good to be aware)
    # This is tricky because total_count=0 for Multinomial is often problematic.
    # PyTorch Multinomial(0, logits) gives log_prob(zeros)=0, log_prob(non_zeros)=-inf
    # Let's ensure total_count is reasonable for testing.
    
    # A batch item where one channel has all zero true counts for its sequence
    true_counts_channel_zeros = true_counts_example.clone()
    true_counts_channel_zeros[0, :, 0] = 0 # First batch, first channel, all seq positions are 0
    
    # If total_count for a multinomial instance is 0, its log_prob(zeros) is 0.
    # If log_prob(non_zeros) is -inf. This is fine.
    # If true_counts_perm has non-zeros where total_count is 0, it implies an issue.
    # This should not happen if true_counts_perm is the source of total_count.
    
    loss_val_channel_zeros = multinomial_nll_pytorch(true_counts_channel_zeros, logits_example)
    print(f"Calculated multinomial_nll loss (channel zeros): {loss_val_channel_zeros.item()}")
    
    print("Multinomial NLL tests completed.")
    print("-" * 30)


# ###########################################
# CountsMultinomialNLL PyTorch Module/Class
# ###########################################

class CountsMultinomialNLL(nn.Module):
    def __init__(self, c_task_weight=0.0):
        super(CountsMultinomialNLL, self).__init__()
        self.c_task_weight = c_task_weight
        self.epsilon = 1e-7 # For numerical stability

    def forward(self, true_counts, preds):
        # true_counts, preds: (batch, seqlen, channels)

        # Logits calculation
        # Sum over sequence length (dim 1 or -2)
        sum_preds_seqlen = torch.sum(preds, dim=1, keepdim=True)
        # Add epsilon to prevent division by zero if sum_preds_seqlen is 0
        probs = preds / (sum_preds_seqlen + self.epsilon)
        
        # Add epsilon to prevent log(0) or division by zero in (1-probs)
        # Clamp probs to avoid probs > 1 due to numerical issues if preds are noisy
        probs_clamped = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        
        logits = torch.log(probs_clamped / (1.0 - probs_clamped)) # logit = log(p/(1-p))

        multinomial_loss = multinomial_nll_pytorch(true_counts, logits)

        # MSE loss part
        # Sum over sequence length and channels (dim 1 and 2, or -2 and -1)
        sum_true_counts_total = torch.sum(true_counts, dim=(1, 2))
        sum_preds_total = torch.sum(preds, dim=(1, 2))

        log_sum_true = torch.log(1.0 + sum_true_counts_total)
        log_sum_preds = torch.log(1.0 + sum_preds_total)
        
        # MSELoss expects (input, target)
        mse_loss = F.mse_loss(log_sum_preds, log_sum_true, reduction='mean') # Ensure scalar loss

        return multinomial_loss + self.c_task_weight * mse_loss

if __name__ == '__main__':
    # Test multinomial_nll_pytorch
    print("Testing multinomial_nll_pytorch...")
    batch_size, seq_len, num_channels = 2, 10, 3
    true_counts_example = torch.randint(0, 5, (batch_size, seq_len, num_channels)).float()
    true_counts_example[0,0,0] = 1.0 
    true_counts_example[1,0,0] = 1.0
    logits_example = torch.randn(batch_size, seq_len, num_channels)
    loss_val_multi = multinomial_nll_pytorch(true_counts_example, logits_example)
    print(f"Calculated multinomial_nll loss: {loss_val_multi.item()}")
    print("-" * 30)

    # Test CountsMultinomialNLL
    print("Testing CountsMultinomialNLL...")
    c_task_weight_example = 0.1
    counts_loss_fn = CountsMultinomialNLL(c_task_weight=c_task_weight_example)

    # Example predictions (positive values, sum over seqlen should be > 0 for probs)
    preds_example = torch.rand(batch_size, seq_len, num_channels) * 5 + 0.1 # Ensure positive and non-zero sum
    
    total_loss_counts = counts_loss_fn(true_counts_example, preds_example)
    print(f"Calculated CountsMultinomialNLL total loss: {total_loss_counts.item()}")
    
    # Test with zero c_task_weight
    counts_loss_fn_no_mse = CountsMultinomialNLL(c_task_weight=0.0)
    total_loss_counts_no_mse = counts_loss_fn_no_mse(true_counts_example, preds_example)
    # Manually calculate multinomial part for comparison
    _sum_preds_seqlen = torch.sum(preds_example, dim=1, keepdim=True)
    _probs = preds_example / (_sum_preds_seqlen + counts_loss_fn_no_mse.epsilon)
    _probs_clamped = torch.clamp(_probs, counts_loss_fn_no_mse.epsilon, 1.0 - counts_loss_fn_no_mse.epsilon)
    _logits_manual = torch.log(_probs_clamped / (1.0 - _probs_clamped))
    manual_multinomial_loss = multinomial_nll_pytorch(true_counts_example, _logits_manual)
    print(f"Calculated CountsMultinomialNLL (no mse) loss: {total_loss_counts_no_mse.item()}")
    print(f"Manually calculated multinomial part: {manual_multinomial_loss.item()}")
    assert torch.isclose(total_loss_counts_no_mse, manual_multinomial_loss), "Mismatch when c_task_weight is zero"
    print("CountsMultinomialNLL tests completed.")
    print("-" * 30)


# ###########################################
# PoissonMultinomialNLL PyTorch Module/Class
# ###########################################

class PoissonMultinomialNLL(nn.Module):
    def __init__(self, c_task_weight=0.0):
        super(PoissonMultinomialNLL, self).__init__()
        self.c_task_weight = c_task_weight
        self.epsilon = 1e-7 # For numerical stability

    def forward(self, true_counts, preds):
        # true_counts, preds: (batch, seqlen, channels)

        # Logits calculation (same as in CountsMultinomialNLL)
        sum_preds_seqlen = torch.sum(preds, dim=1, keepdim=True)
        probs = preds / (sum_preds_seqlen + self.epsilon)
        probs_clamped = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        logits = torch.log(probs_clamped / (1.0 - probs_clamped))

        multinomial_loss = multinomial_nll_pytorch(true_counts, logits)

        # Poisson loss part
        # Sum over sequence length and channels (dim 1 and 2, or -2 and -1)
        sum_true_counts_total = torch.sum(true_counts, dim=(1, 2))
        sum_preds_total = torch.sum(preds, dim=(1, 2))
        
        # Ensure predictions for Poisson are non-negative
        # sum_preds_total_non_negative = F.relu(sum_preds_total) # Or ensure preds are non-negative earlier
        # Clamping sum_preds_total to be > epsilon for log_input=True if that was used.
        # For log_input=False, non-negativity is good practice, though PoissonNLLLoss might handle small negatives.
        # Let's assume preds (and thus sum_preds_total) are expected to be non-negative.
        # If sum_preds_total can be zero, log_input=True would need careful handling (e.g. log(X + eps)).
        # With log_input=False, sum_preds_total = 0 is fine.

        # PoissonNLLLoss expects (input, target). input is prediction, target is true.
        # reduction='mean' to average over the batch.
        poisson_loss = F.poisson_nll_loss(sum_preds_total, 
                                          sum_true_counts_total, 
                                          log_input=False, 
                                          full=False, # Uses Stirling approx for target factorial if target is large. Keras default is False.
                                          reduction='mean') 

        return multinomial_loss + self.c_task_weight * poisson_loss

if __name__ == '__main__':
    # Test multinomial_nll_pytorch
    print("Testing multinomial_nll_pytorch...")
    batch_size, seq_len, num_channels = 2, 10, 3
    true_counts_example = torch.randint(0, 5, (batch_size, seq_len, num_channels)).float()
    true_counts_example[0,0,0] = 1.0 
    true_counts_example[1,0,0] = 1.0
    logits_example = torch.randn(batch_size, seq_len, num_channels)
    loss_val_multi = multinomial_nll_pytorch(true_counts_example, logits_example)
    print(f"Calculated multinomial_nll loss: {loss_val_multi.item()}")
    print("-" * 30)

    # Test CountsMultinomialNLL
    print("Testing CountsMultinomialNLL...")
    c_task_weight_example_counts = 0.1
    counts_loss_fn = CountsMultinomialNLL(c_task_weight=c_task_weight_example_counts)
    preds_example_counts = torch.rand(batch_size, seq_len, num_channels) * 5 + 0.1
    total_loss_counts = counts_loss_fn(true_counts_example, preds_example_counts)
    print(f"Calculated CountsMultinomialNLL total loss: {total_loss_counts.item()}")
    print("-" * 30)

    # Test PoissonMultinomialNLL
    print("Testing PoissonMultinomialNLL...")
    c_task_weight_example_poisson = 0.05
    poisson_loss_fn = PoissonMultinomialNLL(c_task_weight=c_task_weight_example_poisson)
    
    # preds_example for Poisson should ideally be positive rates
    preds_example_poisson = torch.rand(batch_size, seq_len, num_channels) * 5 + 0.1 

    total_loss_poisson = poisson_loss_fn(true_counts_example, preds_example_poisson)
    print(f"Calculated PoissonMultinomialNLL total loss: {total_loss_poisson.item()}")

    # Test with zero c_task_weight for Poisson
    poisson_loss_fn_no_poisson_term = PoissonMultinomialNLL(c_task_weight=0.0)
    total_loss_poisson_no_poisson_term = poisson_loss_fn_no_poisson_term(true_counts_example, preds_example_poisson)
    # Manually calculate multinomial part for comparison
    _sum_preds_seqlen_p = torch.sum(preds_example_poisson, dim=1, keepdim=True)
    _probs_p = preds_example_poisson / (_sum_preds_seqlen_p + poisson_loss_fn.epsilon)
    _probs_clamped_p = torch.clamp(_probs_p, poisson_loss_fn.epsilon, 1.0 - poisson_loss_fn.epsilon)
    _logits_manual_p = torch.log(_probs_clamped_p / (1.0 - _probs_clamped_p))
    manual_multinomial_loss_p = multinomial_nll_pytorch(true_counts_example, _logits_manual_p)
    print(f"Calculated PoissonMultinomialNLL (no poisson term) loss: {total_loss_poisson_no_poisson_term.item()}")
    print(f"Manually calculated multinomial part (for Poisson test): {manual_multinomial_loss_p.item()}")
    assert torch.isclose(total_loss_poisson_no_poisson_term, manual_multinomial_loss_p), "Mismatch when c_task_weight is zero for Poisson loss"

    print("PoissonMultinomialNLL tests completed.")
    print("-" * 30)


# ###########################################
# ProfileMultinomialNLL PyTorch Module/Class
# ###########################################

class ProfileMultinomialNLL(nn.Module):
    def __init__(self, p_task_weight=0.0, profile_slack=0):
        super(ProfileMultinomialNLL, self).__init__()
        self.p_task_weight = p_task_weight
        self.profile_slack = profile_slack # Corresponds to PROFILE_SLACK in Keras version
        self.epsilon = 1e-7 # For numerical stability

    def forward(self, y_true, y_pred):
        # y_true, y_pred: (batch, seqlen_full, num_outputs_per_task)
        # where num_outputs_per_task is typically 2 (counts/logits at index 0, profile at index 1)
        # This assumes y_true and y_pred are for a single task already, or this loss is mapped over tasks.

        if y_true.shape[-1] != 2 or y_pred.shape[-1] != 2:
            raise ValueError("Last dimension of y_true and y_pred should be 2 (counts/logits and profile). "
                             f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

        # Apply profile slack if any
        if self.profile_slack > 0:
            true_counts_full = y_true[..., self.profile_slack:-self.profile_slack, 0]
            true_profs_full = y_true[..., self.profile_slack:-self.profile_slack, 1]
            logits_full = y_pred[..., self.profile_slack:-self.profile_slack, 0]
            pred_profs_full = y_pred[..., self.profile_slack:-self.profile_slack, 1]
        elif self.profile_slack < 0: # Keras version supports -self.PROFILE_SLACK which means :self.PROFILE_SLACK from end
            true_counts_full = y_true[..., :self.profile_slack, 0]
            true_profs_full = y_true[..., :self.profile_slack, 1]
            logits_full = y_pred[..., :self.profile_slack, 0]
            pred_profs_full = y_pred[..., :self.profile_slack, 1]
        else: # self.profile_slack == 0
            true_counts_full = y_true[..., 0]
            true_profs_full = y_true[..., 1]
            logits_full = y_pred[..., 0]
            pred_profs_full = y_pred[..., 1]
            
        # Add a channel dimension for multinomial_nll which expects (batch, seqlen, channels)
        # Here, for a single task, counts/logits effectively have 1 channel.
        true_counts = true_counts_full.unsqueeze(-1) 
        logits = logits_full.unsqueeze(-1)
        
        # true_profs and pred_profs are (batch, seqlen)
        # No unsqueeze needed for their direct MSE calculation part.

        multinomial_loss = multinomial_nll_pytorch(true_counts, logits)

        # Profile MSE loss part
        # Ensure pred_profs are positive for log(1+x)
        # pred_profs_non_negative = F.relu(pred_profs_full) # Or ensure model outputs non-negative profiles
        # Using original pred_profs_full assuming they are appropriately scaled by model (e.g. softplus)
        
        log_true_profs = torch.log(1.0 + true_profs_full)
        log_pred_profs = torch.log(1.0 + pred_profs_full + self.epsilon) # Add epsilon to pred for stability if it can be ~0

        profile_loss = F.mse_loss(log_pred_profs, log_true_profs, reduction='mean')

        return multinomial_loss + self.p_task_weight * profile_loss

if __name__ == '__main__':
    # Test multinomial_nll_pytorch
    print("Testing multinomial_nll_pytorch...")
    batch_size, seq_len, num_channels = 2, 10, 1 # For single task profile/counts
    true_counts_example_single_ch = torch.randint(0, 5, (batch_size, seq_len, num_channels)).float()
    true_counts_example_single_ch[0,0,0] = 1.0 
    true_counts_example_single_ch[1,0,0] = 1.0
    logits_example_single_ch = torch.randn(batch_size, seq_len, num_channels)
    loss_val_multi_single_ch = multinomial_nll_pytorch(true_counts_example_single_ch, logits_example_single_ch)
    print(f"Calculated multinomial_nll loss (single channel): {loss_val_multi_single_ch.item()}")
    print("-" * 30)
    
    # Test CountsMultinomialNLL (using multi-channel examples from before for variety)
    print("Testing CountsMultinomialNLL...")
    batch_size_mc, seq_len_mc, num_channels_mc = 2, 10, 3
    true_counts_example_mc = torch.randint(0, 5, (batch_size_mc, seq_len_mc, num_channels_mc)).float()
    true_counts_example_mc[0,0,0] = 1.0; true_counts_example_mc[1,0,0] = 1.0
    preds_example_mc = torch.rand(batch_size_mc, seq_len_mc, num_channels_mc) * 5 + 0.1
    c_task_weight_example_counts = 0.1
    counts_loss_fn = CountsMultinomialNLL(c_task_weight=c_task_weight_example_counts)
    total_loss_counts = counts_loss_fn(true_counts_example_mc, preds_example_mc)
    print(f"Calculated CountsMultinomialNLL total loss: {total_loss_counts.item()}")
    print("-" * 30)

    # Test PoissonMultinomialNLL (using multi-channel examples)
    print("Testing PoissonMultinomialNLL...")
    c_task_weight_example_poisson = 0.05
    poisson_loss_fn = PoissonMultinomialNLL(c_task_weight=c_task_weight_example_poisson)
    total_loss_poisson = poisson_loss_fn(true_counts_example_mc, preds_example_mc) # Using same preds as counts for test
    print(f"Calculated PoissonMultinomialNLL total loss: {total_loss_poisson.item()}")
    print("-" * 30)

    # Test ProfileMultinomialNLL
    print("Testing ProfileMultinomialNLL...")
    p_task_weight_example_profile = 0.5
    profile_slack_example = 0 # Assuming no slack for basic test
    # seq_len_full is same as seq_len if profile_slack is 0
    
    # y_true: (batch, seq_len, 2) -> 0: counts, 1: profile
    y_true_example = torch.zeros(batch_size, seq_len, 2)
    y_true_example[..., 0] = true_counts_example_single_ch.squeeze(-1) # Counts part
    y_true_example[..., 1] = torch.rand(batch_size, seq_len) * 10 # True profiles
    
    # y_pred: (batch, seq_len, 2) -> 0: logits, 1: predicted profile
    y_pred_example = torch.zeros(batch_size, seq_len, 2)
    y_pred_example[..., 0] = logits_example_single_ch.squeeze(-1) # Logits part
    y_pred_example[..., 1] = torch.rand(batch_size, seq_len) * 10 + 0.1 # Predicted profiles
                                                           
    profile_loss_fn = ProfileMultinomialNLL(p_task_weight=p_task_weight_example_profile, 
                                            profile_slack=profile_slack_example)
    total_loss_profile = profile_loss_fn(y_true_example, y_pred_example)
    print(f"Calculated ProfileMultinomialNLL total loss: {total_loss_profile.item()}")

    # Test with profile_slack
    profile_slack_val = 2
    seq_len_full_slack = seq_len + 2 * profile_slack_val # e.g. 10 + 4 = 14
    y_true_slack = torch.rand(batch_size, seq_len_full_slack, 2)
    y_pred_slack = torch.rand(batch_size, seq_len_full_slack, 2)
    # Ensure counts part of y_true_slack (after crop) sums to non-zero for multinomial
    y_true_slack_counts_target = y_true_slack[:, profile_slack_val:-profile_slack_val, 0]
    y_true_slack_counts_target[0,0] = y_true_slack_counts_target[0,0] + 1.0 # ensure some count
    
    profile_loss_fn_slack = ProfileMultinomialNLL(p_task_weight=p_task_weight_example_profile,
                                                  profile_slack=profile_slack_val)
    total_loss_profile_slack = profile_loss_fn_slack(y_true_slack, y_pred_slack)
    print(f"Calculated ProfileMultinomialNLL total loss (with slack={profile_slack_val}): {total_loss_profile_slack.item()}")

    print("ProfileMultinomialNLL tests completed.")
    print("-" * 30)
