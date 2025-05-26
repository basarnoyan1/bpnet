"""
PyTorch Loss functions for BPNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Poisson
import gin
from typing import Optional, Dict, Any
import numpy as np


def get_from_module(name, module_dict):
    """Helper function to get class from module dictionary"""
    if name in module_dict:
        return module_dict[name]
    else:
        raise KeyError(f"Module {name} not found in available modules")


@gin.configurable
def multinomial_nll(true_counts: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute the multinomial negative log-likelihood along the sequence (axis=1)
    and sum the values across all channels

    Args:
      true_counts: observed count values (batch, seqlen, channels)
      logits: predicted logit values (batch, seqlen, channels)
      
    Returns:
      Scalar loss value
    """
    # Swap axes such that the final axis will be the positional axis
    # PyTorch expects (batch, channels, seqlen) for most operations
    logits_perm = logits.transpose(1, 2)  # (batch, channels, seqlen)
    true_counts_perm = true_counts.transpose(1, 2)  # (batch, channels, seqlen)
    
    # Sum counts per example across sequence positions
    counts_per_example = torch.sum(true_counts_perm, dim=-1)  # (batch, channels)
    
    # Handle edge case where total counts might be zero
    counts_per_example = torch.clamp(counts_per_example, min=1e-8)
    
    batch_size, n_channels, seq_len = logits_perm.shape
    total_loss = 0.0
    
    # Compute multinomial loss for each channel separately
    for channel in range(n_channels):
        channel_logits = logits_perm[:, channel, :]  # (batch, seqlen)
        channel_counts = true_counts_perm[:, channel, :]  # (batch, seqlen)
        channel_total = counts_per_example[:, channel]  # (batch,)
        
        # Create multinomial distribution
        try:
            dist = Multinomial(total_count=channel_total, logits=channel_logits)
            # Compute log probability
            log_prob = dist.log_prob(channel_counts)
            total_loss += -torch.sum(log_prob)
        except Exception as e:
            # Fallback to manual computation if distribution fails
            probs = F.softmax(channel_logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
            log_probs = torch.log(probs)
            total_loss += -torch.sum(channel_counts * log_probs)
    
    # Normalize by batch size
    batch_size = float(true_counts.shape[0])
    return total_loss / batch_size


@gin.configurable
class CountsMultinomialNLL(nn.Module):
    """Combined multinomial NLL and MSE loss for count predictions"""

    def __init__(self, c_task_weight: float = 1.0):
        super().__init__()
        self.c_task_weight = c_task_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, preds: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: predicted count values (batch, seqlen, channels)
            true_counts: observed count values (batch, seqlen, channels)
        """
        # Ensure non-negative predictions
        preds = torch.clamp(preds, min=1e-8)
        
        # Compute probabilities by normalizing across sequence length
        preds_sum = torch.sum(preds, dim=1, keepdim=True)  # (batch, 1, channels)
        preds_sum = torch.clamp(preds_sum, min=1e-8)
        probs = preds / preds_sum
        
        # Convert probabilities to logits
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        logits = torch.log(probs / (1 - probs))
        
        # Multinomial loss
        multinomial_loss = multinomial_nll(true_counts, logits)
        
        # MSE loss on log-transformed total counts
        true_total = torch.sum(true_counts, dim=(1, 2))  # Sum over seqlen and channels
        pred_total = torch.sum(preds, dim=(1, 2))
        
        log_true_total = torch.log(1 + true_total)
        log_pred_total = torch.log(1 + pred_total)
        
        mse_loss = self.mse_loss(log_pred_total, log_true_total)
        
        return multinomial_loss + self.c_task_weight * mse_loss

    def get_config(self) -> Dict[str, Any]:
        return {"c_task_weight": self.c_task_weight}


@gin.configurable
class PoissonMultinomialNLL(nn.Module):
    """Combined multinomial NLL and Poisson loss for count predictions"""

    def __init__(self, c_task_weight: float = 1.0):
        super().__init__()
        self.c_task_weight = c_task_weight

    def forward(self, preds: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: predicted count values (batch, seqlen, channels)
            true_counts: observed count values (batch, seqlen, channels)
        """
        # Ensure non-negative predictions
        preds = torch.clamp(preds, min=1e-8)
        
        # Compute probabilities by normalizing across sequence length
        preds_sum = torch.sum(preds, dim=1, keepdim=True)  # (batch, 1, channels)
        preds_sum = torch.clamp(preds_sum, min=1e-8)
        probs = preds / preds_sum
        
        # Convert probabilities to logits
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        logits = torch.log(probs / (1 - probs))
        
        # Multinomial loss
        multinomial_loss = multinomial_nll(true_counts, logits)
        
        # Poisson loss on total counts
        true_total = torch.sum(true_counts, dim=(1, 2))  # Sum over seqlen and channels
        pred_total = torch.sum(preds, dim=(1, 2))
        
        # Poisson NLL loss
        poisson_loss = self._poisson_nll(true_total, pred_total)
        
        return multinomial_loss + self.c_task_weight * poisson_loss

    def _poisson_nll(self, true_counts: torch.Tensor, pred_rates: torch.Tensor) -> torch.Tensor:
        """Compute Poisson negative log-likelihood"""
        # Clamp predictions to avoid numerical issues
        pred_rates = torch.clamp(pred_rates, min=1e-8)
        
        # Poisson NLL: -log(P(k|λ)) = λ - k*log(λ) + log(k!)
        # We ignore the log(k!) term as it's constant w.r.t. parameters
        nll = pred_rates - true_counts * torch.log(pred_rates)
        return torch.mean(nll)

    def get_config(self) -> Dict[str, Any]:
        return {"c_task_weight": self.c_task_weight}


@gin.configurable
class BPNetLoss(nn.Module):
    """Complete BPNet loss combining profile and count losses"""
    
    def __init__(self, 
                 profile_loss_weight: float = 1.0,
                 count_loss_weight: float = 1.0,
                 profile_loss_type: str = "multinomial",
                 count_loss_type: str = "mse"):
        super().__init__()
        self.profile_loss_weight = profile_loss_weight
        self.count_loss_weight = count_loss_weight
        self.profile_loss_type = profile_loss_type
        self.count_loss_type = count_loss_type
        
        # Initialize profile loss
        if profile_loss_type == "multinomial":
            self.profile_loss_fn = CountsMultinomialNLL(c_task_weight=0)  # No count component
        elif profile_loss_type == "poisson_multinomial":
            self.profile_loss_fn = PoissonMultinomialNLL(c_task_weight=0)
        else:
            raise ValueError(f"Unknown profile loss type: {profile_loss_type}")
        
        # Initialize count loss
        if count_loss_type == "mse":
            self.count_loss_fn = nn.MSELoss()
        elif count_loss_type == "poisson":
            self.count_loss_fn = lambda pred, true: self._poisson_nll(true, pred)
        else:
            raise ValueError(f"Unknown count loss type: {count_loss_type}")

    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions: Dictionary containing 'profile' and 'counts' predictions
            targets: Dictionary containing 'profile' and 'counts' targets
        """
        total_loss = 0.0
        
        # Profile loss
        if 'profile' in predictions and 'profile' in targets:
            profile_loss = self.profile_loss_fn(predictions['profile'], targets['profile'])
            total_loss += self.profile_loss_weight * profile_loss
        
        # Count loss
        if 'counts' in predictions and 'counts' in targets:
            if self.count_loss_type == "mse":
                # Apply log transform for MSE
                pred_counts = torch.log(1 + predictions['counts'])
                true_counts = torch.log(1 + targets['counts'])
                count_loss = self.count_loss_fn(pred_counts, true_counts)
            else:
                count_loss = self.count_loss_fn(predictions['counts'], targets['counts'])
            total_loss += self.count_loss_weight * count_loss
        
        return total_loss

    def _poisson_nll(self, true_counts: torch.Tensor, pred_rates: torch.Tensor) -> torch.Tensor:
        """Compute Poisson negative log-likelihood"""
        pred_rates = torch.clamp(pred_rates, min=1e-8)
        nll = pred_rates - true_counts * torch.log(pred_rates)
        return torch.mean(nll)


# Standalone loss functions for backward compatibility
@gin.configurable
def mse_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss"""
    return F.mse_loss(pred, true)


@gin.configurable 
def poisson_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Poisson loss"""
    pred = torch.clamp(pred, min=1e-8)
    return torch.mean(pred - true * torch.log(pred))


@gin.configurable
def mae_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean absolute error loss"""
    return F.l1_loss(pred, true)


# Available loss functions
AVAILABLE = [
    "multinomial_nll",
    "CountsMultinomialNLL", 
    "PoissonMultinomialNLL",
    "BPNetLoss",
    "mse_loss",
    "poisson_loss", 
    "mae_loss"
]


def get(name: str):
    """Get loss function by name"""
    # First try PyTorch built-in losses
    pytorch_losses = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'l1': nn.L1Loss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),
        'bce_with_logits': nn.BCEWithLogitsLoss(),
        'nll': nn.NLLLoss(),
        'poisson_nll': nn.PoissonNLLLoss(),
        'kl_div': nn.KLDivLoss(),
        'huber': nn.HuberLoss(),
        'smooth_l1': nn.SmoothL1Loss(),
    }
    
    if name.lower() in pytorch_losses:
        return pytorch_losses[name.lower()]
    
    # Then try our custom losses
    try:
        return get_from_module(name, globals())
    except KeyError:
        raise ValueError(f"Unknown loss function: {name}. Available: {AVAILABLE + list(pytorch_losses.keys())}")