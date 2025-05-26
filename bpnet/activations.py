import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from typing import Optional, Union


def get_from_module(name, module_dict):
    """Helper function to get class from module dictionary"""
    if name in module_dict:
        return module_dict[name]
    else:
        raise KeyError(f"Module {name} not found in available modules")


@gin.configurable
def clipped_exp(x: torch.Tensor, min_value: float = -50, max_value: float = 50) -> torch.Tensor:
    """
    Exponential function with clipping to prevent overflow/underflow
    
    Args:
        x: Input tensor
        min_value: Minimum value to clip input before exp
        max_value: Maximum value to clip input before exp
    
    Returns:
        torch.Tensor: exp(clip(x, min_value, max_value))
    """
    clipped = torch.clamp(x, min=min_value, max=max_value)
    return torch.exp(clipped)


class ClippedExp(nn.Module):
    """
    Clipped exponential activation as a PyTorch module
    """
    
    def __init__(self, min_value: float = -50, max_value: float = 50):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return clipped_exp(x, self.min_value, self.max_value)
    
    def extra_repr(self) -> str:
        return f'min_value={self.min_value}, max_value={self.max_value}'


@gin.configurable
def softmax_2(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax along the second-last axis (axis=-2)
    
    Args:
        x: Input tensor
    
    Returns:
        torch.Tensor: Softmax applied along axis=-2
    """
    return F.softmax(x, dim=-2)


class Softmax2(nn.Module):
    """
    Softmax along the second-last axis as a PyTorch module
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax_2(x)


@gin.configurable
def stable_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable softmax
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        eps: Small epsilon for numerical stability
    
    Returns:
        torch.Tensor: Stable softmax output
    """
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / (sum_exp + eps)


class StableSoftmax(nn.Module):
    """
    Numerically stable softmax as a PyTorch module
    """
    
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stable_softmax(x, self.dim, self.eps)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


@gin.configurable
def gated_activation(x: torch.Tensor, gate_fn: str = 'sigmoid', activation_fn: str = 'tanh') -> torch.Tensor:
    """
    Gated activation function (like in GLU - Gated Linear Unit)
    
    Args:
        x: Input tensor (last dimension should be even)
        gate_fn: Gate function ('sigmoid', 'softmax', etc.)
        activation_fn: Activation function ('tanh', 'relu', etc.)
    
    Returns:
        torch.Tensor: Gated activation output
    """
    # Split input in half along last dimension
    split_size = x.shape[-1] // 2
    input_part, gate_part = torch.split(x, split_size, dim=-1)
    
    # Apply gate function
    if gate_fn == 'sigmoid':
        gate = torch.sigmoid(gate_part)
    elif gate_fn == 'softmax':
        gate = F.softmax(gate_part, dim=-1)
    elif gate_fn == 'tanh':
        gate = torch.tanh(gate_part)
    else:
        raise ValueError(f"Unknown gate function: {gate_fn}")
    
    # Apply activation function
    if activation_fn == 'tanh':
        activation = torch.tanh(input_part)
    elif activation_fn == 'relu':
        activation = F.relu(input_part)
    elif activation_fn == 'linear':
        activation = input_part
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")
    
    return activation * gate


class GatedActivation(nn.Module):
    """
    Gated activation as a PyTorch module
    """
    
    def __init__(self, gate_fn: str = 'sigmoid', activation_fn: str = 'tanh'):
        super().__init__()
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gated_activation(x, self.gate_fn, self.activation_fn)
    
    def extra_repr(self) -> str:
        return f'gate_fn={self.gate_fn}, activation_fn={self.activation_fn}'


@gin.configurable
def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish activation function: x * sigmoid(beta * x)
    
    Args:
        x: Input tensor
        beta: Scaling factor
    
    Returns:
        torch.Tensor: Swish activation output
    """
    return x * torch.sigmoid(beta * x)


class Swish(nn.Module):
    """
    Swish activation as a PyTorch module
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, self.beta)
    
    def extra_repr(self) -> str:
        return f'beta={self.beta}'


@gin.configurable
def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Mish activation function: x * tanh(softplus(x))
    
    Args:
        x: Input tensor
    
    Returns:
        torch.Tensor: Mish activation output
    """
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    """
    Mish activation as a PyTorch module
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


@gin.configurable
def gelu_precise(x: torch.Tensor) -> torch.Tensor:
    """
    Precise GELU activation function
    
    Args:
        x: Input tensor
    
    Returns:
        torch.Tensor: GELU activation output
    """
    return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class GeluPrecise(nn.Module):
    """
    Precise GELU activation as a PyTorch module
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_precise(x)


@gin.configurable
def log_softmax_2(x: torch.Tensor) -> torch.Tensor:
    """
    Log softmax along the second-last axis (axis=-2)
    
    Args:
        x: Input tensor
    
    Returns:
        torch.Tensor: Log softmax applied along axis=-2
    """
    return F.log_softmax(x, dim=-2)


class LogSoftmax2(nn.Module):
    """
    Log softmax along the second-last axis as a PyTorch module
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax_2(x)


# Activation factory function
def get_activation(name: str, **kwargs) -> Union[nn.Module, callable]:
    """
    Get activation function or module by name
    
    Args:
        name: Activation name
        **kwargs: Additional arguments for activation
    
    Returns:
        Activation function or module
    """
    activations = {
        # PyTorch built-ins
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=-1),
        'log_softmax': nn.LogSoftmax(dim=-1),
        'softplus': nn.Softplus(),
        'softsign': nn.Softsign(),
        'hardtanh': nn.Hardtanh(),
        'hardsigmoid': nn.Hardsigmoid(),
        'hardswish': nn.Hardswish(),
        
        # Custom activations
        'clipped_exp': ClippedExp(**kwargs),
        'softmax_2': Softmax2(),
        'log_softmax_2': LogSoftmax2(),
        'stable_softmax': StableSoftmax(**kwargs),
        'gated_activation': GatedActivation(**kwargs),
        'swish': Swish(**kwargs),
        'mish': Mish(),
        'gelu_precise': GeluPrecise(),
        
        # Function versions
        'clipped_exp_fn': lambda x: clipped_exp(x, **kwargs),
        'softmax_2_fn': softmax_2,
        'stable_softmax_fn': lambda x: stable_softmax(x, **kwargs),
        'swish_fn': lambda x: swish(x, **kwargs),
        'mish_fn': mish,
        'gelu_precise_fn': gelu_precise,
    }
    
    if name in activations:
        return activations[name]
    else:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")


# Available activations for backwards compatibility
AVAILABLE = [
    "clipped_exp", 
    "softmax_2", 
    "log_softmax_2",
    "stable_softmax",
    "gated_activation",
    "swish", 
    "mish",
    "gelu_precise",
    "ClippedExp",
    "Softmax2", 
    "LogSoftmax2",
    "StableSoftmax",
    "GatedActivation",
    "Swish",
    "Mish", 
    "GeluPrecise",
    "get_activation"
]


def get(name: str):
    """Get activation by name"""
    return get_from_module(name, globals())


# Functional API for common use cases
def apply_activation(x: torch.Tensor, activation: Union[str, nn.Module, callable], **kwargs) -> torch.Tensor:
    """
    Apply activation function to tensor
    
    Args:
        x: Input tensor
        activation: Activation name, module, or function
        **kwargs: Additional arguments
    
    Returns:
        torch.Tensor: Output after applying activation
    """
    if isinstance(activation, str):
        activation = get_activation(activation, **kwargs)
    
    if isinstance(activation, nn.Module):
        return activation(x)
    elif callable(activation):
        return activation(x)
    else:
        raise ValueError(f"Invalid activation type: {type(activation)}")