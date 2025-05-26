import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gin
from typing import List, Optional, Union
import math


def get_from_module(name, module_dict):
    """Helper function to get class from module dictionary"""
    if name in module_dict:
        return module_dict[name]
    else:
        raise KeyError(f"Module {name} not found in available modules")


class SplineWeight1D(nn.Module):
    """PyTorch implementation of spline-based positional weighting"""
    
    def __init__(self, n_bases=10, share_splines=True):
        super().__init__()
        self.n_bases = n_bases
        self.share_splines = share_splines
        
        # Initialize spline weights
        self.spline_weights = nn.Parameter(torch.randn(n_bases))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, n_features = x.shape
        
        # Create spline basis functions
        positions = torch.linspace(0, 1, seq_len, device=x.device)
        basis_positions = torch.linspace(0, 1, self.n_bases, device=x.device)
        
        # Simple linear interpolation-based splines
        weights = torch.zeros(seq_len, device=x.device)
        for i in range(seq_len):
            pos = positions[i]
            # Find nearest basis functions
            dists = torch.abs(basis_positions - pos)
            weights[i] = torch.sum(self.spline_weights * torch.exp(-dists * 10))
        
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(weights, dim=0)
        
        # Apply weights
        if self.share_splines:
            # Same weights for all features
            weighted_x = x * weights.unsqueeze(0).unsqueeze(-1)
        else:
            # Different weights per feature (would need more parameters)
            weighted_x = x * weights.unsqueeze(0).unsqueeze(-1)
            
        return weighted_x


@gin.configurable
class GlobalAvgPoolFCN(nn.Module):
    """Global Average Pooling followed by Fully Connected Network"""

    def __init__(self,
                 n_tasks: int = 1,
                 dropout: float = 0.0,
                 hidden: Optional[List[int]] = None,
                 dropout_hidden: float = 0.0,
                 n_splines: int = 0,
                 batchnorm: bool = False):
        super().__init__()
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.n_splines = n_splines
        self.hidden = hidden if hidden is not None else []
        
        assert self.n_splines >= 0
        
        # Spline layer if needed
        if self.n_splines > 0:
            self.spline_layer = SplineWeight1D(n_bases=self.n_splines, share_splines=True)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for h in self.hidden:
            layer_list = []
            if self.batchnorm:
                layer_list.append(nn.BatchNorm1d(h))
            layer_list.append(nn.Linear(h, h))
            layer_list.append(nn.ReLU())
            if self.dropout_hidden > 0:
                layer_list.append(nn.Dropout(self.dropout_hidden))
            self.hidden_layers.append(nn.Sequential(*layer_list))
        
        # Final output layer
        self.output_layers = nn.ModuleList()
        if self.batchnorm:
            self.output_layers.append(nn.BatchNorm1d(self.n_tasks))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) or (batch, features, seq_len)
        if len(x.shape) == 3 and x.shape[1] > x.shape[2]:
            # Assume (batch, seq_len, features) format, convert to (batch, features, seq_len)
            x = x.transpose(1, 2)
        
        if self.n_splines > 0:
            # Convert back to (batch, seq_len, features) for spline layer
            x = x.transpose(1, 2)
            x = self.spline_layer(x)
            x = x.transpose(1, 2)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, features, 1)
        x = x.squeeze(-1)  # (batch, features)
        
        if hasattr(self, 'dropout_layer'):
            x = self.dropout_layer(x)
        
        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        # Final dense layer
        if self.batchnorm and len(self.output_layers) > 0:
            x = self.output_layers[0](x)
        
        # Output projection - need to determine input size dynamically
        if not hasattr(self, 'final_layer'):
            self.final_layer = nn.Linear(x.shape[-1], self.n_tasks).to(x.device)
        
        x = self.final_layer(x)
        return x


@gin.configurable
class FCN(nn.Module):
    """Fully Connected Network"""

    def __init__(self,
                 n_tasks: int = 1,
                 hidden: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 dropout_hidden: float = 0.0,
                 batchnorm: bool = False):
        super().__init__()
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.hidden = hidden if hidden is not None else []
        
        # Input dropout
        if self.dropout > 0:
            self.input_dropout = nn.Dropout(self.dropout)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for h in self.hidden:
            layer_list = []
            if self.batchnorm:
                layer_list.append(nn.BatchNorm1d(h))
            layer_list.append(nn.Linear(h, h))
            layer_list.append(nn.ReLU())
            if self.dropout_hidden > 0:
                layer_list.append(nn.Dropout(self.dropout_hidden))
            self.hidden_layers.append(nn.Sequential(*layer_list))
        
        # Output batch norm
        if self.batchnorm:
            self.output_bn = nn.BatchNorm1d(self.n_tasks)

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        if hasattr(self, 'input_dropout'):
            x = self.input_dropout(x)
        
        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        # Final output layer
        if hasattr(self, 'output_bn'):
            x = self.output_bn(x)
        
        # Output projection - need to determine input size dynamically
        if not hasattr(self, 'final_layer'):
            self.final_layer = nn.Linear(x.shape[-1], self.n_tasks).to(x.device)
        
        x = self.final_layer(x)
        return x


@gin.configurable
class DilatedConv1D(nn.Module):
    """Dilated Convolutional layers with skip connections"""

    def __init__(self, 
                 filters: int = 21,
                 conv1_kernel_size: int = 25,
                 n_dil_layers: int = 6,
                 skip_type: str = 'residual',  # or 'dense', None
                 padding: str = 'same',
                 batchnorm: bool = False,
                 add_pointwise: bool = False):
        super().__init__()
        self.filters = filters
        self.conv1_kernel_size = conv1_kernel_size
        self.n_dil_layers = n_dil_layers
        self.skip_type = skip_type
        self.padding = padding
        self.batchnorm = batchnorm
        self.add_pointwise = add_pointwise
        
        # First convolution
        self.first_conv = nn.Conv1d(4, filters, kernel_size=conv1_kernel_size, 
                                   padding=conv1_kernel_size//2)
        
        # Optional pointwise convolution
        if self.add_pointwise:
            if self.batchnorm:
                self.first_bn = nn.BatchNorm1d(filters)
            self.pointwise_conv = nn.Conv1d(filters, filters, kernel_size=1)
        
        # Dilated convolutions
        self.dil_convs = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(1, n_dil_layers + 1):
            dilation = 2 ** i
            conv = nn.Conv1d(filters, filters, kernel_size=3, 
                           padding=dilation, dilation=dilation)
            self.dil_convs.append(conv)
            
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(filters))
        
        # Final convolution for dense connections
        if self.skip_type == 'dense':
            total_filters = filters * (n_dil_layers + 1)
            self.final_conv = nn.Conv1d(total_filters, filters, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, seq_len, 4) -> (batch, 4, seq_len)
        if x.shape[-1] == 4:
            x = x.transpose(1, 2)
        
        # First convolution
        x = F.relu(self.first_conv(x))
        
        if self.add_pointwise:
            if self.batchnorm:
                x = self.first_bn(x)
            x = F.relu(self.pointwise_conv(x))
        
        prev_layer = x
        dense_layers = [prev_layer] if self.skip_type == 'dense' else None
        
        # Dilated convolutions with skip connections
        for i, conv in enumerate(self.dil_convs):
            if self.batchnorm and i < len(self.bn_layers):
                layer_input = self.bn_layers[i](prev_layer)
            else:
                layer_input = prev_layer
            
            conv_output = F.relu(conv(layer_input))
            
            # Skip connections
            if self.skip_type is None:
                prev_layer = conv_output
            elif self.skip_type == 'residual':
                prev_layer = prev_layer + conv_output
            elif self.skip_type == 'dense':
                dense_layers.append(conv_output)
                prev_layer = torch.cat(dense_layers, dim=1)
            else:
                raise ValueError("skip_type needs to be 'residual' or 'dense' or None")
        
        combined_conv = prev_layer
        
        # Final convolution for dense connections
        if self.skip_type == 'dense':
            combined_conv = F.relu(self.final_conv(combined_conv))
        
        # Handle valid padding by cropping
        if self.padding == 'valid':
            crop_size = abs(self.get_len_change()) // 2
            if crop_size > 0:
                combined_conv = combined_conv[:, :, crop_size:-crop_size]
        
        return combined_conv

    def get_len_change(self):
        """Calculate length change for valid padding"""
        if self.padding == 'same':
            return 0
        else:
            d = 0
            # First conv
            d -= 2 * (self.conv1_kernel_size // 2)
            # Dilated convs
            for i in range(1, self.n_dil_layers + 1):
                dilation = 2 ** i
                d -= 2 * dilation
            return d


@gin.configurable
class DeConv1D(nn.Module):
    """Deconvolutional layer for upsampling"""

    def __init__(self, 
                 filters: int,
                 n_tasks: int,
                 tconv_kernel_size: int = 25,
                 padding: str = 'same',
                 n_hidden: int = 0,
                 batchnorm: bool = False):
        super().__init__()
        self.filters = filters
        self.n_tasks = n_tasks
        self.tconv_kernel_size = tconv_kernel_size
        self.n_hidden = n_hidden
        self.batchnorm = batchnorm
        self.padding = padding
        
        # Hidden conv layers
        self.hidden_convs = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        
        for i in range(n_hidden):
            self.hidden_convs.append(nn.Conv1d(filters, filters, kernel_size=1))
            if self.batchnorm:
                self.hidden_bns.append(nn.BatchNorm1d(filters))
        
        # Transpose convolution
        if self.batchnorm:
            self.tconv_bn = nn.BatchNorm1d(filters)
        
        self.tconv = nn.ConvTranspose1d(filters, n_tasks, 
                                       kernel_size=tconv_kernel_size,
                                       padding=tconv_kernel_size//2)

    def forward(self, x):
        # x shape: (batch, features, seq_len)
        
        # Hidden conv layers
        for i in range(self.n_hidden):
            if self.batchnorm and i < len(self.hidden_bns):
                x = self.hidden_bns[i](x)
            x = F.relu(self.hidden_convs[i](x))
        
        # Transpose convolution
        if self.batchnorm and hasattr(self, 'tconv_bn'):
            x = self.tconv_bn(x)
        
        x = self.tconv(x)
        
        # Handle valid padding
        if self.padding == 'valid':
            crop_size = abs(self.get_len_change()) // 2
            if crop_size > 0:
                x = x[:, :, crop_size:-crop_size]
        
        return x

    def get_len_change(self):
        """Calculate length change for valid padding"""
        if self.padding == 'same':
            return 0
        else:
            return -2 * (self.tconv_kernel_size // 2)


@gin.configurable
class MovingAverages(nn.Module):
    """Layer to compute moving averages at multiple resolutions"""

    def __init__(self, window_sizes: List[int]):
        super().__init__()
        self.window_sizes = window_sizes
        
        # Create fixed convolution kernels for moving averages
        self.avg_convs = nn.ModuleList()
        for window_size in window_sizes:
            if window_size > 1:
                conv = nn.Conv1d(1, 1, kernel_size=window_size, 
                               padding=window_size//2, bias=False)
                # Initialize with ones for averaging
                with torch.no_grad():
                    conv.weight.fill_(1.0 / window_size)
                conv.weight.requires_grad = False  # Fixed weights
                self.avg_convs.append(conv)
            else:
                self.avg_convs.append(nn.Identity())
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv1d(len(window_sizes), 1, kernel_size=1, bias=False)

    def forward(self, x):
        # x shape: (batch, features, seq_len)
        batch_size, n_features, seq_len = x.shape
        
        outputs = []
        for i, (window_size, conv) in enumerate(zip(self.window_sizes, self.avg_convs)):
            if window_size == 1:
                # No averaging needed
                outputs.append(x)
            else:
                # Apply moving average to each feature separately
                feature_outputs = []
                for feat_idx in range(n_features):
                    feat_input = x[:, feat_idx:feat_idx+1, :]  # (batch, 1, seq_len)
                    feat_output = conv(feat_input)
                    feature_outputs.append(feat_output)
                
                # Concatenate features back
                avg_output = torch.cat(feature_outputs, dim=1)
                outputs.append(avg_output)
        
        # Concatenate all window sizes
        # Need to sum across features for each window size
        summed_outputs = []
        for output in outputs:
            summed_output = torch.sum(output, dim=1, keepdim=True)  # (batch, 1, seq_len)
            summed_outputs.append(summed_output)
        
        # Concatenate different window sizes
        concatenated = torch.cat(summed_outputs, dim=1)  # (batch, len(window_sizes), seq_len)
        
        # Final convolution
        result = self.final_conv(concatenated)  # (batch, 1, seq_len)
        
        return result


# Available modules for dynamic loading
AVAILABLE = [
    'GlobalAvgPoolFCN',
    'FCN', 
    'DilatedConv1D',
    'DeConv1D',
    'MovingAverages',
    'SplineWeight1D'
]


def get(name):
    """Get module by name"""
    return get_from_module(name, globals())