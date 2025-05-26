import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from typing import List, Optional, Dict, Union, Callable, Any
import gin
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def get_from_module(name, module_dict):
    """Helper function to get class from module dictionary"""
    if name in module_dict:
        return module_dict[name]
    else:
        raise KeyError(f"Module {name} not found in available modules")


def clipped_exp(x: torch.Tensor, clip_value: float = 6.0) -> torch.Tensor:
    """Exponential activation with clipping to prevent overflow"""
    return torch.exp(torch.clamp(x, max=clip_value))


class ProfileHead(nn.Module):
    """Profile prediction head for BPNet"""
    
    def __init__(self, 
                 net: nn.Module,
                 target_name: str,
                 loss_fn: Optional[Callable] = None,
                 loss_weight: float = 1.0,
                 activation: Optional[Union[str, Callable]] = None,
                 postproc_fn: Optional[Callable] = None,
                 use_bias: bool = False,
                 bias_net: Optional[nn.Module] = None,
                 bias_shape: Optional[tuple] = None,
                 metric: Optional[Callable] = None):
        super().__init__()
        
        self.net = net
        self.target_name = target_name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.activation = activation
        self.postproc_fn = postproc_fn
        self.use_bias = use_bias
        self.bias_net = bias_net
        self.bias_shape = bias_shape
        self.metric = metric
        
        # Setup activation function
        if isinstance(activation, str):
            if activation == 'sigmoid':
                self.activation_fn = torch.sigmoid
            elif activation == 'relu':
                self.activation_fn = F.relu
            elif activation == 'softmax':
                self.activation_fn = lambda x: F.softmax(x, dim=-1)
            elif activation == 'clipped_exp':
                self.activation_fn = clipped_exp
            else:
                self.activation_fn = None
        else:
            self.activation_fn = activation
    
    def forward(self, x: torch.Tensor, bias_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Main network forward pass
        out = self.net(x)
        
        # Add bias if specified
        if self.use_bias and bias_input is not None and self.bias_net is not None:
            bias_out = self.bias_net(bias_input)
            # Broadcast bias to match output shape
            if bias_out.shape != out.shape:
                bias_out = bias_out.expand_as(out)
            out = out + bias_out
        
        # Apply activation
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        
        # Apply post-processing
        if self.postproc_fn is not None:
            out = self.postproc_fn(out)
        
        return out


class ScalarHead(nn.Module):
    """Scalar prediction head for BPNet"""
    
    def __init__(self,
                 net: nn.Module,
                 target_name: str,
                 loss_fn: Optional[Union[str, Callable]] = None,
                 loss_weight: float = 1.0,
                 activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = False,
                 bias_shape: Optional[tuple] = None,
                 metric: Optional[Callable] = None):
        super().__init__()
        
        self.net = net
        self.target_name = target_name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.use_bias = use_bias
        self.bias_shape = bias_shape
        self.metric = metric
        
        # Setup activation function
        if isinstance(activation, str):
            if activation == 'sigmoid':
                self.activation_fn = torch.sigmoid
            elif activation == 'relu':
                self.activation_fn = F.relu
            elif activation == 'tanh':
                self.activation_fn = torch.tanh
            else:
                self.activation_fn = None
        else:
            self.activation_fn = activation
        
        # Setup loss function
        if isinstance(loss_fn, str):
            if loss_fn == 'mse':
                self.loss_fn = nn.MSELoss()
            elif loss_fn == 'binary_crossentropy':
                self.loss_fn = nn.BCELoss()
            elif loss_fn == 'cross_entropy':
                self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, bias_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Main network forward pass
        out = self.net(x)
        
        # Add bias if specified
        if self.use_bias and bias_input is not None:
            if bias_input.shape[-1] == out.shape[-1]:
                out = out + bias_input
        
        # Apply activation
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        
        return out


class SeqModel(nn.Module):
    """Main sequence model combining body and heads"""
    
    def __init__(self,
                 body: nn.Module,
                 heads: List[Union[ProfileHead, ScalarHead]],
                 tasks: List[str],
                 optimizer_class: type = Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 seqlen: Optional[int] = None):
        super().__init__()
        
        self.body = body
        self.heads = nn.ModuleList(heads)
        self.tasks = tasks
        self.seqlen = seqlen
        
        # Setup optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.004}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None
    
    def create_optimizer(self):
        """Create optimizer for the model"""
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        return self.optimizer
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract sequence input
        seq_input = inputs['seq']
        
        # Forward through body
        body_output = self.body(seq_input)
        
        # Forward through heads
        outputs = {}
        for head in self.heads:
            # Get bias input if needed
            bias_input = None
            if head.use_bias:
                # Try to find bias input for this head
                for task in self.tasks:
                    bias_key = head.target_name.format(task=task).replace('profile', 'bias/profile').replace('counts', 'bias/counts')
                    if bias_key in inputs:
                        bias_input = inputs[bias_key]
                        break
            
            # Forward through head for each task
            for task in self.tasks:
                target_key = head.target_name.format(task=task)
                head_output = head(body_output, bias_input)
                outputs[target_key] = head_output
        
        return outputs
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total loss from all heads"""
        total_loss = 0.0
        
        for head in self.heads:
            if head.loss_fn is not None:
                for task in self.tasks:
                    target_key = head.target_name.format(task=task)
                    if target_key in predictions and target_key in targets:
                        if hasattr(head.loss_fn, '__call__'):
                            loss = head.loss_fn(predictions[target_key], targets[target_key])
                        else:
                            # Handle string loss functions
                            if head.loss_fn == 'mse':
                                loss = F.mse_loss(predictions[target_key], targets[target_key])
                            elif head.loss_fn == 'binary_crossentropy':
                                loss = F.binary_cross_entropy(predictions[target_key], targets[target_key])
                            else:
                                raise ValueError(f"Unknown loss function: {head.loss_fn}")
                        
                        total_loss += head.loss_weight * loss
        
        return total_loss


@gin.configurable
def bpnet_model(tasks: List[str],
                filters: int,
                n_dil_layers: int,
                conv1_kernel_size: int,
                tconv_kernel_size: int,
                b_loss_weight: float = 1.0,
                c_loss_weight: float = 1.0,
                p_loss_weight: float = 1.0,
                c_splines: int = 0,
                b_splines: int = 20,
                merge_profile_reg: bool = False,
                lr: float = 0.004,
                tracks_per_task: int = 2,
                padding: str = 'same',
                batchnorm: bool = False,
                use_bias: bool = False,
                n_bias_tracks: int = 2,
                profile_metric: Optional[Callable] = None,
                count_metric: Optional[Callable] = None,
                profile_bias_window_sizes: List[int] = [1, 50],
                seqlen: Optional[int] = None,
                skip_type: str = 'residual',
                optimizer: str = 'adam') -> SeqModel:
    """Setup the BPNet model architecture in PyTorch

    Args:
      tasks: list of tasks
      filters: number of convolutional filters to use at each layer
      n_dil_layers: number of dilated convolutional filters to use
      conv1_kernel_size: kernel_size of the first convolutional layer
      tconv_kernel_size: kernel_size of the transpose/de-convolutional final layer
      b_loss_weight: binary classification weight
      c_loss_weight: total count regression weight
      p_loss_weight: profile regression weight
      c_splines: number of splines to use in the binary classification output head
      b_splines: number of splines to use in the profile regression output head
      merge_profile_reg: if True, total count and profile prediction will be part of
        a single profile output head
      lr: learning rate of the optimizer
      padding: padding in the convolutional layers
      batchnorm: if True, add Batchnorm after every layer
      use_bias: if True, correct for the bias
      n_bias_tracks: how many bias tracks to expect
      profile_bias_window_sizes: window sizes for profile bias
      seqlen: sequence length
      skip_type: skip connection type ('residual' or 'dense')
      optimizer: optimizer type ('adam', 'adamw', 'sgd')

    Returns:
      SeqModel: Complete BPNet model
    """
    # Import the required modules (these should be defined in the previous modules)
    from bpnet.heads import DilatedConv1D, DeConv1D, GlobalAvgPoolFCN, MovingAverages
    from bpnet.metrics import BPNetMetricSingleProfile, PeakPredictionProfileMetric
    from bpnet.metrics import ClassificationMetrics, RegressionMetrics
    from bpnet.losses import multinomial_nll, CountsMultinomialNLL
    
    assert p_loss_weight >= 0
    assert c_loss_weight >= 0
    assert b_loss_weight >= 0

    # Default metrics
    if profile_metric is None:
        print("Using the default profile prediction metric")
        profile_metric = PeakPredictionProfileMetric(
            pos_min_threshold=0.015,
            neg_max_threshold=0.005,
            required_min_pos_counts=2.5,
            binsizes=[1, 10]
        )

    if count_metric is None:
        print("Using the default regression prediction metrics")
        count_metric = RegressionMetrics()

    # Setup optimizer class
    optimizer_map = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    optimizer_class = optimizer_map.get(optimizer.lower(), Adam)
    optimizer_kwargs = {'lr': lr}

    # Heads -------------------------------------------------
    heads = []
    
    # Profile prediction
    if p_loss_weight > 0:
        if not merge_profile_reg:
            heads.append(ProfileHead(
                target_name='{task}/profile',
                net=DeConv1D(
                    filters=filters,
                    n_tasks=tracks_per_task,
                    tconv_kernel_size=tconv_kernel_size,
                    padding=padding,
                    n_hidden=0,
                    batchnorm=batchnorm
                ),
                loss_fn=multinomial_nll,
                loss_weight=p_loss_weight,
                postproc_fn=lambda x: F.softmax(x, dim=-1),
                use_bias=use_bias,
                bias_net=MovingAverages(window_sizes=profile_bias_window_sizes) if use_bias else None,
                bias_shape=(None, n_bias_tracks) if use_bias else None,
                metric=profile_metric
            ))
        else:
            heads.append(ProfileHead(
                target_name='{task}/profile',
                net=DeConv1D(
                    filters=filters,
                    n_tasks=tracks_per_task,
                    tconv_kernel_size=tconv_kernel_size,
                    padding=padding,
                    n_hidden=1,  # use 1 hidden layer in that case
                    batchnorm=batchnorm
                ),
                activation=clipped_exp,
                loss_fn=CountsMultinomialNLL(c_task_weight=c_loss_weight),
                loss_weight=p_loss_weight,
                use_bias=use_bias,
                bias_net=MovingAverages(window_sizes=profile_bias_window_sizes) if use_bias else None,
                bias_shape=(None, n_bias_tracks) if use_bias else None,
                metric=BPNetMetricSingleProfile(
                    count_metric=count_metric,
                    profile_metric=profile_metric
                )
            ))
            c_loss_weight = 0  # don't need to use the other count loss

    # Count regression
    if c_loss_weight > 0:
        heads.append(ScalarHead(
            target_name='{task}/counts',
            net=GlobalAvgPoolFCN(
                n_tasks=tracks_per_task,
                n_splines=c_splines,
                batchnorm=batchnorm
            ),
            activation=None,
            loss_fn='mse',
            loss_weight=c_loss_weight,
            use_bias=use_bias,
            bias_shape=(n_bias_tracks,) if use_bias else None,
            metric=count_metric,
        ))

    # Binary classification
    if b_loss_weight > 0:
        heads.append(ScalarHead(
            target_name='{task}/class',
            net=GlobalAvgPoolFCN(
                n_tasks=1,
                n_splines=b_splines,
                batchnorm=batchnorm
            ),
            activation='sigmoid',
            loss_fn='binary_crossentropy',
            loss_weight=b_loss_weight,
            metric=ClassificationMetrics(),
        ))

    # Create the complete model
    model = SeqModel(
        body=DilatedConv1D(
            filters=filters,
            conv1_kernel_size=conv1_kernel_size,
            n_dil_layers=n_dil_layers,
            padding=padding,
            batchnorm=batchnorm,
            skip_type=skip_type
        ),
        heads=heads,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        seqlen=seqlen,
    )
    
    return model


@gin.configurable
def binary_seq_model(tasks: List[str],
                     net_body: nn.Module,
                     net_head: nn.Module,
                     lr: float = 0.004,
                     seqlen: Optional[int] = None,
                     optimizer: str = 'adam') -> SeqModel:
    """Create a binary sequence classification model
    
    Args:
        tasks: list of task names
        net_body: body network (e.g., convolutional layers)
        net_head: head network (e.g., fully connected layers)
        lr: learning rate
        seqlen: sequence length
        optimizer: optimizer type
    
    Returns:
        SeqModel: Complete binary classification model
    """
    from bpnet.metrics import ClassificationMetrics
    
    # Setup optimizer
    optimizer_map = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    optimizer_class = optimizer_map.get(optimizer.lower(), Adam)
    optimizer_kwargs = {'lr': lr}
    
    # Create binary classification head
    heads = [ScalarHead(
        target_name='{task}/class',
        net=net_head,
        activation='sigmoid',
        loss_fn='binary_crossentropy',
        metric=ClassificationMetrics(),
    )]
    
    # Create the model
    model = SeqModel(
        body=net_body,
        heads=heads,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        seqlen=seqlen,
    )
    
    return model


@gin.configurable
def simple_conv_model(tasks: List[str],
                      n_conv_layers: int = 3,
                      filters: List[int] = [64, 128, 256],
                      kernel_sizes: List[int] = [7, 5, 3],
                      pool_sizes: List[int] = [2, 2, 2],
                      dropout: float = 0.2,
                      n_fc_layers: int = 2,
                      fc_units: List[int] = [512, 256],
                      lr: float = 0.001,
                      seqlen: Optional[int] = None,
                      optimizer: str = 'adam') -> SeqModel:
    """Create a simple convolutional model
    
    Args:
        tasks: list of task names
        n_conv_layers: number of convolutional layers
        filters: list of filter sizes for each conv layer
        kernel_sizes: list of kernel sizes for each conv layer
        pool_sizes: list of pooling sizes
        dropout: dropout rate
        n_fc_layers: number of fully connected layers
        fc_units: list of units for each FC layer
        lr: learning rate
        seqlen: sequence length
        optimizer: optimizer type
    
    Returns:
        SeqModel: Simple convolutional model
    """
    from bpnet.metrics import ClassificationMetrics
    
    # Ensure lists have correct length
    if len(filters) < n_conv_layers:
        filters = filters + [filters[-1]] * (n_conv_layers - len(filters))
    if len(kernel_sizes) < n_conv_layers:
        kernel_sizes = kernel_sizes + [kernel_sizes[-1]] * (n_conv_layers - len(kernel_sizes))
    if len(pool_sizes) < n_conv_layers:
        pool_sizes = pool_sizes + [pool_sizes[-1]] * (n_conv_layers - len(pool_sizes))
    if len(fc_units) < n_fc_layers:
        fc_units = fc_units + [fc_units[-1]] * (n_fc_layers - len(fc_units))
    
    # Create body network
    body_layers = []
    input_channels = 4  # DNA sequence has 4 channels (A, T, G, C)
    
    for i in range(n_conv_layers):
        body_layers.extend([
            nn.Conv1d(input_channels, filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2),
            nn.BatchNorm1d(filters[i]),
            nn.ReLU(),
            nn.MaxPool1d(pool_sizes[i]),
            nn.Dropout(dropout)
        ])
        input_channels = filters[i]
    
    body_layers.append(nn.AdaptiveAvgPool1d(1))
    body_layers.append(nn.Flatten())
    
    body = nn.Sequential(*body_layers)
    
    # Create head network
    head_layers = []
    input_size = filters[-1]
    
    for i in range(n_fc_layers):
        head_layers.extend([
            nn.Linear(input_size, fc_units[i]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        input_size = fc_units[i]
    
    head_layers.append(nn.Linear(input_size, 1))  # Binary classification
    head = nn.Sequential(*head_layers)
    
    # Setup optimizer
    optimizer_map = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    optimizer_class = optimizer_map.get(optimizer.lower(), Adam)
    optimizer_kwargs = {'lr': lr}
    
    # Create binary classification head
    heads = [ScalarHead(
        target_name='{task}/class',
        net=head,
        activation='sigmoid',
        loss_fn='binary_crossentropy',
        metric=ClassificationMetrics(),
    )]
    
    # Create the model
    model = SeqModel(
        body=body,
        heads=heads,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        seqlen=seqlen,
    )
    
    return model


def get(name: str):
    """Get model by name"""
    return get_from_module(name, globals())


# Available models
AVAILABLE = [
    'bpnet_model',
    'binary_seq_model', 
    'simple_conv_model',
    'SeqModel',
    'ProfileHead',
    'ScalarHead'
]