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
    """Main sequence model combining body and heads - PyTorch implementation"""
    
    def __init__(self,
                 body: nn.Module,
                 heads: List[Union[ProfileHead, ScalarHead]],
                 tasks: List[str],
                 optimizer_class: type = Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 seqlen: Optional[int] = None,
                 input_shape: Optional[tuple] = None,
                 input_name: str = 'seq'):
        super().__init__()
        
        self.body = body
        self.heads = nn.ModuleList(heads)
        self.tasks = tasks
        self.seqlen = seqlen
        self.input_shape = input_shape or (seqlen, 4)
        self.input_name = input_name
        
        # Setup optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.004}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None
        
        # Track model components
        self.all_heads = {task: [] for task in tasks}
        self.target_names = []
        self.postproc_fns = []
        self.losses = []
        self.loss_weights = []
        
        # Build target names and organize heads
        for task in tasks:
            for head in heads:
                self.all_heads[task].append(head)
                target_name = head.target_name.format(task=task)
                self.target_names.append(target_name)
                self.postproc_fns.append(getattr(head, 'postproc_fn', None))
                self.losses.append(head.loss_fn)
                self.loss_weights.append(head.loss_weight)
        
        # Contribution functions for interpretability
        self.contrib_fns = {}
        
        # Hook for bottleneck features
        self.bottleneck_features = None
        self.body.register_forward_hook(self._save_bottleneck)
    
    def _save_bottleneck(self, module, input, output):
        """Hook to save bottleneck features"""
        self.bottleneck_features = output
    
    def create_optimizer(self):
        """Create optimizer for the model"""
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        return self.optimizer
    
    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Handle different input formats
        if isinstance(inputs, torch.Tensor):
            seq_input = inputs
            bias_inputs = {}
        elif isinstance(inputs, dict):
            seq_input = inputs.get(self.input_name, inputs.get('seq'))
            bias_inputs = {k: v for k, v in inputs.items() if k != self.input_name and k != 'seq'}
        else:
            raise ValueError("Input must be tensor or dictionary")
        
        # Forward through body
        body_output = self.body(seq_input)
        
        # Forward through heads
        outputs = {}
        for head in self.heads:
            # Get bias input if needed
            bias_input = None
            if hasattr(head, 'use_bias') and head.use_bias:
                # Try to find appropriate bias input
                for task in self.tasks:
                    bias_key_patterns = [
                        f'bias/{task}/profile',
                        f'bias/{task}/counts',
                        f'bias/{task}',
                        f'{task}/bias'
                    ]
                    for pattern in bias_key_patterns:
                        if pattern in bias_inputs:
                            bias_input = bias_inputs[pattern]
                            break
                    if bias_input is not None:
                        break
            
            # Forward through head for each task
            for task in self.tasks:
                target_key = head.target_name.format(task=task)
                head_output = head(body_output, bias_input)
                outputs[target_key] = head_output
        
        return outputs
    
    def predict(self, seq: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                batch_size: Optional[int] = 256) -> Dict[str, torch.Tensor]:
        """Predict with optional batch processing"""
        self.eval()
        
        with torch.no_grad():
            if batch_size is None or (isinstance(seq, torch.Tensor) and seq.shape[0] <= batch_size):
                # Single batch prediction
                if isinstance(seq, torch.Tensor) and seq.shape[0] > 0:
                    # Add neutral bias if needed
                    seq = self._add_neutral_bias(seq)
                
                raw_preds = self.forward(seq)
                
                # Apply post-processing
                preds = {}
                for target_name, pred, postproc_fn in zip(self.target_names, raw_preds.values(), self.postproc_fns):
                    if postproc_fn is not None:
                        preds[target_name] = postproc_fn(pred)
                    else:
                        preds[target_name] = pred
                        
                return preds
            else:
                # Batch processing
                return self._predict_batched(seq, batch_size)
    
    def _predict_batched(self, seq: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                        batch_size: int) -> Dict[str, torch.Tensor]:
        """Predict with batching for large inputs"""
        if isinstance(seq, torch.Tensor):
            n_samples = seq.shape[0]
            seq_dict = {'seq': seq}
        else:
            n_samples = next(iter(seq.values())).shape[0]
            seq_dict = seq
        
        all_preds = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            # Create batch
            batch = {}
            for key, tensor in seq_dict.items():
                batch[key] = tensor[i:end_idx]
            
            # Add neutral bias if needed
            if 'seq' in batch and len(batch) == 1:
                batch = self._add_neutral_bias(batch['seq'], as_dict=True)
            
            # Predict batch
            batch_preds = self.forward(batch)
            all_preds.append(batch_preds)
        
        # Concatenate results
        final_preds = {}
        for key in all_preds[0].keys():
            final_preds[key] = torch.cat([pred[key] for pred in all_preds], dim=0)
        
        # Apply post-processing
        processed_preds = {}
        for target_name, pred, postproc_fn in zip(self.target_names, final_preds.values(), self.postproc_fns):
            if postproc_fn is not None:
                processed_preds[target_name] = postproc_fn(pred)
            else:
                processed_preds[target_name] = pred
        
        return processed_preds
    
    def _add_neutral_bias(self, seq: torch.Tensor, as_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add neutral bias inputs if the model expects them"""
        if as_dict:
            result = {'seq': seq}
        else:
            result = seq
            
        # Check if any heads use bias
        bias_needed = any(hasattr(head, 'use_bias') and head.use_bias for head in self.heads)
        
        if bias_needed:
            batch_size, seqlen = seq.shape[0], seq.shape[1]
            
            # Create neutral bias for each task
            for task in self.tasks:
                for head in self.heads:
                    if hasattr(head, 'use_bias') and head.use_bias:
                        if hasattr(head, 'bias_shape') and head.bias_shape:
                            if head.bias_shape[0] is None:  # Variable sequence length
                                bias_shape = (batch_size, seqlen, head.bias_shape[1])
                            else:
                                bias_shape = (batch_size,) + head.bias_shape
                            
                            bias_key = f'bias/{task}/profile' if 'profile' in head.target_name else f'bias/{task}/counts'
                            neutral_bias = torch.zeros(bias_shape, device=seq.device, dtype=seq.dtype)
                            
                            if as_dict:
                                result[bias_key] = neutral_bias
        
        return result
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total loss from all heads"""
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for head in self.heads:
            if head.loss_fn is not None:
                for task in self.tasks:
                    target_key = head.target_name.format(task=task)
                    if target_key in predictions and target_key in targets:
                        pred = predictions[target_key]
                        target = targets[target_key]
                        
                        if hasattr(head.loss_fn, '__call__'):
                            loss = head.loss_fn(pred, target)
                        else:
                            # Handle string loss functions
                            if head.loss_fn == 'mse':
                                loss = F.mse_loss(pred, target)
                            elif head.loss_fn == 'binary_crossentropy':
                                loss = F.binary_cross_entropy(pred, target)
                            elif head.loss_fn == 'cross_entropy':
                                loss = F.cross_entropy(pred, target)
                            else:
                                raise ValueError(f"Unknown loss function: {head.loss_fn}")
                        
                        total_loss += head.loss_weight * loss
        
        return total_loss
    
    def get_bottleneck_features(self) -> Optional[torch.Tensor]:
        """Get bottleneck features from the last forward pass"""
        return self.bottleneck_features
    
    def bottleneck_model(self) -> nn.Module:
        """Create a model that outputs only bottleneck features"""
        class BottleneckModel(nn.Module):
            def __init__(self, body):
                super().__init__()
                self.body = body
            
            def forward(self, x):
                return self.body(x)
        
        return BottleneckModel(self.body)
    
    def evaluate(self, dataset, eval_metric: Optional[Callable] = None, 
                 batch_size: int = 256, device: Optional[torch.device] = None) -> Dict:
        """Evaluate model on dataset"""
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataset:
                # Move to device
                if isinstance(batch_inputs, dict):
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                else:
                    batch_inputs = batch_inputs.to(device)
                
                if isinstance(batch_targets, dict):
                    batch_targets = {k: v.to(device) for k, v in batch_targets.items()}
                
                # Predict
                batch_preds = self.predict(batch_inputs, batch_size=None)
                
                # Filter predictions to match targets
                filtered_preds = {k: v for k, v in batch_preds.items() if k in batch_targets}
                
                all_preds.append(filtered_preds)
                all_labels.append(batch_targets)
        
        # Concatenate all results
        final_preds = {}
        final_labels = {}
        
        for key in all_preds[0].keys():
            final_preds[key] = torch.cat([pred[key] for pred in all_preds], dim=0)
        
        for key in all_labels[0].keys():
            final_labels[key] = torch.cat([label[key] for label in all_labels], dim=0)
        
        # Compute metrics
        if eval_metric is not None:
            return eval_metric(final_labels, final_preds)
        else:
            # Use head-specific metrics
            results = {}
            task_avg_metrics = {}
            
            for task in self.tasks:
                for head in self.heads:
                    target_name = head.target_name.format(task=task)
                    if target_name in final_labels and hasattr(head, 'metric') and head.metric:
                        pred = final_preds[target_name].cpu().numpy()
                        label = final_labels[target_name].cpu().numpy()
                        
                        metric_result = head.metric(label, pred)
                        results[target_name] = metric_result
                        
                        # Collect for task averaging
                        if isinstance(metric_result, dict):
                            for metric_name, value in metric_result.items():
                                avg_key = head.target_name.replace("{task}", "avg") + "/" + metric_name
                                if avg_key not in task_avg_metrics:
                                    task_avg_metrics[avg_key] = []
                                task_avg_metrics[avg_key].append(value)
            
            # Add averaged metrics
            for key, values in task_avg_metrics.items():
                results[key] = sum(values) / len(values)
            
            return results
    
    def save(self, file_path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'tasks': self.tasks,
                'seqlen': self.seqlen,
                'input_shape': self.input_shape,
                'input_name': self.input_name,
                'optimizer_class': self.optimizer_class.__name__,
                'optimizer_kwargs': self.optimizer_kwargs
            }
        }, file_path)
    
    @classmethod
    def load(cls, file_path: str, body: nn.Module, heads: List[Union[ProfileHead, ScalarHead]]) -> 'SeqModel':
        """Load model from file"""
        checkpoint = torch.load(file_path, map_location='cpu')
        config = checkpoint['config']
        
        # Recreate model
        model = cls(
            body=body,
            heads=heads,
            tasks=config['tasks'],
            optimizer_class=globals().get(config['optimizer_class'], Adam),
            optimizer_kwargs=config['optimizer_kwargs'],
            seqlen=config['seqlen'],
            input_shape=config['input_shape'],
            input_name=config['input_name']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and model.optimizer:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model


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