import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from collections import OrderedDict, defaultdict
import logging
import pandas as pd
from tqdm import tqdm
import gin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_prefix_key(d: Dict, prefix: str) -> Dict:
    """Add prefix to all dictionary keys"""
    return {f"{prefix}{k}": v for k, v in d.items()}


def write_json(data: Dict, path: str, indent: int = 2):
    """Write dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_state = model.state_dict().copy()
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and self.best_state:
                    model.load_state_dict(self.best_state)
        
        return self.should_stop


class MetricsLogger:
    """Metrics logging and tracking"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.history = []
        
    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        entry = {'epoch': epoch, **metrics}
        self.history.append(entry)
        
        if self.log_file:
            df = pd.DataFrame(self.history)
            df.to_csv(self.log_file, index=False)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> Dict:
        """Get metrics from best epoch"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        if metric not in df.columns:
            return {}
        
        if mode == 'min':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        
        return df.iloc[best_idx].to_dict()


@gin.configurable
class SeqModelTrainer:
    """PyTorch trainer for sequence models"""
    
    def __init__(self, 
                 model,
                 train_dataset: Dataset,
                 valid_dataset: Union[Dataset, List[Tuple[str, Dataset]]],
                 output_dir: str,
                 cometml_experiment=None,
                 wandb_run=None,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: PyTorch SeqModel
            train_dataset: training dataset
            valid_dataset: validation dataset(s)
            output_dir: output directory for logging and checkpoints
            cometml_experiment: CometML experiment for logging
            wandb_run: Weights & Biases run for logging
            device: PyTorch device
        """
        self.seq_model = model
        self.train_dataset = train_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.seq_model.to(self.device)
        
        # Handle validation datasets
        if not isinstance(valid_dataset, list):
            self.valid_dataset = [('valid', valid_dataset)]
        else:
            self.valid_dataset = valid_dataset
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ckp_file = self.output_dir / "model.pth"
        if self.ckp_file.exists():
            logger.warning(f"Checkpoint file already exists: {self.ckp_file}")
        
        self.history_path = self.output_dir / "history.csv"
        self.evaluation_path = self.output_dir / "evaluation.valid.json"
        
        # Setup logging
        self.metrics_logger = MetricsLogger(str(self.history_path))
        self.tensorboard_writer = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(log_dir=str(self.output_dir))
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int, prefix: str = ""):
        """Log metrics to all configured loggers"""
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics}
        
        # Log to CSV
        self.metrics_logger.log(epoch, metrics)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, epoch)
        
        # Log to CometML
        if self.cometml_experiment:
            self.cometml_experiment.log_metrics(metrics, epoch=epoch)
        
        # Log to Weights & Biases
        if self.wandb_run:
            self.wandb_run.log(metrics, step=epoch)
    
    def _create_dataloader(self, dataset: Dataset, batch_size: int, 
                          shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Create DataLoader from dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True if shuffle else False
        )
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   max_batches: Optional[int] = None) -> Dict[str, float]:
        """Train for one epoch"""
        self.seq_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.seq_model(inputs)
            loss = self.seq_model.compute_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return {'train_loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate_epoch(self, dataloader: DataLoader, 
                      max_batches: Optional[int] = None) -> Dict[str, float]:
        """Validate for one epoch"""
        self.seq_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.seq_model(inputs)
                loss = self.seq_model.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def train(self,
              batch_size: int = 256,
              epochs: int = 100,
              early_stop_patience: int = 4,
              num_workers: int = 0,
              train_epoch_frac: float = 1.0,
              valid_epoch_frac: float = 1.0,
              train_samples_per_epoch: Optional[int] = None,
              validation_samples: Optional[int] = None,
              lr: float = 0.004,
              optimizer_class: type = torch.optim.Adam,
              scheduler_class: Optional[type] = None,
              scheduler_kwargs: Optional[Dict] = None,
              tensorboard: bool = True,
              save_best_only: bool = True):
        """Train the model
        
        Args:
            batch_size: batch size for training
            epochs: number of epochs to train
            early_stop_patience: patience for early stopping
            num_workers: number of workers for data loading
            train_epoch_frac: fraction of training data to use per epoch
            valid_epoch_frac: fraction of validation data to use per epoch
            train_samples_per_epoch: explicit number of training samples per epoch
            validation_samples: explicit number of validation samples
            lr: learning rate
            optimizer_class: optimizer class to use
            scheduler_class: learning rate scheduler class
            scheduler_kwargs: kwargs for scheduler
            tensorboard: whether to log to tensorboard
            save_best_only: whether to save only the best model
        """
        # Setup tensorboard
        if tensorboard:
            self._setup_tensorboard()
        
        # Create data loaders
        train_loader = self._create_dataloader(
            self.train_dataset, batch_size, shuffle=True, num_workers=num_workers
        )
        
        valid_loader = self._create_dataloader(
            self.valid_dataset[0][1], batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Calculate steps per epoch
        if train_samples_per_epoch:
            max_train_batches = max(int(train_samples_per_epoch / batch_size), 1)
        else:
            max_train_batches = max(int(len(train_loader) * train_epoch_frac), 1)
        
        if validation_samples:
            max_val_batches = max(int(validation_samples / batch_size), 1)
        else:
            max_val_batches = max(int(len(valid_loader) * valid_epoch_frac), 1)
        
        # Setup optimizer
        if not hasattr(self.seq_model, 'optimizer') or self.seq_model.optimizer is None:
            optimizer = optimizer_class(self.seq_model.parameters(), lr=lr)
        else:
            optimizer = self.seq_model.optimizer
        
        # Setup scheduler
        scheduler = None
        if scheduler_class:
            scheduler_kwargs = scheduler_kwargs or {}
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stop_patience, restore_best_weights=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(
                train_loader, optimizer, scheduler, max_train_batches
            )
            
            # Validate epoch
            val_metrics = self.validate_epoch(valid_loader, max_val_batches)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            self._log_metrics(epoch_metrics, epoch)
            
            # Save checkpoint
            current_val_loss = val_metrics['val_loss']
            if save_best_only:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.save_checkpoint()
            else:
                self.save_checkpoint()
            
            # Check early stopping
            if early_stopping(current_val_loss, self.seq_model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Print progress
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
        
        # Save final model
        if not save_best_only:
            self.save_checkpoint()
        
        # Log best epoch metrics
        try:
            best_metrics = self.metrics_logger.get_best_epoch('val_loss', 'min')
            if best_metrics:
                self._log_metrics(best_metrics, epoch, prefix="best-epoch")
        except Exception as e:
            logger.warning(f"Could not log best epoch metrics: {e}")
        
        # Close tensorboard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.seq_model.state_dict(),
            'optimizer_state_dict': self.seq_model.optimizer.state_dict() if self.seq_model.optimizer else None,
            'epoch': self.current_epoch,
            'config': {
                'tasks': self.seq_model.tasks,
                'seqlen': self.seq_model.seqlen,
                'input_shape': getattr(self.seq_model, 'input_shape', None),
                'input_name': getattr(self.seq_model, 'input_name', 'seq')
            }
        }
        
        torch.save(checkpoint, self.ckp_file)
        
        # Also save the complete model for compatibility
        model_pkl_path = self.output_dir / 'seq_model.pkl'
        try:
            with open(model_pkl_path, 'wb') as f:
                pickle.dump(self.seq_model, f)
        except Exception as e:
            logger.warning(f"Could not save pickled model: {e}")
    
    def evaluate(self, 
                 metric: Optional[Callable] = None,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 eval_train: bool = False,
                 eval_skip: List[str] = [],
                 save: bool = True,
                 **kwargs) -> Dict[str, Any]:
        """Evaluate the model on validation sets
        
        Args:
            metric: evaluation metric function
            batch_size: batch size for evaluation
            num_workers: number of workers for data loading
            eval_train: whether to evaluate on training set
            eval_skip: dataset names to skip
            save: whether to save results to JSON
            **kwargs: additional arguments
        """
        if len(kwargs) > 0:
            logger.warning(f"Extra kwargs provided to evaluate(): {kwargs}")
        
        # Save complete model
        self.save_checkpoint()
        
        # Construct list of datasets to evaluate
        eval_datasets = []
        if eval_train:
            eval_datasets.append(('train', self.train_dataset))
        eval_datasets.extend(self.valid_dataset)
        
        # Skip specified datasets
        if eval_skip:
            logger.info(f"Skipping datasets: {eval_skip}")
            eval_datasets = [(name, dataset) for name, dataset in eval_datasets 
                           if name not in eval_skip]
        
        # Evaluate each dataset
        results = OrderedDict()
        
        for dataset_info in eval_datasets:
            if len(dataset_info) == 2:
                dataset_name, dataset = dataset_info
                eval_metric = metric
            elif len(dataset_info) == 3:
                dataset_name, dataset, eval_metric = dataset_info
            else:
                raise ValueError("Dataset info must be tuple of 2 or 3 elements")
            
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            # Create dataloader
            dataloader = self._create_dataloader(
                dataset, batch_size, shuffle=False, num_workers=num_workers
            )
            
            # Evaluate
            if eval_metric:
                # Use custom metric
                dataset_results = self._evaluate_with_metric(dataloader, eval_metric)
            else:
                # Use model's built-in evaluation
                dataset_results = self._evaluate_builtin(dataloader)
            
            results[dataset_name] = dataset_results
        
        # Save results
        if save:
            write_json(results, str(self.evaluation_path), indent=2)
            logger.info(f"Saved evaluation results to {self.evaluation_path}")
        
        # Log to external services
        if self.cometml_experiment:
            flattened = flatten_dict(results, sep='/')
            self.cometml_experiment.log_metrics(flattened, prefix="eval/")
        
        if self.wandb_run:
            flattened = flatten_dict(dict_prefix_key(results, "eval/"), sep='/')
            self.wandb_run.summary.update(flattened)
        
        return results
    
    def _evaluate_with_metric(self, dataloader: DataLoader, metric: Callable) -> Dict:
        """Evaluate using custom metric function"""
        self.seq_model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                
                # Predict
                predictions = self.seq_model.predict(inputs, batch_size=None)
                
                # Convert to numpy and store
                if isinstance(predictions, dict):
                    pred_numpy = {k: v.cpu().numpy() for k, v in predictions.items()}
                else:
                    pred_numpy = predictions.cpu().numpy()
                
                if isinstance(targets, dict):
                    target_numpy = {k: v.numpy() for k, v in targets.items()}
                else:
                    target_numpy = targets.numpy()
                
                all_predictions.append(pred_numpy)
                all_targets.append(target_numpy)
        
        # Concatenate results
        if isinstance(all_predictions[0], dict):
            final_predictions = {}
            final_targets = {}
            for key in all_predictions[0].keys():
                final_predictions[key] = torch.cat([
                    torch.from_numpy(pred[key]) for pred in all_predictions
                ], dim=0).numpy()
            for key in all_targets[0].keys():
                final_targets[key] = torch.cat([
                    torch.from_numpy(target[key]) for target in all_targets
                ], dim=0).numpy()
        else:
            final_predictions = torch.cat([
                torch.from_numpy(pred) for pred in all_predictions
            ], dim=0).numpy()
            final_targets = torch.cat([
                torch.from_numpy(target) for target in all_targets
            ], dim=0).numpy()
        
        # Compute metrics
        return metric(final_targets, final_predictions)
    
    def _evaluate_builtin(self, dataloader: DataLoader) -> Dict:
        """Evaluate using model's built-in metrics"""
        return self.seq_model.evaluate(dataloader.dataset, batch_size=dataloader.batch_size)


# Utility functions for backwards compatibility
def create_trainer(model, train_dataset, valid_dataset, output_dir, **kwargs):
    """Create trainer instance"""
    return SeqModelTrainer(model, train_dataset, valid_dataset, output_dir, **kwargs)