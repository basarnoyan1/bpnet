import sys
import os
import pickle
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from weakref import WeakValueDictionary
import uuid
from threading import Lock
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def jupyter_nbconvert(input_ipynb: str):
    """Convert Jupyter notebook to HTML"""
    # NOTE: cwd is used since the input_ipynb could contain some strange output
    # characters like '[' which would mess up the paths
    subprocess.call(["jupyter",
                     "nbconvert",
                     os.path.basename(input_ipynb),
                     "--to", "html"],
                    cwd=os.path.dirname(input_ipynb))


def render_ipynb(template_ipynb: str, rendered_ipynb: str, params: Dict = None):
    """Render the ipython notebook

    Args:
      template_ipynb: template ipython notebook where one cell defines the following metadata:
        {"tags": ["parameters"]}
      render_ipynb: output ipython notebook path
      params: parameters used to execute the ipython notebook
    """
    if params is None:
        params = {}
    
    try:
        import papermill as pm
        import jupyter_client
    except ImportError:
        logger.error("papermill and jupyter_client required for notebook rendering")
        return

    os.makedirs(os.path.dirname(rendered_ipynb), exist_ok=True)
    kernel_name = os.environ.get("CONDA_DEFAULT_ENV", 'python3')
    # Get rid of '/' characters which are unsupported.
    kernel_name = kernel_name.replace('/', '_')
    if kernel_name not in jupyter_client.kernelspec.find_kernel_specs():
        logger.info(f"Installing the ipython kernel for the current conda environment: {kernel_name}")
        from ipykernel.kernelspec import install
        install(user=True, kernel_name=kernel_name)

    pm.execute_notebook(
        template_ipynb,  # input template
        rendered_ipynb,
        kernel_name=kernel_name,  # default kernel
        parameters=params
    )
    jupyter_nbconvert(rendered_ipynb)


def tqdm_restart():
    """Restart tqdm to not print to every line"""
    try:
        from tqdm import tqdm as tqdm_cls
        inst = tqdm_cls._instances
        for i in range(len(inst)):
            inst.pop().close()
    except Exception as e:
        logger.warning(f"Could not restart tqdm: {e}")


def touch_file(file: str, verbose: bool = True):
    """Touch file using vmtouch if available"""
    try:
        if verbose:
            add = "v"
        else:
            add = ""
        subprocess.run(["vmtouch", f'-{add}tf', file])
    except FileNotFoundError:
        # vmtouch not available, just check if file exists
        if not os.path.exists(file):
            logger.warning(f"File does not exist: {file}")


def remove_exists(output_path: str, overwrite: bool = False):
    """Remove file if it exists"""
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            raise ValueError(f"File exists {str(output_path)}. Use overwrite=True to overwrite it")


def write_pkl(obj: Any, fname: str, create_dirs: bool = True, protocol: int = 2):
    """Write object to pickle file using cloudpickle"""
    try:
        import cloudpickle
    except ImportError:
        logger.warning("cloudpickle not available, using standard pickle")
        import pickle as cloudpickle
    
    if create_dirs:
        if os.path.dirname(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    with open(fname, 'wb') as f:
        cloudpickle.dump(obj, f, protocol=protocol)


def read_pkl(fname: str) -> Any:
    """Read object from pickle file"""
    try:
        import cloudpickle
    except ImportError:
        logger.warning("cloudpickle not available, using standard pickle")
        import pickle as cloudpickle
    
    with open(fname, 'rb') as f:
        return cloudpickle.load(f)


def read_json(fname: str) -> Dict:
    """Read JSON file"""
    with open(fname) as f:
        return json.load(f)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and PyTorch tensors"""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by returning their dict representation
            try:
                return obj.__dict__
            except:
                return str(obj)
        return json.JSONEncoder.default(self, obj)


def write_json(obj: Any, fname: str, **kwargs):
    """Write object to JSON file"""
    with open(fname, "w") as f:
        return json.dump(obj, f, cls=NumpyAwareJSONEncoder, **kwargs)


# Aliases for backwards compatibility
dump = write_pkl
load = read_pkl


def _listify(arg: Any) -> List:
    """Convert argument to list if it isn't already"""
    if hasattr(type(arg), '__len__') and not isinstance(arg, str):
        return arg
    return [arg]


def to_list(l: Any) -> List:
    """Convert to list"""
    if isinstance(l, list):
        return l
    else:
        return [l]


def reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence"""
    alt_map = {'ins': '0'}
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    for k, v in alt_map.items():
        seq = seq.replace(k, v)
    bases = list(seq)
    bases = reversed([complement.get(base, base) for base in bases])
    bases = ''.join(bases)
    for k, v in alt_map.items():
        bases = bases.replace(v, k)
    return bases


def related_dump_yaml(obj: Any, path: str, verbose: bool = False):
    """Dump object to YAML using related library"""
    try:
        import related
        generated_yaml = related.to_yaml(obj,
                                         suppress_empty_values=False,
                                         suppress_map_key_values=True)
        if verbose:
            print(generated_yaml)

        with open(path, "w") as f:
            f.write(generated_yaml)
    except ImportError:
        logger.error("related library not available for YAML serialization")


def shuffle_list(l: List) -> List:
    """Shuffle list randomly"""
    return pd.Series(l).sample(frac=1).tolist()


def flatten_list(l: List[List]) -> List:
    """Flattens a nested list"""
    return [x for nl in l for x in nl]


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


class Logger(object):
    """Tee functionality in python. If this object exists,
    then all of stdout gets logged to the file

    Adopted from:
    https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/3423392#3423392
    """

    def __init__(self, name: str, mode: str = 'a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data: str):
        self.file.write(data)
        self.stdout.write(data)
        # flush right away
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def add_file_logging(output_dir: str, logger: logging.Logger, name: str = 'stdout'):
    """Add file logging to logger"""
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)
    log = Logger(os.path.join(output_dir, 'log', name + '.log'), 'a+')  # log to the file
    fh = logging.FileHandler(os.path.join(output_dir, 'log', name + '.log'), 'a+')
    fh.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return log


def halve(n: int) -> Tuple[int, int]:
    """Halve an integer"""
    return n // 2 + n % 2, n // 2


def expand_str_list(l: List[str], prefix: str = "", suffix: str = "") -> List[str]:
    """Add strings to the beginning or to the end of the string"""
    return [prefix + x + suffix for x in l]


def kv_string2dict(s: str) -> Dict:
    """Convert a key-value string: k=v,k2=v2,... into a dictionary"""
    import yaml
    return yaml.load(s.replace(",", "\n").replace("=", ": "), Loader=yaml.SafeLoader)


def dict_suffix_key(d: Dict, suffix: str) -> Dict:
    """Add suffix to all dictionary keys"""
    return {k + suffix: v for k, v in d.items()}


def dict_prefix_key(d: Dict, prefix: str) -> Dict:
    """Add prefix to all dictionary keys"""
    return {prefix + k: v for k, v in d.items()}


def kwargs_str2kwargs(hparams: str) -> Dict:
    """Converts a string to a dictionary of kwargs

    Example:
        >>> kwargs_str2kwargs("a=1;b=[1,2]")
        {'a': 1, 'b': [1, 2]}
    """
    import yaml
    return yaml.load(hparams.replace(";", "\n").replace("=", ": "), Loader=yaml.SafeLoader)


def apply_parallel(df_grouped, func: callable, n_jobs: int = -1, verbose: bool = True):
    """Apply function in parallel to grouped dataframe"""
    try:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        retLst = Parallel(n_jobs=n_jobs)(delayed(func)(group)
                                         for name, group in tqdm(df_grouped, disable=not verbose))
        return pd.concat(retLst)
    except ImportError:
        logger.error("joblib required for parallel processing")
        return None


def get_timestamp() -> str:
    """Get current time-stamp as a string: 2018-12-10_14:20:04"""
    import datetime
    import time
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


class ConditionalRun:
    """Simple class keeping track whether the command has already been run or not"""

    def __init__(self, main_cmd: str, cmd: str, output_dir: str, force: bool = False):
        self.main_cmd = main_cmd
        self.cmd = cmd
        self.output_dir = output_dir
        self.force = force

    def set_cmd(self, cmd: str):
        self.cmd = cmd
        return self

    def get_done_file(self) -> str:
        return os.path.join(self.output_dir, f".{self.main_cmd}/{self.cmd}.done")

    def done(self) -> bool:
        ret = os.path.exists(self.get_done_file())
        if self.force:
            # always run the command
            ret = False
        if ret:
            logger.info(f"Skipping {self.cmd}")
        else:
            logger.info(f"Running {self.cmd}")
        return ret

    def write(self):
        fname = self.get_done_file()
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w") as f:
            f.write(get_timestamp())


def fnmatch_any(s: str, patterns: List[str]) -> bool:
    """Check if string matches any of the patterns"""
    from fnmatch import fnmatch  # unix-style pattern matching
    return any([fnmatch(s, p) for p in patterns])


def pd_first_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Set `cols` to be the first columns in pd.DataFrame df"""
    for c in cols:
        assert c in df
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols]


def pd_col_prepend(df: pd.DataFrame, column: Union[str, List[str]], 
                   prefix: str = '', suffix: str = '') -> pd.DataFrame:
    """Add a prefix or suffix to all the columns names in pd.DataFrame"""
    if isinstance(column, list):
        for c in column:
            df[c] = prefix + df[c] + suffix
    else:
        df[column] = prefix + df[column] + suffix
    return df


def setup_torch_device(device_id: Optional[Union[int, str]] = None, 
                      memory_fraction: float = 0.45) -> torch.device:
    """Setup PyTorch device (replaces TensorFlow session setup)
    
    Args:
        device_id: GPU device ID or 'cpu'
        memory_fraction: Memory fraction for GPU (not directly applicable in PyTorch)
    
    Returns:
        torch.device: Configured PyTorch device
    """
    if device_id == 'cpu':
        device = torch.device('cpu')
        logger.info("Using CPU device")
    elif device_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"Using GPU device: {device}")
            
            # Set CUDA device
            torch.cuda.set_device(device)
            
            # Optional: Set memory growth (not exact equivalent to TF)
            if memory_fraction == 1.0:
                # Allow growth - PyTorch does this by default
                pass
            else:
                # PyTorch doesn't have direct memory fraction control
                # but we can set memory fraction via environment variable
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{int(memory_fraction * 1024)}'
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Auto-selected CUDA device")
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected CPU device")
    
    return device


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device that a PyTorch model is on"""
    return next(model.parameters()).device


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Move tensors/models to specified device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_summary(model: nn.Module) -> str:
    """Get model summary string"""
    param_counts = count_parameters(model)
    
    summary = f"""
Model Summary:
==============
Total parameters: {param_counts['total_parameters']:,}
Trainable parameters: {param_counts['trainable_parameters']:,}
Non-trainable parameters: {param_counts['non_trainable_parameters']:,}

Architecture:
{model}
"""
    return summary


def set_random_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nested_numpy_minibatch(data: Union[Dict, List, np.ndarray], batch_size: int):
    """Create minibatches from nested numpy data structures"""
    if isinstance(data, dict):
        keys = list(data.keys())
        values = list(data.values())
        n_samples = len(values[0])
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            yield {k: v[i:end_idx] for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        n_samples = len(data[0])
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            yield [item[i:end_idx] for item in data]
    
    else:
        n_samples = len(data)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            yield data[i:end_idx]


class SerializableLock(object):
    _locks = WeakValueDictionary()
    """A Serializable per-process Lock

    This wraps a normal ``threading.Lock`` object and satisfies the same
    interface. However, this lock can also be serialized and sent to different
    processes. It will not block concurrent operations between processes (for
    this you should look at ``multiprocessing.Lock`` or ``locket.lock_file``
    but will consistently deserialize into the same lock.

    This class was taken from dask.utils.py with appropriate license.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or str(uuid.uuid4())
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = Lock()
            SerializableLock._locks[self.token] = self.lock

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.lock.release(*args, **kwargs)

    def __enter__(self):
        self.lock.__enter__()

    def __exit__(self, *args):
        self.lock.__exit__(*args)

    def locked(self):
        return self.lock.locked()

    def __getstate__(self):
        return self.token

    def __setstate__(self, token):
        self.__init__(token)

    def __str__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.token)

    __repr__ = __str__


# PyTorch-specific utilities
def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str, **kwargs):
    """Save PyTorch model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         filepath: str) -> Dict:
    """Load PyTorch model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint