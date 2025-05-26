import sklearn.metrics as skm
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import os
import json
from tqdm import tqdm
import matplotlib
import pandas as pd
import numpy as np
from collections import OrderedDict
import gin
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr as scipy_pearsonr, spearmanr as scipy_spearmanr, kendalltau

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert PyTorch tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def softmax(x: Union[torch.Tensor, np.ndarray], axis: int = -1) -> np.ndarray:
    """Apply softmax function"""
    if isinstance(x, torch.Tensor):
        return F.softmax(x, dim=axis).detach().cpu().numpy()
    else:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def mean(x: List[float]) -> float:
    """Compute mean of a list"""
    return sum(x) / len(x) if x else 0.0


# Metric helpers
def average_profile(pe: Dict) -> Dict:
    tasks = list(pe)
    binsizes = list(pe[tasks[0]])
    return {binsize: {"auprc": mean([pe[task][binsize]['auprc'] for task in tasks])}
            for binsize in binsizes}


def average_counts(pe: Dict) -> Dict:
    tasks = list(pe)
    metrics = list(pe[tasks[0]])
    return {metric: mean([pe[task][metric] for task in tasks])
            for metric in metrics}


def bin_counts_max(x: np.ndarray, binsize: int = 2) -> np.ndarray:
    """Bin the counts using max operation"""
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


def bin_counts_amb(x: np.ndarray, binsize: int = 2) -> np.ndarray:
    """Bin the counts handling ambiguous labels"""
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout


def bin_counts_summary(x: np.ndarray, binsize: int = 2, fn: Callable = np.max) -> np.ndarray:
    """Bin the counts using summary function"""
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = np.apply_along_axis(fn, 1, x[:, (binsize * i):(binsize * (i + 1)), :])
    return xout


def permute_array(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Permute array along specified axis"""
    idx = np.random.permutation(arr.shape[axis])
    return np.take(arr, idx, axis=axis)


def eval_profile(yt: np.ndarray, yp: np.ndarray,
                 pos_min_threshold: float = 0.05,
                 neg_max_threshold: float = 0.01,
                 required_min_pos_counts: float = 2.5,
                 binsizes: List[int] = [1, 2, 4, 10]) -> pd.DataFrame:
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold below which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # Convert tensors to numpy if needed
    yt = to_numpy(yt)
    yp = to_numpy(yp)
    
    # The filtering criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold

    # make sure everything sums to one
    yp_sum = yp.sum(axis=1, keepdims=True)
    yp_sum = np.where(yp_sum == 0, 1e-8, yp_sum)  # Avoid division by zero
    yp = yp / yp_sum
    
    fracs_sum = yt.sum(axis=1, keepdims=True)
    fracs_sum = np.where(fracs_sum == 0, 1e-8, fracs_sum)
    fracs = yt / fracs_sum

    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = []
    
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambiguous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambiguous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / max(np.sum(y_true >= 0), 1)
        n_positives = np.sum(y_true == 1)
        n_ambiguous = np.sum(y_true == -1)
        frac_ambiguous = n_ambiguous / y_true.size

        try:
            res = auprc(y_true, np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true, np.ravel(bin_counts_max(yp_random, binsize)))
        except Exception:
            res = np.nan
            res_random = np.nan

        out.append({
            "binsize": binsize,
            "auprc": res,
            "random_auprc": res_random,
            "n_positives": n_positives,
            "frac_ambiguous": frac_ambiguous,
            "imbalance": imbalance
        })

    return pd.DataFrame.from_dict(out)


# Post-processing classes
@gin.configurable
class BPNetSeparatePostproc:
    """Post-processor for separate profile and count predictions"""

    def __init__(self, tasks: List[str]):
        self.tasks = tasks

    def __call__(self, y_true: Dict, preds: List[torch.Tensor]) -> Tuple[Dict, Dict]:
        profile_preds = {task: softmax(to_numpy(preds[task_i]))
                         for task_i, task in enumerate(self.tasks)}
        count_preds = {task: to_numpy(preds[len(self.tasks) + task_i]).sum(axis=-1)
                       for task_i, task in enumerate(self.tasks)}
        
        profile_true = {task: to_numpy(y_true[f'profile/{task}'])
                        for task in self.tasks}
        counts_true = {task: to_numpy(y_true[f'counts/{task}']).sum(axis=-1)
                       for task in self.tasks}
        
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetSinglePostproc:
    """Post-processor for single track prediction"""

    def __init__(self, tasks: List[str]):
        self.tasks = tasks

    def __call__(self, y_true: Dict, preds: List[torch.Tensor]) -> Tuple[Dict, Dict]:
        profile_preds = {}
        count_preds = {}
        
        for task_i, task in enumerate(self.tasks):
            pred = to_numpy(preds[task_i])
            pred_sum = pred.sum(axis=-2, keepdims=True)
            pred_sum = np.where(pred_sum == 0, 1e-8, pred_sum)
            profile_preds[task] = pred / pred_sum
            count_preds[task] = np.log(1 + pred.sum(axis=(-2, -1)))

        profile_true = {task: to_numpy(y_true[f'profile/{task}'])
                        for task in self.tasks}
        counts_true = {task: np.log(1 + to_numpy(y_true[f'profile/{task}']).sum(axis=(-2, -1)))
                       for task in self.tasks}
        
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetMetric:
    """BPNet metrics when the net is predicting counts and profile separately"""

    def __init__(self, tasks: List[str], count_metric: Callable,
                 profile_metric: Optional[Callable] = None,
                 postproc_fn: Optional[Callable] = None):
        self.tasks = tasks
        self.count_metric = count_metric
        self.profile_metric = profile_metric

        if postproc_fn is None:
            self.postproc_fn = BPNetSeparatePostproc(tasks=self.tasks)
        else:
            self.postproc_fn = postproc_fn

    def __call__(self, y_true: Dict, preds: List[torch.Tensor]) -> Dict:
        y_true, preds = self.postproc_fn(y_true, preds)

        out = {}
        out["counts"] = {task: self.count_metric(y_true['counts'][task],
                                                 preds['counts'][task])
                         for task in self.tasks}
        out["counts"]['avg'] = average_counts(out["counts"])
        out["avg"] = {"counts": out["counts"]['avg']}

        if self.profile_metric is not None:
            out["profile"] = {task: self.profile_metric(y_true['profile'][task],
                                                        preds['profile'][task])
                              for task in self.tasks}
            out["profile"]['avg'] = average_profile(out["profile"])
            out["avg"]['profile'] = out["profile"]['avg']
        
        return out


@gin.configurable
class BPNetMetricSingleProfile:
    """BPNet metrics for single profile prediction"""

    def __init__(self, count_metric: Callable, profile_metric: Optional[Callable] = None):
        self.count_metric = count_metric
        self.profile_metric = profile_metric

    def __call__(self, y_true: torch.Tensor, preds: torch.Tensor) -> Dict:
        y_true = to_numpy(y_true)
        preds = to_numpy(preds)
        
        out = {}
        out["counts"] = self.count_metric(np.log(1 + y_true.sum(axis=(-2, -1))),
                                          np.log(1 + preds.sum(axis=(-2, -1))))

        if self.profile_metric is not None:
            out["profile"] = self.profile_metric(y_true, preds)
        
        return out


@gin.configurable
class PeakPredictionProfileMetric:
    """Metric for evaluating peak prediction from profiles"""

    def __init__(self, pos_min_threshold: float = 0.05,
                 neg_max_threshold: float = 0.01,
                 required_min_pos_counts: float = 2.5,
                 binsizes: List[int] = [1, 10]):
        self.pos_min_threshold = pos_min_threshold
        self.neg_max_threshold = neg_max_threshold
        self.required_min_pos_counts = required_min_pos_counts
        self.binsizes = binsizes

    def __call__(self, y_true: Union[torch.Tensor, np.ndarray], 
                 y_pred: Union[torch.Tensor, np.ndarray]) -> Dict:
        out = eval_profile(y_true, y_pred,
                           pos_min_threshold=self.pos_min_threshold,
                           neg_max_threshold=self.neg_max_threshold,
                           required_min_pos_counts=self.required_min_pos_counts,
                           binsizes=self.binsizes)

        return {f"binsize={k}": v for k, v in out.set_index("binsize").to_dict("index").items()}


# Default metric
default_peak_pred_metric = PeakPredictionProfileMetric(pos_min_threshold=0.015,
                                                       neg_max_threshold=0.005,
                                                       required_min_pos_counts=2.5,
                                                       binsizes=[1, 10])


# Combined metrics
@gin.configurable
class BootstrapMetric:
    """Bootstrap sampling for metrics"""

    def __init__(self, metric: Callable, n: int):
        self.metric = metric
        self.n = n

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> List:
        outl = []
        for i in range(self.n):
            bsamples = (
                pd.Series(np.arange(len(y_true))).sample(frac=1, replace=True).values
            )
            outl.append(self.metric(y_true[bsamples], y_pred[bsamples]))
        return outl


@gin.configurable
class MetricsList:
    """Wraps a list of metrics into a single metric returning a list"""

    def __init__(self, metrics: List[Callable]):
        self.metrics = metrics

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> List:
        return [metric(y_true, y_pred) for metric in self.metrics]


@gin.configurable
class MetricsDict:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        return {k: metric(y_true, y_pred) for k, metric in self.metrics.items()}


@gin.configurable
class MetricsOrderedDict:
    """Wraps metrics into an OrderedDict"""

    def __init__(self, metrics: List[Tuple[str, Callable]]):
        self.metrics = metrics

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> OrderedDict:
        return OrderedDict([(k, metric(y_true, y_pred)) for k, metric in self.metrics])


# Binary classification helpers
MASK_VALUE = -1


def _mask_nan(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from arrays"""
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        logger.warning(f"y_pred contains {np.sum(np.isnan(y_pred))}/{y_pred.size} np.nan values. removing them...")
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true: np.ndarray, y_pred: np.ndarray, mask: int = MASK_VALUE) -> Tuple[np.ndarray, np.ndarray]:
    """Remove masked values from arrays"""
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true: np.ndarray, y_pred: np.ndarray, mask: int = MASK_VALUE) -> Tuple[np.ndarray, np.ndarray]:
    """Remove both NaN and masked values"""
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)


# Binary Classification Metrics
@gin.configurable
def n_positive(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Count positive samples"""
    y_true = to_numpy(y_true)
    return float(y_true.sum())


@gin.configurable
def n_negative(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Count negative samples"""
    y_true = to_numpy(y_true)
    return float((1 - y_true).sum())


@gin.configurable
def frac_positive(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Fraction of positive samples"""
    y_true = to_numpy(y_true)
    return float(y_true.mean())


@gin.configurable
def accuracy(y_true: Union[torch.Tensor, np.ndarray], 
             y_pred: Union[torch.Tensor, np.ndarray], round_vals: bool = True) -> float:
    """Classification accuracy"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round_vals:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return float(skm.accuracy_score(y_true, y_pred))


@gin.configurable
def auc(y_true: Union[torch.Tensor, np.ndarray], 
        y_pred: Union[torch.Tensor, np.ndarray], round_vals: bool = True) -> float:
    """Area under the ROC curve"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_value_nan(y_true, y_pred)

    if round_vals:
        y_true = y_true.round()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(skm.roc_auc_score(y_true, y_pred))


@gin.configurable
def auprc(y_true: Union[torch.Tensor, np.ndarray], 
          y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Area under the precision-recall curve"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(skm.average_precision_score(y_true, y_pred))


@gin.configurable
def mcc(y_true: Union[torch.Tensor, np.ndarray], 
        y_pred: Union[torch.Tensor, np.ndarray], round_vals: bool = True) -> float:
    """Matthews correlation coefficient"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round_vals:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return float(skm.matthews_corrcoef(y_true, y_pred))


@gin.configurable
def f1(y_true: Union[torch.Tensor, np.ndarray], 
       y_pred: Union[torch.Tensor, np.ndarray], round_vals: bool = True) -> float:
    """F1 score"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round_vals:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return float(skm.f1_score(y_true, y_pred))


# Regression Metrics
@gin.configurable
def cor(y_true: Union[torch.Tensor, np.ndarray], 
        y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Pearson correlation coefficient"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


@gin.configurable
def pearsonr(y_true: Union[torch.Tensor, np.ndarray], 
             y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Pearson correlation coefficient using scipy"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan
    return float(scipy_pearsonr(y_true, y_pred)[0])


@gin.configurable
def spearmanr(y_true: Union[torch.Tensor, np.ndarray], 
              y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Spearman correlation coefficient"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan
    return float(scipy_spearmanr(y_true, y_pred)[0])


@gin.configurable
def pearson_spearman(y_true: Union[torch.Tensor, np.ndarray], 
                     y_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """Both Pearson and Spearman correlation"""
    return {"pearsonr": pearsonr(y_true, y_pred),
            "spearmanr": spearmanr(y_true, y_pred)}


@gin.configurable
def mse(y_true: Union[torch.Tensor, np.ndarray], 
        y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean squared error"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return float(((y_true - y_pred) ** 2).mean())


@gin.configurable
def rmse(y_true: Union[torch.Tensor, np.ndarray], 
         y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Root mean squared error"""
    return float(np.sqrt(mse(y_true, y_pred)))


@gin.configurable
def mad(y_true: Union[torch.Tensor, np.ndarray], 
        y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean absolute deviation"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


@gin.configurable
def var_explained(y_true: Union[torch.Tensor, np.ndarray], 
                  y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Fraction of variance explained"""
    y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    if var_y_true == 0:
        return np.nan
    return float(1 - var_resid / var_y_true)


# Metric collections
classification_metrics = [
    ("auPR", auprc),
    ("auROC", auc),
    ("accuracy", accuracy),
    ("n_positive", n_positive),
    ("n_negative", n_negative),
    ("frac_positive", frac_positive),
]

regression_metrics = [
    ("mse", mse),
    ("var_explained", var_explained),
    ("pearsonr", pearsonr),
    ("spearmanr", spearmanr),
    ("mad", mad),
]


@gin.configurable
class ClassificationMetrics:
    """All classification metrics"""

    def __init__(self):
        self.classification_metric = MetricsOrderedDict(classification_metrics)

    def __call__(self, y_true: Union[torch.Tensor, np.ndarray], 
                 y_pred: Union[torch.Tensor, np.ndarray]) -> OrderedDict:
        return self.classification_metric(y_true, y_pred)


@gin.configurable
class RegressionMetrics:
    """All regression metrics"""

    def __init__(self):
        self.regression_metric = MetricsOrderedDict(regression_metrics)

    def __call__(self, y_true: Union[torch.Tensor, np.ndarray], 
                 y_pred: Union[torch.Tensor, np.ndarray]) -> OrderedDict:
        y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
        
        # Squeeze last dimension if needed
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = np.ravel(y_true)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        return self.regression_metric(y_true, y_pred)


# Available metrics
BINARY_CLASS = ["auc", "auprc", "accuracy", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "rmse", "var_explained", "pearsonr", "spearmanr"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION