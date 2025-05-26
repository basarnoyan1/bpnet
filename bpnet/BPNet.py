import os
from typing import List, Dict, Optional, Union, Tuple, Any
import matplotlib.ticker as ticker
from genomelake.extractors import FastaExtractor
from collections import OrderedDict
from bpnet.plot.tracks import plot_tracks, filter_tracks
from bpnet.extractors import extract_seq
from bpnet.models import SeqModel
from tqdm import tqdm
from bpnet.utils import flatten_list, nested_numpy_minibatch
from concise.utils.plot import seqlogo
from concise.preprocessing import encodeDNA
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from genomelake.extractors import BigwigExtractor
import pyBigWig
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def softmax(x, axis=-1):
    """Softmax function for numpy arrays"""
    if isinstance(x, torch.Tensor):
        return F.softmax(x, dim=axis).detach().cpu().numpy()
    else:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def mean(x):
    """Mean function"""
    return sum(x) / len(x) if x else 0.0


def get_dataset_item(data, idx):
    """Extract item from dataset"""
    if isinstance(data, dict):
        return {k: v[idx] for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [item[idx] for item in data]
    else:
        return data[idx]


class BPNetSeqModel:
    """PyTorch BPNet wrapper based on SeqModel"""

    def __init__(self, seqmodel: SeqModel, fasta_file: Optional[str] = None, device: Optional[torch.device] = None):
        self.seqmodel = seqmodel
        self.tasks = self.seqmodel.tasks
        self.fasta_file = fasta_file
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.seqmodel.to(self.device)
        
        assert isinstance(self.seqmodel, SeqModel)

    @classmethod
    def from_mdir(cls, model_dir: str, device: Optional[torch.device] = None):
        """Load model from model directory"""
        from bpnet.models import SeqModel
        from bpnet.dataspecs import DataSpec
        
        # Load dataspec if available
        ds_path = os.path.join(model_dir, "dataspec.yml")
        if os.path.exists(ds_path):
            ds = DataSpec.load(ds_path)
            fasta_file = ds.fasta_file
        else:
            fasta_file = None
        
        # Load model
        model_path = os.path.join(model_dir, 'seq_model.pkl')
        if os.path.exists(model_path):
            seqmodel = SeqModel.load(model_path)
        else:
            # Try loading PyTorch checkpoint
            checkpoint_path = os.path.join(model_dir, 'model.pth')
            if os.path.exists(checkpoint_path):
                # This would require model architecture to be specified
                raise NotImplementedError("Loading from PyTorch checkpoint requires architecture specification")
            else:
                raise FileNotFoundError(f"No model found in {model_dir}")
        
        return cls(seqmodel, fasta_file=fasta_file, device=device)

    def input_seqlen(self) -> int:
        """Get input sequence length"""
        return self.seqmodel.seqlen

    def predict(self, seq: Union[np.ndarray, torch.Tensor], batch_size: int = 512) -> Dict[str, np.ndarray]:
        """Make model prediction

        Args:
            seq: numpy array or torch tensor of one-hot-encoded sequences
            batch_size: batch size

        Returns:
            dictionary key=task and value=prediction for the task
        """
        self.seqmodel.eval()
        
        # Convert to torch tensor if needed
        if isinstance(seq, np.ndarray):
            seq_tensor = torch.from_numpy(seq).float().to(self.device)
        else:
            seq_tensor = seq.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            if batch_size is None or seq_tensor.shape[0] <= batch_size:
                # Single batch prediction
                preds = self.seqmodel.predict({'seq': seq_tensor}, batch_size=None)
            else:
                # Batched prediction
                preds = self._predict_batched(seq_tensor, batch_size)
        
        # Convert predictions to expected format (profile * exp(counts))
        result = {}
        for task in self.seqmodel.tasks:
            profile_key = f'{task}/profile'
            counts_key = f'{task}/counts'
            
            if profile_key in preds and counts_key in preds:
                profile = preds[profile_key]
                counts = preds[counts_key]
                
                # Convert to numpy
                if isinstance(profile, torch.Tensor):
                    profile = profile.cpu().numpy()
                if isinstance(counts, torch.Tensor):
                    counts = counts.cpu().numpy()
                
                # Scale profile by exponential of counts
                if len(counts.shape) == 1:
                    counts = counts[:, None]  # Add dimension for broadcasting
                
                result[task] = profile * np.exp(counts)
            elif profile_key in preds:
                profile = preds[profile_key]
                if isinstance(profile, torch.Tensor):
                    profile = profile.cpu().numpy()
                result[task] = profile
        
        return result

    def _predict_batched(self, seq_tensor: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Make batched predictions"""
        n_samples = seq_tensor.shape[0]
        all_preds = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_seq = seq_tensor[i:end_idx]
            
            batch_preds = self.seqmodel.predict({'seq': batch_seq}, batch_size=None)
            all_preds.append(batch_preds)
        
        # Concatenate results
        final_preds = {}
        for key in all_preds[0].keys():
            final_preds[key] = torch.cat([pred[key] for pred in all_preds], dim=0)
        
        return final_preds

    def contrib_score_all(self, seq: Union[np.ndarray, torch.Tensor], 
                         method: str = 'grad', 
                         aggregate_strand: bool = True, 
                         batch_size: int = 512,
                         pred_summaries: List[str] = ['profile/wn', 'counts/pre-act']) -> Dict[str, np.ndarray]:
        """Compute all contribution scores using gradient-based methods

        Args:
            seq: one-hot encoded DNA sequences
            method: 'grad', 'integrated_grad', or 'saliency'
            aggregate_strand: if True, average contribution scores across strands
            batch_size: batch size when computing contribution scores
            pred_summaries: prediction summaries to compute

        Returns:
            dictionary with contribution scores
        """
        # Convert to tensor if needed
        if isinstance(seq, np.ndarray):
            seq_tensor = torch.from_numpy(seq).float().to(self.device)
        else:
            seq_tensor = seq.to(self.device)
        
        seq_tensor.requires_grad_(True)
        
        self.seqmodel.eval()
        
        # Compute gradients
        contrib_scores = {}
        
        if method == 'grad':
            contrib_scores = self._compute_gradients(seq_tensor, pred_summaries, batch_size)
        elif method == 'integrated_grad':
            contrib_scores = self._compute_integrated_gradients(seq_tensor, pred_summaries, batch_size)
        else:
            raise ValueError(f"Method {method} not implemented. Use 'grad' or 'integrated_grad'")
        
        # Convert old nomenclature
        result = {}
        for task in self.seqmodel.tasks:
            for pred_summary in pred_summaries:
                old_name = self._get_old_contrib_score_name(pred_summary)
                new_key = f"{task}/{old_name}"
                original_key = f"{task}/{pred_summary}"
                if original_key in contrib_scores:
                    result[new_key] = contrib_scores[original_key]
        
        return result

    def _compute_gradients(self, seq_tensor: torch.Tensor, pred_summaries: List[str], batch_size: int) -> Dict[str, np.ndarray]:
        """Compute gradients for contribution scores"""
        contrib_scores = {}
        
        n_samples = seq_tensor.shape[0]
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_seq = seq_tensor[i:end_idx]
            batch_seq.requires_grad_(True)
            
            # Forward pass
            preds = self.seqmodel({'seq': batch_seq})
            
            # Compute gradients for each task and prediction summary
            for task in self.seqmodel.tasks:
                for pred_summary in pred_summaries:
                    if pred_summary == 'profile/wn':
                        target_key = f'{task}/profile'
                    elif pred_summary == 'counts/pre-act':
                        target_key = f'{task}/counts'
                    else:
                        continue
                    
                    if target_key in preds:
                        target = preds[target_key]
                        
                        # Sum over all dimensions except batch and sequence
                        if target.dim() > 2:
                            target_sum = target.sum(dim=tuple(range(2, target.dim())))
                        else:
                            target_sum = target
                        
                        target_sum = target_sum.sum()
                        
                        # Compute gradients
                        grads = torch.autograd.grad(target_sum, batch_seq, 
                                                  retain_graph=True, create_graph=False)[0]
                        
                        # Store gradients
                        key = f"{task}/{pred_summary}"
                        if key not in contrib_scores:
                            contrib_scores[key] = []
                        
                        contrib_scores[key].append(grads.detach().cpu().numpy())
        
        # Concatenate all batches
        for key in contrib_scores:
            contrib_scores[key] = np.concatenate(contrib_scores[key], axis=0)
        
        return contrib_scores

    def _compute_integrated_gradients(self, seq_tensor: torch.Tensor, pred_summaries: List[str], 
                                    batch_size: int, n_steps: int = 50) -> Dict[str, np.ndarray]:
        """Compute integrated gradients"""
        # Create baseline (all zeros)
        baseline = torch.zeros_like(seq_tensor)
        
        contrib_scores = {}
        n_samples = seq_tensor.shape[0]
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_seq = seq_tensor[i:end_idx]
            batch_baseline = baseline[i:end_idx]
            
            batch_contrib = self._integrated_gradients_batch(batch_seq, batch_baseline, pred_summaries, n_steps)
            
            for key, value in batch_contrib.items():
                if key not in contrib_scores:
                    contrib_scores[key] = []
                contrib_scores[key].append(value)
        
        # Concatenate all batches
        for key in contrib_scores:
            contrib_scores[key] = np.concatenate(contrib_scores[key], axis=0)
        
        return contrib_scores

    def _integrated_gradients_batch(self, seq_batch: torch.Tensor, baseline_batch: torch.Tensor, 
                                  pred_summaries: List[str], n_steps: int) -> Dict[str, np.ndarray]:
        """Compute integrated gradients for a batch"""
        integrated_grads = {}
        
        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline_batch + alpha * (seq_batch - baseline_batch)
            interpolated.requires_grad_(True)
            
            preds = self.seqmodel({'seq': interpolated})
            
            for task in self.seqmodel.tasks:
                for pred_summary in pred_summaries:
                    if pred_summary == 'profile/wn':
                        target_key = f'{task}/profile'
                    elif pred_summary == 'counts/pre-act':
                        target_key = f'{task}/counts'
                    else:
                        continue
                    
                    if target_key in preds:
                        target = preds[target_key]
                        target_sum = target.sum()
                        
                        grads = torch.autograd.grad(target_sum, interpolated, 
                                                  retain_graph=True, create_graph=False)[0]
                        
                        key = f"{task}/{pred_summary}"
                        if key not in integrated_grads:
                            integrated_grads[key] = grads.detach()
                        else:
                            integrated_grads[key] += grads.detach()
        
        # Multiply by (input - baseline) and divide by n_steps
        for key in integrated_grads:
            integrated_grads[key] = (integrated_grads[key] / n_steps * (seq_batch - baseline_batch)).cpu().numpy()
        
        return integrated_grads

    def _get_old_contrib_score_name(self, s: str) -> str:
        """Convert new nomenclature to old nomenclature"""
        s2s = {"profile/wn": 'profile', 'counts/pre-act': 'count'}
        return s2s.get(s, s)

    def sim_pred(self, central_motif: str, side_motif: Optional[str] = None, 
                side_distances: List[int] = [], repeat: int = 128, 
                contribution: List[str] = []) -> Dict[str, np.ndarray]:
        """Embed two motifs in random sequences and obtain their average predictions.

        Args:
            central_motif: central motif sequence
            side_motif: side motif sequence
            side_distances: distances for side motif
            repeat: number of repeats
            contribution: list of contribution scores to compute
        """
        from bpnet.simulate import generate_seq, average_profiles, flatten
        
        batch_size = repeat
        seqlen = self.seqmodel.seqlen
        tasks = self.seqmodel.tasks

        # simulate sequence
        seqs = encodeDNA([generate_seq(central_motif, side_motif=side_motif,
                                      side_distances=side_distances, seqlen=seqlen)
                         for i in range(repeat)])

        # get predictions
        scaled_preds = self.predict(seqs, batch_size=batch_size)

        if contribution:
            # get the contribution scores
            contrib_scores_all = self.contrib_score_all(seqs, method='grad',
                                                      pred_summaries=['profile/wn', 'counts/pre-act'])
            contrib_scores = {t: {self._get_old_contrib_score_name(contrib_score_name): seqs * contrib_scores_all[f'{t}/{contrib_score_name}']
                                 for contrib_score_name in contribution}
                             for t in tasks}

            # merge and aggregate the profiles
            out = {"contrib": contrib_scores, "profile": scaled_preds}
        else:
            out = {"profile": scaled_preds}
        
        return average_profiles(flatten(out, "/"))

    def get_seq(self, regions: List, variants: Optional[List] = None, 
               use_strand: bool = False, fasta_file: Optional[str] = None) -> np.ndarray:
        """Get the one-hot-encoded sequence used to make model predictions"""
        if fasta_file is None:
            fasta_file = self.fasta_file

        if variants is not None:
            if use_strand:
                raise NotImplementedError("use_strand=True not implemented for variants")
            # Augment the regions using a variant
            if not isinstance(variants, list):
                variants = [variants] * len(regions)
            else:
                assert len(variants) == len(regions)
            seq = np.stack([extract_seq(interval, variant, fasta_file, one_hot=True)
                           for variant, interval in zip(variants, regions)])
        else:
            variants = [None] * len(regions)
            seq = FastaExtractor(fasta_file, use_strand=use_strand)(regions)
        return seq

    def predict_all(self, seq: Union[np.ndarray, torch.Tensor], 
                   contrib_method: Optional[str] = 'grad', 
                   batch_size: int = 512, 
                   pred_summaries: List[str] = ['profile/wn', 'counts/pre-act']) -> List[Dict]:
        """Make model prediction and compute contribution scores"""
        preds = self.predict(seq, batch_size=batch_size)

        if contrib_method is not None:
            contrib_scores = self.contrib_score_all(seq, method=contrib_method, 
                                                  aggregate_strand=True,
                                                  batch_size=batch_size, 
                                                  pred_summaries=pred_summaries)
        else:
            contrib_scores = dict()

        out = [dict(
            seq=get_dataset_item(seq, i),
            pred=get_dataset_item(preds, i),
            contrib_score=get_dataset_item(contrib_scores, i),
        ) for i in range(len(seq))]
        return out

    def predict_regions(self, regions: List,
                       variants: Optional[List] = None,
                       contrib_method: Optional[str] = 'grad',
                       pred_summaries: List[str] = ['profile/wn', 'counts/pre-act'],
                       use_strand: bool = False,
                       fasta_file: Optional[str] = None,
                       batch_size: int = 512) -> List[Dict]:
        """Predict on genomic regions"""
        seq = self.get_seq(regions, variants, use_strand=use_strand, fasta_file=fasta_file)

        preds = self.predict_all(seq, contrib_method, batch_size, pred_summaries=pred_summaries)

        # append regions
        for i in range(len(seq)):
            preds[i]['interval'] = regions[i]
            if variants is not None:
                preds[i]['variant'] = variants[i]
        return preds

    def plot_regions(self, regions: List, ds=None, variants: Optional[List] = None,
                    seqlets: List = [],
                    pred_summary: str = 'profile/wn',
                    contrib_method: str = 'grad',
                    batch_size: int = 128,
                    xlim: Optional[Tuple[int, int]] = None,
                    rotate_y: int = 0,
                    add_title: bool = True,
                    fig_height_per_track: float = 2,
                    same_ylim: bool = False,
                    fig_width: float = 20) -> List[plt.Figure]:
        """Plot predictions for genomic regions"""
        out = self.predict_regions(regions,
                                  variants=variants,
                                  contrib_method=contrib_method,
                                  batch_size=batch_size)
        figs = []
        if xlim is None:
            xmin = 0
        else:
            xmin = xlim[0]
        shifted_seqlets = [s.shift(-xmin) for s in seqlets]

        for i in range(len(out)):
            pred = out[i]
            interval = out[i]['interval']

            if ds is not None:
                obs = {task: ds.task_specs[task].load_counts([interval])[0] for task in self.tasks}
            else:
                obs = None

            title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', ''))

            # handle the DNase case
            if isinstance(pred['seq'], dict):
                seq = pred['seq']['seq']
            else:
                seq = pred['seq']

            if obs is None:
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Contrib profile", pred['contrib_score'][f"{task}/{self._get_old_contrib_score_name(pred_summary)}"] * seq),
                ] for task_idx, task in enumerate(self.tasks)]))
            else:
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Obs", obs[task]),
                    (f"{task} Contrib profile", pred['contrib_score'][f"{task}/{self._get_old_contrib_score_name(pred_summary)}"] * seq),
                ] for task_idx, task in enumerate(self.tasks)]))

            if add_title:
                title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', ''))
            else:
                title = None

            if same_ylim:
                fmax = {feature: max([np.abs(viz_dict[f"{task} {feature}"]).max() for task in self.tasks])
                       for feature in ['Pred', 'Contrib profile', 'Obs'] if any(f"{task} {feature}" in viz_dict for task in self.tasks)}

                ylim = []
                for k in viz_dict:
                    f = k.split(" ", 1)[1]
                    if "Contrib" in f and f in fmax:
                        ylim.append((-fmax[f], fmax[f]))
                    elif f in fmax:
                        ylim.append((0, fmax[f]))
                    else:
                        ylim.append(None)
            else:
                ylim = None
                
            fig = plot_tracks(filter_tracks(viz_dict, xlim),
                             seqlets=shifted_seqlets,
                             title=title,
                             fig_height_per_track=fig_height_per_track,
                             rotate_y=rotate_y,
                             fig_width=fig_width,
                             ylim=ylim,
                             legend=True)
            figs.append(fig)
        return figs

    def export_bw(self,
                 regions: List,
                 output_prefix: str,
                 fasta_file: Optional[str] = None,
                 contrib_method: str = 'grad',
                 pred_summaries: List[str] = ['profile/wn', 'counts/pre-act'],
                 batch_size: int = 512,
                 scale_contribution: bool = False,
                 flip_negative_strand: bool = False,
                 chromosomes: Optional[List[str]] = None):
        """Export predictions and model contributions to big-wig files"""
        from pysam import FastaFile
        
        logger.info("Get model predictions and contribution scores")
        out = self.predict_regions(regions,
                                  contrib_method=contrib_method,
                                  pred_summaries=pred_summaries,
                                  fasta_file=fasta_file,
                                  batch_size=batch_size)

        # Determine how many strands to write in export-bw
        n_tracks = out[0]['pred'][self.tasks[0]].shape[1]
        assert n_tracks <= 2, "More than 2 tracks predicted...please evaluate application of exporting bigwig tracks..."
        if n_tracks == 1:
            output_feats = ['preds', 'contrib.profile', 'contrib.counts']
        elif n_tracks == 2:
            output_feats = ['preds.pos', 'preds.neg', 'contrib.profile', 'contrib.counts']

        logger.info("Setup bigWigs for writing")
        # Get the genome lengths
        if fasta_file is None:
            fasta_file = self.fasta_file

        fa = FastaFile(fasta_file)
        if chromosomes is None:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths)])
        else:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths) if c in chromosomes])
        fa.close()

        # make sure the regions are in the right order
        first_chr = list(np.unique(np.array([interval.chrom for interval in regions])))
        last_chr = [c for c, l in genome.items() if c not in first_chr]
        genome = [(c, l) for c in first_chr + last_chr if c in genome]

        # open bigWigs for writing
        bws = {}
        for task in self.tasks:
            bws[task] = {}
            for feat in output_feats:
                delim = "." if not output_prefix.endswith("/") else ""
                bw_file = pyBigWig.open(f"{output_prefix}{delim}{task}.{feat}.bw", "w")
                bw_file.addHeader(genome)
                bws[task][feat] = bw_file

        def add_entry(bw, arr, interval, start_idx=0):
            """Add an entry to the bigwig file"""
            assert arr.ndim == 1
            assert start_idx < len(arr)

            if interval.stop - interval.start != len(arr):
                logger.warning(f"interval.stop - interval.start ({interval.stop - interval.start})!= len(arr) ({len(arr)})")
                logger.warning(f"Skipping the entry: {interval}")
                return
            bw.addEntries(interval.chrom, interval.start + start_idx,
                         values=arr[start_idx:].astype(float),
                         span=1, step=1)

        def to_1d_contrib(hyp_contrib, seq):
            # mask the hyp_contrib + add them up
            return (hyp_contrib * seq).sum(axis=-1)

        logger.info("Writing to bigWigs")
        prev_stop = None
        prev_chrom = None
        for i in tqdm(range(len(out))):
            interval = out[i]['interval']

            if prev_chrom != interval.chrom:
                prev_stop = 0
                prev_chrom = interval.chrom

            if prev_stop >= interval.stop:
                continue
            start_idx = max(prev_stop - interval.start, 0)

            for tid, task in enumerate(self.tasks):
                # Write predictions
                preds = out[i]['pred'][task]
                if n_tracks == 1:
                    add_entry(bws[task]['preds'], preds[:, 0],
                             interval, start_idx)
                elif n_tracks == 2:
                    add_entry(bws[task]['preds.pos'], preds[:, 0],
                             interval, start_idx)
                    if flip_negative_strand:
                        add_entry(bws[task]['preds.neg'], preds[:, 1] * -1,
                                 interval, start_idx)
                    else:
                        add_entry(bws[task]['preds.neg'], preds[:, 1],
                                 interval, start_idx)

                # Get the contribution scores
                seq = out[i]['seq']
                hyp_contrib = out[i]['contrib_score']

                if scale_contribution:
                    si_profile = preds.sum()
                    si_counts = preds.sum()
                else:
                    si_profile = 1
                    si_counts = 1

                # Check for valid sequence encoding
                if not np.all(seq.astype(bool).sum(axis=-1).max() == 1):
                    continue

                if 'profile/wn' in pred_summaries:
                    add_entry(bws[task]['contrib.profile'],
                             to_1d_contrib(hyp_contrib[f'{task}/profile'], seq) * si_profile,
                             interval, start_idx)
                if 'counts/pre-act' in pred_summaries:
                    add_entry(bws[task]['contrib.counts'],
                             to_1d_contrib(hyp_contrib[f'{task}/count'], seq) * si_counts,
                             interval, start_idx)

            prev_stop = max(interval.stop, prev_stop)

        logger.info("Done writing. Closing bigWigs")
        # Close all the big-wig files
        for task in self.tasks:
            for feat in output_feats:
                bws[task][feat].close()
        
        delim = "." if not output_prefix.endswith("/") else ""
        logger.info(f"Done! Output files stored as: {output_prefix}{delim}*")