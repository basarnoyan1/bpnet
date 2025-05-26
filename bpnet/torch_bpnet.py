import torch
import numpy as np
from collections import OrderedDict
import pysam
import os # Added for from_mdir
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms.functional import one_hot_encode

# Attempt to import DataSpec
try:
    from bpnet.dataspecs import DataSpec
except ImportError:
    DataSpec = None # Placeholder if DataSpec is not available
    print("Warning: bpnet.dataspecs.DataSpec could not be imported. FASTA file path cannot be loaded from dataspec.yml.")
# Placeholder for pyBigWig if not available in basic environment
try:
    import pyBigWig
except ImportError:
    pyBigWig = None

from bpnet.torch_seqmodel import TorchSeqModel
# Assuming torch_plot_utils and torch_utils will be created later or functions moved
# from bpnet.plot.tracks import plot_tracks # This will need to be PyTorch compatible if it processes model internals
# from bpnet.utils import _adjust_seq_len, _get_region_center # These are likely fine

# Helper: _adjust_seq_len (from bpnet.utils) - copied here for now if not in a shared util
def _adjust_seq_len(seq, length):
    if seq.shape[0] == length:
        return seq
    elif seq.shape[0] > length:
        # trim from edges
        trim = (seq.shape[0] - length) // 2
        return seq[trim: trim + length]
    else: # length > seq.shape[0]
        # pad with N's
        pad = length - seq.shape[0]
        l_pad = pad // 2
        r_pad = pad - l_pad
        if seq.ndim == 2:
            return np.pad(seq, ((l_pad, r_pad), (0,0)), 'constant', constant_values=0.25)
        elif seq.ndim == 1:
            # Assuming string sequence, pad with 'N'
            # This part needs to be compatible with one_hot_encode if used before it
            # For now, assuming seq is already one-hot or this function is used carefully
            # Convert to string if it's not already (e.g. numpy array of characters)
            if isinstance(seq, np.ndarray):
                seq_str = "".join(seq.astype(str))
            else:
                seq_str = str(seq)
            
            pad_char = 'N'
            padded_seq = pad_char * l_pad + seq_str + pad_char * r_pad
            return padded_seq
        else:
            raise ValueError("seq.ndim has to be 1 or 2")

# Helper: _get_region_center (from bpnet.utils)
def _get_region_center(region):
    return region.chrom, (region.start + region.end) // 2


class TorchBPNetSeqModel:
    def __init__(self, seqmodel, fasta_file=None):
        if not isinstance(seqmodel, TorchSeqModel):
            raise TypeError("seqmodel must be an instance of TorchSeqModel.")
        
        self.seqmodel = seqmodel
        self.tasks = list(self.seqmodel.tasks) # Ensure it's a list copy
        self.fasta_file = fasta_file
        if fasta_file:
            self.fasta_extractor = FastaStringExtractor(fasta_file, use_strand=True) # Assuming BPNet needs stranded seq
        else:
            self.fasta_extractor = None

    @property
    def input_seqlen(self):
        return self.seqmodel.seqlen

    def predict(self, seq_input, batch_size=512, device='cpu', verbose=False):
        """
        Args:
            seq_input (np.ndarray): Batch of one-hot encoded sequences (N, L, C)
            batch_size (int): Batch size for prediction.
            device (str): 'cpu' or 'cuda'.
            verbose (bool): Print progress.
        Returns:
            dict: Dictionary of predictions, where keys are task names and values are
                  predicted profiles scaled by counts (N, L_profile, C_profile_task).
                  Shape depends on the specific head's output for profile.
        """
        if not isinstance(seq_input, np.ndarray):
            raise TypeError("seq_input must be a NumPy array.")
        if seq_input.ndim != 3:
            raise ValueError("seq_input must be a 3D array (N, L, C).")

        # Get raw predictions from TorchSeqModel (profiles and log-counts)
        # This will be a dict like {'taskA/profile': np_array, 'taskA/counts': np_array}
        raw_predictions = self.seqmodel.predict(seq_input, 
                                                batch_size=batch_size, 
                                                device=device, 
                                                verbose=verbose)
        
        final_preds = OrderedDict()
        for task in self.tasks:
            profile_key = f"{task}/profile"
            counts_key = f"{task}/counts"

            if profile_key not in raw_predictions:
                raise KeyError(f"Profile key '{profile_key}' not found in seqmodel predictions.")
            if counts_key not in raw_predictions:
                raise KeyError(f"Counts key '{counts_key}' not found in seqmodel predictions.")

            pred_profile = raw_predictions[profile_key] # (N, L_prof, C_prof_task)
            pred_counts_log = raw_predictions[counts_key] # (N, C_counts_task) or (N,) if C_counts_task=1

            pred_counts = np.exp(pred_counts_log)

            # Ensure pred_counts can be broadcast correctly with pred_profile
            # pred_profile: (N, L_prof, C_prof_task)
            # pred_counts: (N, C_counts_task)
            # We want to scale profile by counts. If C_prof_task == C_counts_task, or one of them is 1.
            # Original Keras: pred_profile * pred_counts[:, np.newaxis, :]
            # This implies pred_counts becomes (N, 1, C_counts_task) and broadcasts over L_prof.
            # This assumes C_prof_task == C_counts_task.
            # If C_counts_task is 1 (single count value per task), it becomes (N,1,1)
            # If C_prof_task is also 1 (single profile track per task), then (N,L,1) * (N,1,1) -> (N,L,1)
            
            if pred_counts.ndim == 1: # (N,) if counts head outputs single value
                pred_counts_reshaped = pred_counts[:, np.newaxis, np.newaxis] # (N, 1, 1) for broadcasting
            elif pred_counts.ndim == 2: # (N, C_counts_task)
                pred_counts_reshaped = pred_counts[:, np.newaxis, :] # (N, 1, C_counts_task)
            else:
                raise ValueError(f"Unexpected shape for pred_counts_log for task {task}: {pred_counts_log.shape}")

            final_preds[task] = pred_profile * pred_counts_reshaped
            
        return final_preds

    def get_seq(self, region, flank=0):
        """Extracts and one-hot encodes sequence for a given region.
        Args:
            region (kipoiseq.Interval): Region for which to extract sequence.
            flank (int): Length of sequence to add on each side of the region.
        Returns:
            np.ndarray: One-hot encoded sequence (L, 4).
        """
        if self.fasta_extractor is None:
            raise ValueError("FastaExtractor not available. Initialize TorchBPNetSeqModel with fasta_file.")
        
        # Adjust region for flank
        inflank_region = region.resize(region.width + 2 * flank)
        
        # Get sequence
        seq = self.fasta_extractor.extract(inflank_region)
        
        # One-hot encode
        ohe_seq = one_hot_encode(seq) # (L, 4)
        
        # Adjust length if model has fixed input length
        if self.input_seqlen is not None:
            ohe_seq = _adjust_seq_len(ohe_seq, self.input_seqlen)
            
        return ohe_seq

    def predict_all(self, region, flank=0, batch_size=512, device='cpu', verbose=False, 
                  contrib_method='deeplift', contrib_pred_summaries=None):
        """Predict profiles and (optionally) contribution scores for a region.
        Args:
            region (kipoiseq.Interval): Region.
            flank (int): Flank size.
            batch_size (int): Batch size for prediction.
            device (str): Device for computation.
            verbose (bool): Verbosity.
            contrib_method (str, optional): Method for contribution scoring (e.g., 'deeplift'). 
                                           If None, scores are not computed. Defaults to 'deeplift'.
            contrib_pred_summaries (list, optional): List of prediction summaries for which to compute
                                                    contribution scores. Defaults to ['profile/wn', 'counts/pre-act'].
        Returns:
            dict: Dictionary containing predictions and contribution scores.
        """
        if contrib_pred_summaries is None:
            contrib_pred_summaries = ['profile/wn', 'counts/pre-act']

        seq_ohe = self.get_seq(region, flank=flank) # (L, 4)
        seq_ohe_batch = seq_ohe[np.newaxis] # (1, L, 4)
        
        preds = self.predict(seq_ohe_batch, batch_size=batch_size, device=device, verbose=verbose)
        # preds is {'task': array(1, L_out, C_out_task), ...}
        
        # Squeeze batch dimension from predictions
        preds_squeezed = {task: arr[0] for task, arr in preds.items()}
        
        contribution_scores = {}
        if contrib_method is not None:
            contribution_scores = self.contrib_score_all(
                seq_input_arr=seq_ohe_batch,
                method=contrib_method,
                pred_summaries=contrib_pred_summaries,
                batch_size=batch_size,
                device=device,
                verbose=verbose
            )
            # Squeeze batch dim from contrib scores if they are (1, L, C)
            # This depends on the output shape of self.contrib_score_all
            # Assuming it returns dict of {'summary_key': array(N, L, C_in)}
            # And for single region input N=1.
            contribution_scores = {key: val[0] for key, val in contribution_scores.items()}


        return {'preds': preds_squeezed, 'contrib': contribution_scores}

    def predict_regions(self, regions, flank=0, batch_size=512, device='cpu', verbose=False, 
                        contrib_method='deeplift', contrib_pred_summaries=None, **kwargs_to_predict_all):
        """Predict for multiple regions.
        Args:
            regions (list of kipoiseq.Interval): List of regions.
            flank (int): Flank size.
            batch_size (int): Batch size.
            device (str): Device.
            contrib_method (str, optional): Method for contribution scoring.
            contrib_pred_summaries (list, optional): Prediction summaries for contribution scoring.
            **kwargs_to_predict_all: Additional keyword arguments passed to predict_all.
        Yields:
            tuple: (kipoiseq.Interval, dict), where dict contains predictions and contrib scores.
        """
        if contrib_pred_summaries is None: # Ensure default is handled for predict_all call
            contrib_pred_summaries = ['profile/wn', 'counts/pre-act']

        for region in regions:
            if verbose:
                print(f"Predicting for region: {region.chrom}:{region.start}-{region.end}")
            yield region, self.predict_all(region, flank=flank, batch_size=batch_size, device=device, verbose=False,
                                           contrib_method=contrib_method, 
                                           contrib_pred_summaries=contrib_pred_summaries, 
                                           **kwargs_to_predict_all)


    def export_bw(self, regions, out_prefix, flank=0, batch_size=512, device='cpu', verbose=False, chromosomes=None,
                  scale_contribution: bool = False, flip_negative_strand: bool = False):
        """Export predictions and contribution scores as bigWig files.
        Args:
            regions (list of kipoiseq.Interval): List of regions.
            out_prefix (str): Prefix for output bigWig files.
            flank (int): Flank size.
            batch_size (int): Batch size.
            device (str): Device.
            chromosomes (list, optional): List of chromosome names to include in the header. 
                                         If None, all chromosomes from the FASTA file are used.
        """
        if pyBigWig is None:
            raise ImportError("pyBigWig is required to export bigWig files.")

        if not self.fasta_file:
            raise ValueError("fasta_file is not defined in TorchBPNetSeqModel. "
                             "Cannot determine chromosome sizes for bigWig header.")

        genome_header = []
        with pysam.FastaFile(self.fasta_file) as fasta:
            if chromosomes is None:
                # Use all chromosomes from FASTA
                for chrom_name in fasta.references:
                    genome_header.append((chrom_name, fasta.get_reference_length(chrom_name)))
            else:
                # Use specified chromosomes
                for chrom_name in chromosomes:
                    if chrom_name not in fasta.references:
                        raise ValueError(f"Chromosome {chrom_name} not found in FASTA file {self.fasta_file}")
                    genome_header.append((chrom_name, fasta.get_reference_length(chrom_name)))
        
        if not genome_header:
            raise ValueError("No chromosome sizes could be determined. "
                             "Ensure FASTA file is valid and chromosomes are correctly specified.")

        # Create bigWig files for each task and strand
        bigwigs_fwd = {}
        bigwigs_rev = {}
        bigwigs_contrib = {} # For contribution scores
        
        try:
            for task in self.tasks:
                bw_fwd = pyBigWig.open(f"{out_prefix}.{task}.fwd.bw", "w")
                bw_fwd.addHeader(genome_header)
                bigwigs_fwd[task] = bw_fwd

                bw_rev = pyBigWig.open(f"{out_prefix}.{task}.rev.bw", "w")
                bw_rev.addHeader(genome_header)
                bigwigs_rev[task] = bw_rev
            
            # Placeholder for opening contrib bigwigs, will be done per region if contrib_key is new
            # This is because contrib_keys might not be known in advance like self.tasks
            
            # Pass contrib_method and relevant params from export_bw to predict_regions
            # predict_regions will then pass them to predict_all
            # Default for contrib_method in predict_all is 'deeplift', so if not specified here, it will use that.
            # If want to disable contrib for export_bw, pass contrib_method=None
            predict_regions_iter = self.predict_regions(
                regions, flank=flank, batch_size=batch_size, device=device, verbose=verbose,
                # Default contrib_method in predict_all is 'deeplift'. 
                # To disable during export, pass contrib_method=None to export_bw.
                # For now, this means export_bw will calculate contribs by default.
                # This could be made explicit: e.g. contrib_method_for_export = None if you don't want them.
            )

            for region, results in predict_regions_iter:
                preds = results['preds'] # This is {task: array (L_out, C_out_task)}
                contribs = results['contrib'] # This is {contrib_key: array (L_contrib, C_in) or (L_contrib,)}

                pred_seq_len = next(iter(preds.values())).shape[0] # Assuming all tasks have same profile length
                
                center = (region.start + region.end) // 2
                profile_region_start = center - pred_seq_len // 2
                
                chrom = region.chrom
                starts = np.arange(profile_region_start, profile_region_start + pred_seq_len).astype(np.int32)
                ends = starts + 1

                for task in self.tasks:
                    task_preds = preds[task] # (L_out, C_out_task)
                    n_tracks = task_preds.shape[1] if task_preds.ndim == 2 else 1

                    profile_values_fwd = task_preds[:, 0].astype(np.float64) if task_preds.ndim == 2 else task_preds.astype(np.float64)
                    
                    if len(starts) == len(profile_values_fwd) and len(starts) > 0:
                        bigwigs_fwd[task].addEntries([chrom] * len(starts), starts.tolist(), ends=ends.tolist(), values=profile_values_fwd.tolist())

                    if n_tracks == 2:
                        profile_values_rev = task_preds[:, 1].astype(np.float64)
                        if flip_negative_strand:
                            profile_values_rev = profile_values_rev * -1
                        
                        if len(starts) == len(profile_values_rev) and len(starts) > 0:
                            bigwigs_rev[task].addEntries([chrom] * len(starts), starts.tolist(), ends=ends.tolist(), values=profile_values_rev.tolist())
                
                # Handle contribution scores
                for task in self.tasks: # Scaling factor is per-task based on its total signal
                    contrib_scaling_factor = 1.0
                    if scale_contribution:
                        task_total_signal = preds[task].sum()
                        # Avoid division by zero or very small numbers if necessary, though sum of profile usually positive
                        if task_total_signal != 0: # Basic check
                            contrib_scaling_factor = task_total_signal 
                        # Keras original had a more complex logic for scaling factor involving counts.
                        # This is a simplified version based on profile sum.

                    for contrib_key, contrib_values_array in contribs.items():
                        # Assuming contrib_key is like 'taskA/profile/deeplift' or similar
                        # We only want to scale if the contrib_key is related to the current task.
                        # This check might need refinement based on actual contrib_key naming.
                        if task not in contrib_key: # Simple check, might need to be more robust
                            continue

                        # Contribution scores are typically (L, C_in), where C_in is num input channels (e.g., 4 for DNA)
                        # For bigwig, we usually sum over the C_in dimension if it exists.
                        if contrib_values_array.ndim == 2 and contrib_values_array.shape[0] == len(starts): # (L, C_in)
                            contrib_to_write = contrib_values_array.sum(axis=1) # Sum over input channels
                        elif contrib_values_array.ndim == 1 and len(contrib_values_array) == len(starts): # (L,)
                            contrib_to_write = contrib_values_array
                        else:
                            if contrib_values_array.size > 0: # If it's not empty but doesn't match
                                print(f"Warning: Contribution score array '{contrib_key}' for task '{task}' "
                                      f"has shape {contrib_values_array.shape} and length {len(contrib_values_array)}, "
                                      f"but expected length {len(starts)} for profile-like scores. Skipping for this region.")
                            continue # Skip this contrib_key

                        scaled_contrib_values = contrib_to_write.astype(np.float64) * contrib_scaling_factor
                        
                        # Sanitize contrib_key for filename
                        safe_contrib_key = contrib_key.replace('/', '_') # Basic sanitization
                        bw_contrib_path = f"{out_prefix}.{safe_contrib_key}.bw"

                        if safe_contrib_key not in bigwigs_contrib:
                            bw_c = pyBigWig.open(bw_contrib_path, "w")
                            bw_c.addHeader(genome_header)
                            bigwigs_contrib[safe_contrib_key] = bw_c
                        
                        if len(starts) > 0:
                            bigwigs_contrib[safe_contrib_key].addEntries(
                                [chrom] * len(starts), starts.tolist(), ends=ends.tolist(), 
                                values=scaled_contrib_values.tolist()
                            )


        finally:
            all_bigwigs_to_close = [bigwigs_fwd, bigwigs_rev, bigwigs_contrib]
            for bw_collection in all_bigwigs_to_close:
                for task_bw in bw_collection.values():
                    if task_bw is not None: 
                        task_bw.close()
        
        print(f"Exported predictions to {out_prefix}.*.bw")

from bpnet.plot.tracks import plot_tracks, filter_tracks # Assuming compatibility
from bpnet.simulate import generate_seq, average_profiles, flatten # For sim_pred

    def plot_regions(self, regions, ds=None, variants=None, seqlets=None, 
                     contrib_key_to_plot=None, xlim=None, rotate_y=0, 
                     add_title=True, fig_height_per_track=2, same_ylim=False, 
                     fig_width=20, **kwargs_to_predict_all):
        """
        Plot predictions for a list of regions.
        Args:
            regions: list of kipoiseq.Interval objects
            ds: (optional) data specification object (e.g. from a HDF5)
                needs to implement ds.get_observed_data(interval, self.tasks)
            variants: (optional) list of variants to plot (kipoiseq.Variant)
            seqlets: (optional) list of seqlets to plot (kipoiseq.Seqlet)
            contrib_key_to_plot (str, optional): Which contribution summary to plot.
                                                Example: 'task/profile/deeplift'.
                                                If None, contribution scores are not plotted.
            xlim (tuple, optional): tuple of (start, end) to limit the x-axis of the plot
            rotate_y (int): rotation angle for y-axis labels
            add_title (bool): if True, add title to the plot
            fig_height_per_track (float): height of each track in inches
            same_ylim (bool): if True, use the same y-axis limits for all tracks
            fig_width (float): width of the figure in inches
            **kwargs_to_predict_all: Additional keyword arguments passed to self.predict_regions,
                                     which then passes them to self.predict_all. 
                                     This can include `contrib_method`, `batch_size`, etc.
        Returns:
            List of matplotlib figures.
        """
        if seqlets is None:
            seqlets = []

        # self.predict_regions is a generator, collect its output
        # It needs to return dicts with 'interval', 'preds', 'contrib', and 'seq'
        # The original Keras BPNet.py has a predict_regions that returns a list of dicts.
        # Let's assume our predict_regions and predict_all are structured to provide this.
        # Specifically, predict_all returns {'preds': ..., 'contrib': ...}
        # And predict_regions yields (interval, {'preds': ..., 'contrib': ...})
        # We also need 'seq' in the output of predict_all or predict_regions.
        # For now, let's modify predict_all to also return 'seq' (the input ohe_seq_batch[0])
        # And predict_regions to package it. This is a gap to fill.
        
        # TEMPORARY: Assuming predict_regions will be updated to yield dicts in the expected format.
        # For now, we'll manually call get_seq within the loop as a workaround.
        # This is NOT ideal and should be refactored by changing predict_regions/predict_all.

        prediction_outputs = []
        for region_interval, results_dict in self.predict_regions(
                regions, 
                # variants=variants, # variants not directly used by predict_regions in current torch impl.
                **kwargs_to_predict_all):
            
            # Manual fetching of sequence - this should ideally be part of results_dict
            # from predict_regions or predict_all.
            seq_ohe_for_region = self.get_seq(region_interval, flank=kwargs_to_predict_all.get('flank',0))

            prediction_outputs.append({
                'interval': region_interval,
                'preds': results_dict['preds'],
                'contrib': results_dict['contrib'],
                'seq': seq_ohe_for_region # Adding sequence here
            })


        figs = []
        for i in range(len(prediction_outputs)):
            pred_item = prediction_outputs[i]
            interval = pred_item['interval']
            
            seq_data = pred_item['seq']
            if isinstance(seq_data, dict): # Should not happen with current 'seq' population
                seq_data = seq_data.get('seq', seq_data) # Failsafe

            obs_data = None
            if ds is not None:
                try:
                    # Placeholder for actual interaction with a DataSpec-like object
                    # This method needs to be defined on the 'ds' object.
                    obs_data = ds.get_observed_data(interval, self.tasks)
                except Exception as e:
                    print(f"Warning: Failed to get observed data for interval {interval}: {e}")
                    obs_data = {} # Ensure it's a dict

            viz_dict = OrderedDict()
            for task in self.tasks:
                viz_dict[f"{task} Pred"] = pred_item['preds'][task]
                if obs_data is not None and task in obs_data and obs_data[task] is not None:
                    # Ensure observed data is adjusted to the same length as predictions if necessary
                    # This was handled by _adjust_seq_len in Keras, might need similar logic here
                    # For now, assuming lengths match or plot_tracks can handle it.
                    viz_dict[f"{task} Obs"] = obs_data[task]

                if contrib_key_to_plot is not None and pred_item['contrib']:
                    # contrib_key_to_plot could be generic e.g. "profile/deeplift"
                    # or task-specific e.g. "task1/profile/deeplift"
                    # Current contrib keys are like "task1/profile/wn/deeplift"
                    
                    # Try to find a contrib key that matches the task and the generic part of contrib_key_to_plot
                    # Example: task="task1", contrib_key_to_plot="profile/deeplift" -> look for "task1/profile/deeplift"
                    # This logic might need to be more robust.
                    actual_contrib_key_found = None
                    if contrib_key_to_plot in pred_item['contrib']: # Exact match
                        actual_contrib_key_found = contrib_key_to_plot
                    else: # Try to construct task-specific key
                        # This assumes contrib_key_to_plot is a suffix like 'profile/deeplift'
                        # And pred_item['contrib'] has keys like 'task_name/profile/deeplift'
                        constructed_key = f"{task}/{contrib_key_to_plot}" 
                        if constructed_key in pred_item['contrib']:
                            actual_contrib_key_found = constructed_key
                        # Fallback: check if contrib_key_to_plot is a generic key that might apply to all tasks
                        # (e.g. if model has a single "attention" output not tied to a task).
                        # This part is more heuristic.
                        # For now, let's assume contrib_key_to_plot is specific enough or an exact match.

                    if actual_contrib_key_found and actual_contrib_key_found in pred_item['contrib']:
                        raw_contrib_scores = pred_item['contrib'][actual_contrib_key_found]
                        
                        # Process contribution scores (e.g., saliency: grad*input, then sum over ACGT)
                        # Raw contrib scores from TorchSeqModel are (L, C_in) e.g. (1000, 4)
                        # For plotting, we usually want a 1D track.
                        if raw_contrib_scores.ndim == 2 and seq_data.ndim == 2 and \
                           raw_contrib_scores.shape == seq_data.shape:
                            # This is for methods like 'saliency' or 'input_x_gradient' that return (L,4)
                            # We multiply by input and sum over channels.
                            processed_contrib = (raw_contrib_scores * seq_data).sum(axis=-1)
                        elif raw_contrib_scores.ndim == 1: # Already 1D, e.g. if sum was done in contrib_score_all
                            processed_contrib = raw_contrib_scores
                        elif raw_contrib_scores.ndim == 2 and raw_contrib_scores.shape[1] == 1 : # (L,1)
                            processed_contrib = raw_contrib_scores[:,0]
                        else:
                            print(f"Warning: Contrib scores for {actual_contrib_key_found} have unexpected shape "
                                  f"{raw_contrib_scores.shape}. Taking sum over last axis if possible.")
                            if raw_contrib_scores.ndim > 1:
                                processed_contrib = raw_contrib_scores.sum(axis=-1)
                            else: # Cannot process, skip
                                processed_contrib = None
                        
                        if processed_contrib is not None:
                             # Ensure track name is unique if multiple tasks share a generic contrib_key_to_plot
                            track_name_contrib = f"{task} Contrib {contrib_key_to_plot.replace('/', '_')}"
                            viz_dict[track_name_contrib] = processed_contrib
                    elif contrib_key_to_plot: # If key was specified but not found for this task
                        print(f"Warning: Contrib key '{contrib_key_to_plot}' (or derived '{task}/{contrib_key_to_plot}') "
                              f"not found in contribution scores for task '{task}'. Available keys: {list(pred_item['contrib'].keys())}")


            current_xlim_start = 0
            if xlim is not None:
                current_xlim_start = xlim[0]
            
            shifted_seqlets = [s.shift(-current_xlim_start) for s in (seqlets if seqlets else [])]

            plot_title = None
            if add_title:
                plot_title = f"{interval.chrom}:{interval.start}-{interval.end} ({interval.strand})"
                if variants is not None and i < len(variants): # Assuming one variant per region if provided
                    plot_title += f" var_id:{variants[i].id if variants[i].id else 'N/A'}"
            
            ylim_values = None
            if same_ylim:
                min_val, max_val = np.inf, -np.inf
                for track_name, data in viz_dict.items():
                    if "Contrib" not in track_name: # Only for profile tracks
                        if data is not None and len(data) > 0:
                            min_val = min(min_val, np.min(data))
                            max_val = max(max_val, np.max(data))
                if np.isfinite(min_val) and np.isfinite(max_val):
                    ylim_values = (min_val, max_val)

            # Filter tracks based on xlim before passing to plot_tracks
            filtered_viz_dict = filter_tracks(viz_dict, xlim)

            fig = plot_tracks(filtered_viz_dict, 
                              seqs=seq_data if seq_data is not None and seq_data.ndim == 2 else None, # plot_tracks expects (L,4)
                              seqlets=shifted_seqlets, 
                              title=plot_title, 
                              fig_height_per_track=fig_height_per_track, 
                              rotate_y=rotate_y, 
                              fig_width=fig_width,
                              ylim=ylim_values, 
                              legend=True, # Keras default was True
                              text_align_y_coord=-0.2 # Default from Keras plot_BPNet_region
                             )
            figs.append(fig)
        return figs

    def sim_pred(self, central_motif, side_motif=None, side_distances=None, 
                 repeat=128, contrib_pred_summaries=None, contrib_method='deeplift', 
                 batch_size=512, device='cpu', verbose=False):
        """
        Simulate sequences based on motifs and predict profiles and contribution scores.

        Args:
            central_motif (str): The central motif sequence.
            side_motif (str, optional): Motif to place on the sides of the central motif.
            side_distances (list of int, optional): List of distances for the side motifs.
            repeat (int): Number of sequences to generate and average over.
            contrib_pred_summaries (list, optional): List of prediction summaries for which 
                                                    to compute contribution scores. 
                                                    If None or empty, scores are not computed.
            contrib_method (str): Method for contribution scoring (e.g., 'deeplift').
            batch_size (int): Batch size for prediction and contribution scoring.
            device (str): Device for computation ('cpu' or 'cuda').
            verbose (bool): Verbosity.

        Returns:
            OrderedDict: Averaged profiles and (optionally) contribution scores.
        """
        if contrib_pred_summaries is None:
            contrib_pred_summaries = []

        # a. Generate DNA sequences
        dna_seqs = [generate_seq(central_motif, side_motif=side_motif, 
                                 side_distances=side_distances or [], 
                                 seqlen=self.input_seqlen) for _ in range(repeat)]
        
        # b. One-hot encode the sequences
        # ohe_seqs shape: (repeat, self.input_seqlen, 4)
        ohe_seqs = np.array([one_hot_encode(s) for s in dna_seqs])
        if ohe_seqs.shape != (repeat, self.input_seqlen, 4):
             # This might happen if one_hot_encode doesn't return fixed length or if self.input_seqlen is None
             # For now, assume self.input_seqlen is correctly used by generate_seq and one_hot_encode
             pass


        # c. Get model predictions
        # scaled_preds is a dict: {task_name: profile_array (N, L_prof, C_prof_task)}
        scaled_preds = self.predict(ohe_seqs, batch_size=batch_size, device=device, verbose=verbose)

        # d. Initialize out_dict and add predictions
        out_dict = OrderedDict()
        out_dict["profile"] = scaled_preds # scaled_preds is already a dict of {task: values}

        # e. If contrib_pred_summaries is not empty, calculate contribution scores
        if contrib_pred_summaries:
            # i. Calculate raw contribution scores
            # raw_contrib_scores is a dict like {summary_key: array (N, L, C_in)}
            raw_contrib_scores = self.contrib_score_all(
                ohe_seqs, 
                method=contrib_method, 
                pred_summaries=contrib_pred_summaries, 
                batch_size=batch_size, 
                device=device, 
                verbose=verbose
            )

            # ii. Create processed_contrib_scores
            processed_contrib_scores = OrderedDict()
            
            # iii. For each summary_key, scores_array in raw_contrib_scores.items():
            for summary_key, scores_array in raw_contrib_scores.items():
                # 1. Multiply by input sequence
                # Assuming scores_array has shape (N, L, C_in) matching ohe_seqs (N, L, 4)
                if scores_array.shape == ohe_seqs.shape:
                    hyp_scores = scores_array * ohe_seqs
                else:
                    # This case needs clarification if scores_array has a different shape
                    # e.g. if it's already (N,L) or (N,L,1) for some summaries
                    # For now, try broadcasting if last dim is 1 or missing
                    if scores_array.shape[:-1] == ohe_seqs.shape[:-1] and scores_array.shape[-1] == 1:
                        hyp_scores = scores_array * ohe_seqs # Broadcasting (N,L,1) with (N,L,4)
                    elif scores_array.shape == ohe_seqs.shape[:-1]: # scores are (N,L)
                         hyp_scores = np.expand_dims(scores_array, axis=-1) * ohe_seqs # (N,L,1) * (N,L,4) -> (N,L,4)
                    else:
                        print(f"Warning: Shape mismatch for contrib score '{summary_key}' ({scores_array.shape}) "
                              f"and ohe_seqs ({ohe_seqs.shape}). Skipping multiplication by input.")
                        hyp_scores = scores_array # Use as is, or could raise error
                
                # 2. Store in processed_contrib_scores
                processed_contrib_scores[summary_key] = hyp_scores
            
            # iv. Add to out_dict
            out_dict["contrib"] = processed_contrib_scores
        
        # f. Flatten out_dict
        # Example: out_dict = {'profile': {'task1': arr1}, 'contrib': {'task1/profile/wn/deeplift': arr2}}
        # Becomes: {'profile/task1': arr1, 'contrib/task1/profile/wn/deeplift': arr2}
        flat_out_dict = flatten(out_dict, "/")

        # g. Average the profiles/scores
        # average_profiles expects dict of {key: array_N_L_C} and returns {key: array_L_C}
        averaged_results = average_profiles(flat_out_dict)
        
        # h. Return averaged_results
        return averaged_results

    @classmethod
    def from_mdir(cls, model_dir):
        # 1. Load fasta_file from dataspec.yml
        fasta_file = None
        if DataSpec is not None: # Check if DataSpec was imported
            ds_path = os.path.join(model_dir, "dataspec.yml")
            if os.path.exists(ds_path):
                try:
                    ds = DataSpec.load(ds_path)
                    fasta_file = ds.fasta_file
                    print(f"Loaded FASTA file path from dataspec.yml: {fasta_file}")
                except Exception as e:
                    print(f"Warning: Could not load dataspec.yml or extract fasta_file: {e}")
            else:
                print(f"dataspec.yml not found in {model_dir}. FASTA file path not loaded.")
        else:
            # This means DataSpec import failed earlier. Message already printed.
            pass

        # 2. Load TorchSeqModel instance
        # This relies on TorchSeqModel having a way to be loaded from a directory.
        # For now, it calls the placeholder added in Part 1.
        # TorchSeqModel is already imported at the top of the file.
        
        seqmodel_instance = TorchSeqModel.load_model_from_dir(model_dir) # This will raise NotImplementedError for now

        # 3. Instantiate TorchBPNetSeqModel
        return cls(seqmodel=seqmodel_instance, fasta_file=fasta_file)


if __name__ == '__main__':
    print("TorchBPNetSeqModel basic structure defined.")
    # This requires TorchSeqModel and its dummy components to be runnable from TorchSeqModel's __main__
    # For a standalone test here, we'd need to redefine or import those dummies.
    
    # Example: (Requires TorchSeqModel and its dummy components to be defined/imported)
    # from bpnet.torch_seqmodel import TorchSeqModel # And its dummy body/head for testing
    # N_BATCH, SEQ_LEN, IN_CHANNELS = 2, 100, 4
    # BODY_OUT_CHANNELS = 16
    # PROFILE_LEN = SEQ_LEN 
    # tasks_list = ['task1']
    # dummy_body_module = TorchSeqModel.DummyBody(IN_CHANNELS, BODY_OUT_CHANNELS, SEQ_LEN) # Assuming DummyBody is accessible
    # heads_dict = nn.ModuleDict({
    #     'task1': TorchSeqModel.DummyHead(BODY_OUT_CHANNELS, PROFILE_LEN, 1, 1)
    # })
    # torch_seq_model_inst = TorchSeqModel(body=dummy_body_module, heads=heads_dict, tasks=tasks_list, seqlen=SEQ_LEN)
    
    # bpnet_model = TorchBPNetSeqModel(seqmodel=torch_seq_model_inst, fasta_file=None) # No fasta for this basic test
    # print(f"BPNet model created with input seqlen: {bpnet_model.input_seqlen}")
    
    # dummy_seq_input = np.random.rand(N_BATCH, SEQ_LEN, IN_CHANNELS).astype(np.float32)
    # predictions = bpnet_model.predict(dummy_seq_input, batch_size=1, device='cpu')
    # print("BPNet predict output keys:", predictions.keys())
    # for task, pred_val in predictions.items():
    #    print(f" - Task '{task}' prediction shape: {pred_val.shape}")

    def contrib_score_all(self, seq_input_arr, method='deeplift', 
                          pred_summaries=None, # Default summaries handled in TorchSeqModel
                          batch_size=512, device='cpu', verbose=False):
        """
        Calculate contribution scores using the underlying TorchSeqModel.
        Args:
            seq_input_arr (np.ndarray): Batch of one-hot encoded sequences (N, L, C).
            method (str): 'deeplift' or 'saliency' or 'input_x_gradient'.
            pred_summaries (list): List of strings like 'taskA/profile/wn'.
                                   If None, uses defaults defined in TorchSeqModel.contrib_score_all.
            batch_size (int): Batch size for processing.
            device (str): Computation device.
            verbose (bool): Print progress messages.
        Returns:
            dict: Dictionary of contribution scores (NumPy arrays), 
                  keys are summary names (e.g. 'taskA/profile/wn').
        """
        if not isinstance(seq_input_arr, np.ndarray):
            raise TypeError("Input 'seq_input_arr' must be a NumPy array.")
        
        seq_input_tensor = torch.from_numpy(seq_input_arr).float().to(device)
        
        # Call TorchSeqModel's contrib_score_all
        # It returns a dict of tensors.
        contrib_scores_tensors = self.seqmodel.contrib_score_all(
            input_seq_tensor=seq_input_tensor,
            method=method,
            pred_summaries_for_contrib=pred_summaries,
            device=device,
            batch_size=batch_size # Pass batch_size to underlying method
        )
        
        # Convert output tensors to NumPy arrays
        contrib_scores_numpy = {
            key: tensor.cpu().numpy() for key, tensor in contrib_scores_tensors.items()
        }

        # The original Keras BPNetSeqModel had a specific naming convention using _get_old_contrib_score_name
        # and potentially multiplied scores by input sequence for some methods.
        # For now, this PyTorch version returns scores keyed by the full summary name.
        # Multiplication by input sequence (e.g. for saliency) is often handled by Captum's InputXGradient
        # or can be done here if required for other methods.
        # If method is 'saliency' and not 'input_x_gradient', manual multiplication might be needed.
        # Captum's Saliency returns plain gradients. InputXGradient returns gradient * input.
        # Let's assume for now that if such multiplication is needed, it's part of the chosen method's
        # interpretation or handled by the user. If 'InputXGradient' is used, it's done.
        # If 'saliency' (raw gradients) is used and element-wise product is desired for visualization,
        # that would be: seq_input_arr * contrib_scores_numpy[summary_key]

        return contrib_scores_numpy
    
    print("To run TorchBPNetSeqModel tests, ensure TorchSeqModel and its dummy components are available.")
