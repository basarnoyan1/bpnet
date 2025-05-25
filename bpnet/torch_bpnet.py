import torch
import numpy as np
from collections import OrderedDict
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms.functional import one_hot_encode
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
            raise NotImplementedError("Padding for string sequence in _adjust_seq_len not fully implemented here for one-hot context.")
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

    def predict_all(self, region, flank=0, batch_size=512, device='cpu', verbose=False):
        """Predict profiles and (optionally) contribution scores for a region.
        Args:
            region (kipoiseq.Interval): Region.
            flank (int): Flank size.
            batch_size (int): Batch size for prediction.
            device (str): Device for computation.
        Returns:
            dict: Dictionary containing predictions. Contribution scores part is placeholder.
        """
        seq_ohe = self.get_seq(region, flank=flank) # (L, 4)
        seq_ohe_batch = seq_ohe[np.newaxis] # (1, L, 4)
        
        preds = self.predict(seq_ohe_batch, batch_size=batch_size, device=device, verbose=verbose)
        # preds is {'task': array(1, L_out, C_out_task), ...}
        
        # Squeeze batch dimension from predictions
        preds_squeezed = {task: arr[0] for task, arr in preds.items()}
        
        # TODO: Add contribution score calculation here when Captum is integrated.
        # For now, contribution_scores will be empty or placeholder.
        contribution_scores = {} 

        return {'preds': preds_squeezed, 'contrib': contribution_scores}

    def predict_regions(self, regions, flank=0, batch_size=512, device='cpu', verbose=False):
        """Predict for multiple regions.
        Args:
            regions (list of kipoiseq.Interval): List of regions.
            flank (int): Flank size.
            batch_size (int): Batch size.
            device (str): Device.
        Yields:
            tuple: (kipoiseq.Interval, dict), where dict contains predictions and contrib scores.
        """
        for region in regions:
            if verbose:
                print(f"Predicting for region: {region.chrom}:{region.start}-{region.end}")
            yield region, self.predict_all(region, flank=flank, batch_size=batch_size, device=device, verbose=False)


    def export_bw(self, regions, out_prefix, flank=0, batch_size=512, device='cpu', verbose=False):
        """Export predictions as bigWig files.
        Args:
            regions (list of kipoiseq.Interval): List of regions.
            out_prefix (str): Prefix for output bigWig files.
            flank (int): Flank size.
            batch_size (int): Batch size.
            device (str): Device.
        """
        if pyBigWig is None:
            raise ImportError("pyBigWig is required to export bigWig files.")

        # Create bigWig files for each task and strand
        bigwigs_fwd = {task: pyBigWig.open(f"{out_prefix}.{task}.fwd.bw", "w") for task in self.tasks}
        bigwigs_rev = {task: pyBigWig.open(f"{out_prefix}.{task}.rev.bw", "w") for task in self.tasks}

        # Get genome sizes (example, replace with actual genome sizes source)
        # This part needs a source for chromosome lengths (e.g., a chrom.sizes file or from FASTA)
        # For now, as a placeholder:
        # genome_sizes = {'chr1': 1000000, 'chr2': 800000} # Example
        # for bw_dict in [bigwigs_fwd, bigwigs_rev]:
        #     for task_bw in bw_dict.values():
        #         task_bw.addHeader(list(genome_sizes.items()))
        # This header addition needs to be done carefully once genome info is available.
        # If regions are guaranteed to be within known chromosomes, header might be auto-generated by some tools.
        # Let's assume for now that regions define the extent, and pyBigWig handles it.

        for region, results in self.predict_regions(regions, flank=flank, batch_size=batch_size, device=device, verbose=verbose):
            preds = results['preds'] # This is {task: array (L_out, C_out_task)}
            
            # The original code might assume single-channel output per task for profile for BW export,
            # or sums over channels if multiple.
            # Let's assume C_out_task is 1 for simplicity here, or we take the first channel.
            
            pred_seq_len = next(iter(preds.values())).shape[0]
            
            # Define coordinates for the output profile
            center = (region.start + region.end) // 2
            # This assumes pred_seq_len matches the region width after flank and model processing.
            # If model changes length, coordinates need care.
            # If input_seqlen is fixed, and get_seq pads/trims to input_seqlen,
            # and model output length matches input_seqlen (e.g. 'same' padding convs),
            # then output profile length corresponds to input_seqlen.
            # The region for BW should correspond to the actual genomic region of the output profile.
            
            # If input to get_seq is `inflank_region` of width `W_inflank = region.width + 2*flank`
            # and model input is fixed to `input_seqlen`, then `_adjust_seq_len` makes it `input_seqlen`.
            # If model output length is also `input_seqlen`.
            # The output profile corresponds to a region of size `input_seqlen` centered at `region`'s center.
            
            profile_region_start = center - pred_seq_len // 2
            
            chrom = region.chrom
            starts = np.arange(profile_region_start, profile_region_start + pred_seq_len).astype(np.int32)
            ends = starts + 1

            for task in self.tasks:
                task_preds = preds[task] # (L_out, C_out_task)
                
                # Assuming C_out_task is 1 or sum/mean over channels if not.
                # For simplicity, taking first channel if multiple.
                if task_preds.ndim == 2 and task_preds.shape[1] > 1:
                    profile_values_fwd = task_preds[:, 0].astype(np.float64) # Forward strand
                    # How reverse strand is handled depends on model architecture (e.g., if it outputs both strands)
                    # Or if input was reverse-complemented.
                    # Original BPNet typically predicts for one strand, and reverse-comp sequence for other.
                    # Here, assuming task_preds might contain both if model is built that way, or just one.
                    # For now, let's assume task_preds is for the input strand.
                    # If only one profile output, use it for fwd, what about rev?
                    # This part needs clarification based on how TorchSeqModel handles strands.
                    # Simplest: assume output is for forward strand. Rev strand would need rc input.
                elif task_preds.ndim == 1: # (L_out,)
                     profile_values_fwd = task_preds.astype(np.float64)
                else: # (L_out, 1)
                     profile_values_fwd = task_preds[:,0].astype(np.float64)

                # Add to forward bigWig
                if len(starts) == len(profile_values_fwd) and len(starts) > 0:
                     bigwigs_fwd[task].addEntries([chrom] * len(starts), starts.tolist(), ends=ends.tolist(), values=profile_values_fwd.tolist())
                # Placeholder for reverse strand - this would typically come from rc prediction
                # bigwigs_rev[task].addEntries(...) 

        for bw_dict in [bigwigs_fwd, bigwigs_rev]:
            for task_bw in bw_dict.values():
                task_bw.close()
        print(f"Exported predictions to {out_prefix}.*.bw")

    # plot_regions would be complex, requires matplotlib and careful handling of tracks.
    # Skipping full implementation for now, but structure can be laid out.
    def plot_regions(self, regions, flank=0, subsample_profile=2, **kwargs):
        """Plot predictions and contribution scores for regions.
        Args:
            regions (list of kipoiseq.Interval): Regions to plot.
            flank (int): Flank size.
            subsample_profile (int): Subsample profile by this factor.
            **kwargs: Passed to plot_tracks.
        """
        print("Plotting regions (basic structure, full plotting logic depends on torch_plot_utils and contrib scores).")
        # from bpnet.plot.tracks import plot_tracks # Needs to be PyTorch compatible if used
        # For now, just print a summary of data that would be plotted.
        
        for region in regions:
            print(f"\nRegion: {str(region)}")
            results = self.predict_all(region, flank=flank)
            preds = results['preds']
            # contrib_scores = results['contrib'] # Placeholder for now

            # Example of how data might be prepared for a plotting function
            plot_data = {}
            for task in self.tasks:
                # Assuming preds[task] is (L, C), and C=1 for typical profile track
                profile = preds[task]
                if profile.ndim == 2 and profile.shape[1] == 1:
                    profile = profile[:,0]
                
                if subsample_profile > 1:
                    profile = profile[::subsample_profile]
                
                plot_data[f'{task}_pred'] = profile
                # Add true profile if available (e.g. from a dataloader)
                # Add contribution scores if available
            
            print(f"  Data prepared for task {task} (shape after subsample): {profile.shape}")
            # plot_tracks(plot_data, region_interval=region.resize(preds[self.tasks[0]].shape[0]), **kwargs)
            print("  (Plotting display skipped in this stub)")

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
                          pred_summaries=['profile/wn', 'counts/pre-act'], # Default summaries
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
