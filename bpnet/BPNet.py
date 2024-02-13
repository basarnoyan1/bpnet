import logging
import os
from collections import OrderedDict

import numpy as np
import pyBigWig
from concise.preprocessing import encodeDNA
from genomelake.extractors import FastaExtractor
from kipoi_utils.data_utils import get_dataset_item
from tqdm import tqdm

from bpnet.extractors import extract_seq
from bpnet.plot.tracks import plot_tracks, filter_tracks
from bpnet.seqmodel import SeqModel
from bpnet.utils import flatten_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO - remove the fasta file
# TODO  is it possible to get rid of this class entirely?


def _get_old_contrib_score_name(s):
    # TODO - get rid of the old nomenclature
    s2s = {"profile/wn": "profile", "counts/pre-act": "count"}
    return s2s[s]


class BPNetSeqModel:
    """BPNet based on SeqModel"""

    def __init__(self, seqmodel, fasta_file=None):
        self.seqmodel = seqmodel
        self.tasks = self.seqmodel.tasks
        self.fasta_file = fasta_file
        assert isinstance(self.seqmodel, SeqModel)
        # TODO - add some sanity checks (profile head available etc)

    @classmethod
    def from_mdir(cls, model_dir):
        from bpnet.seqmodel import SeqModel

        # figure out also the fasta_file if present (from dataspec)
        from bpnet.dataspecs import DataSpec

        ds_path = os.path.join(model_dir, "dataspec.yml")
        if os.path.exists(ds_path):
            ds = DataSpec.load(ds_path)
            fasta_file = ds.fasta_file
        else:
            fasta_file = None
        return cls(SeqModel.from_mdir(model_dir), fasta_file=fasta_file)

    def input_seqlen(self):
        return self.seqmodel.seqlen

    def predict(self, seq, batch_size=512):
        """Make model prediction

        Args:
          seq: numpy array of one-hot-encoded array of sequences
          batch_size: batch size

        Returns:
          dictionary key=task and value=prediction for the task
        """

        preds = self.seqmodel.predict(seq, batch_size=batch_size)
        return {
            task: preds[f"{task}/profile"]
            * np.exp(preds[f"{task}/counts"][:, np.newaxis])
            for task in self.seqmodel.tasks
        }

    def contrib_score_all(
        self, seq, method="deeplift", aggregate_strand=True, pred_summaries=None
    ):
        """Compute all contribution scores

        Args:

          seq: one-hot encoded DNA sequences
          method: 'grad', 'deeplift' or 'ism'
          aggregate_strand: if True, the average contribution scores across strands will be returned
          pred_summaries:
        Returns:
          dictionary with keys: {task}/{pred_summary}/{strand_i} or {task}/{pred_summary}
          and values with the same shape as `seq` corresponding to contribution scores
        """
        if pred_summaries is None:
            pred_summaries = ["profile/wn", "counts/pre-act"]
        assert aggregate_strand

        contrib_scores = self.seqmodel.contrib_score_all(seq, method=method)

        return {
            f"{task}/"
            + _get_old_contrib_score_name(pred_summary): contrib_scores[
                f"{task}/{pred_summary}"
            ]
            for task in self.seqmodel.tasks
            for pred_summary in pred_summaries
        }

    def sim_pred(
        self,
        central_motif,
        side_motif=None,
        side_distances=None,
        repeat=128,
        contribution=None,
    ):
        """Embed two motifs in random sequences and obtain their average predictions.

        Args:
          central_motif:
          side_motif:
          side_distances:
          repeat:
          contribution: list of contribution scores
        """
        if contribution is None:
            contribution = []
        if side_distances is None:
            side_distances = []
        from bpnet.simulate import generate_seq, average_profiles, flatten

        batch_size = repeat
        seqlen = self.seqmodel.seqlen
        tasks = self.seqmodel.tasks

        # simulate sequence
        seqs = encodeDNA(
            [
                generate_seq(
                    central_motif,
                    side_motif=side_motif,
                    side_distances=side_distances,
                    seqlen=seqlen,
                )
                for _ in range(repeat)
            ]
        )

        # get predictions
        scaled_preds = self.predict(seqs, batch_size=batch_size)

        if contribution:
            # get the contribution scores (compute only the profile and counts contribution)
            contrib_scores_all = self.seqmodel.contrib_score_all(
                seqs, intp_pattern=["*/profile/wn", "*/counts/pre-act"]
            )
            contrib_scores = {
                t: {
                    _get_old_contrib_score_name(contrib_score_name): seqs
                    * contrib_scores_all[f"{t}/{contrib_score_name}"]
                    for contrib_score_name in contribution
                }
                for t in tasks
            }

            # merge and aggregate the profiles
            out = {"contrib": contrib_scores, "profile": scaled_preds}
        else:
            out = {"profile": scaled_preds}
        return average_profiles(flatten(out, "/"))

    def get_seq(self, regions, variants=None, use_strand=False, fasta_file=None):
        """Get the one-hot-encoded sequence used to make model predictions and
        optionally augment it with the variants
        """
        if fasta_file is None:
            fasta_file = self.fasta_file

        if variants is not None:
            if use_strand:
                raise NotImplementedError(
                    "use_strand=True not implemented for variants"
                )
            # Augment the regions using a variant
            if not isinstance(variants, list):
                variants = [variants] * len(regions)
            else:
                assert len(variants) == len(regions)
            seq = np.stack(
                [
                    extract_seq(interval, variant, fasta_file, one_hot=True)
                    for variant, interval in zip(variants, regions)
                ]
            )
        else:
            # variants = [None] * len(regions)
            seq = FastaExtractor(fasta_file, use_strand=use_strand)(regions)
        return seq

    def predict_all(
        self, seq, contrib_method="grad", batch_size=512, pred_summaries=None
    ):
        """Make model prediction based"""
        if pred_summaries is None:
            pred_summaries = ["profile/wn", "counts/pre-act"]
        preds = self.predict(seq, batch_size=batch_size)

        if contrib_method is not None:
            contrib_scores = self.contrib_score_all(
                seq,
                method=contrib_method,
                aggregate_strand=True,
                pred_summaries=pred_summaries,
            )
        else:
            contrib_scores = dict()

        out = [
            dict(
                seq=get_dataset_item(seq, i),
                # interval=regions[i],
                pred=get_dataset_item(preds, i),
                # TODO - shall we call it hyp_contrib score or contrib_score?
                contrib_score=get_dataset_item(contrib_scores, i),
            )
            for i in range(len(seq))
        ]
        return out

    def predict_regions(
        self,
        regions,
        variants=None,
        contrib_method="grad",
        pred_summaries=None,
        use_strand=False,
        fasta_file=None,
        batch_size=512,
    ):
        """
        Args:
          regions: list of pybedtools.Interval
          variants: a single instance or a list bpnet.extractors.Variant
          contrib_method:
          pred_summaries: 'mean' or 'max', summary function name for the profile gradients
          use_strand:
          fasta_file:
          batch_size:
          #compute_grads: if False, skip computing gradients
        """
        if pred_summaries is None:
            pred_summaries = ["profile/wn", "counts/pre-act"]
        seq = self.get_seq(
            regions, variants, use_strand=use_strand, fasta_file=fasta_file
        )

        preds = self.predict_all(
            seq, contrib_method, batch_size, pred_summaries=pred_summaries
        )

        # append regions
        for i in range(len(seq)):
            preds[i]["interval"] = regions[i]
            if variants is not None:
                preds[i]["variant"] = variants[i]
        return preds

    def plot_regions(
        self,
        regions,
        ds=None,
        variants=None,
        seqlets=None,
        pred_summary="profile/wn",
        contrib_method="grad",
        batch_size=128,
        # ylim=None,
        xlim=None,
        # seq_height=1,
        rotate_y=0,
        add_title=True,
        fig_height_per_track=2,
        same_ylim=False,
        fig_width=20,
    ):
        """Plot predictions

        Args:
          regions: list of pybedtools.Interval
          variants: a single instance or a list of bpnet.extractors.Variant
          seqlets:
          ds: DataSpec. If provided, the ground truth will be added to the plot
          pred_summary: 'mean' or 'max', summary function name for the profile gradients
          contrib_method:
          batch_size:
          xlim:
          rotate_y:
          add_title:
          fig_height_per_track:
          same_ylim:
          fig_width:
        """
        if seqlets is None:
            seqlets = []
        out = self.predict_regions(
            regions,
            variants=variants,
            contrib_method=contrib_method,
            # pred_summary=pred_summary,
            batch_size=batch_size,
        )
        figs = []
        if xlim is None:
            xmin = 0
        else:
            xmin = xlim[0]
        shifted_seqlets = [s.shift(-xmin) for s in seqlets]

        for i in range(len(out)):
            pred = out[i]
            interval = out[i]["interval"]

            if ds is not None:
                obs = {
                    task: ds.task_specs[task].load_counts([interval])[0]
                    for task in self.tasks
                }
            else:
                obs = None

            # handle the DNase case
            if isinstance(pred["seq"], dict):
                seq = pred["seq"]["seq"]
            else:
                seq = pred["seq"]

            if obs is None:
                # TODO - simplify?
                viz_dict = OrderedDict(
                    flatten_list(
                        [
                            [
                                (f"{task} Pred", pred["pred"][task]),
                                (
                                    f"{task} Contrib profile",
                                    pred["contrib_score"][
                                        f"{task}/{_get_old_contrib_score_name(pred_summary)}"
                                    ]
                                    * seq,
                                ),
                                # (f"{task} Contrib counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                            ]
                            for task_idx, task in enumerate(self.tasks)
                        ]
                    )
                )
            else:
                viz_dict = OrderedDict(
                    flatten_list(
                        [
                            [
                                (f"{task} Pred", pred["pred"][task]),
                                (f"{task} Obs", obs[task]),
                                (
                                    f"{task} Contrib profile",
                                    pred["contrib_score"][
                                        f"{task}/{_get_old_contrib_score_name(pred_summary)}"
                                    ]
                                    * seq,
                                ),
                                # (f"{task} Contrib counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                            ]
                            for task_idx, task in enumerate(self.tasks)
                        ]
                    )
                )

            if add_title:
                title = (
                    "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(
                        i=interval, v=pred.get("variant", "")
                    ),
                )
            else:
                title = None

            if same_ylim:
                fmax = {
                    feature: max(
                        [
                            np.abs(viz_dict[f"{task} {feature}"]).max()
                            for task in self.tasks
                        ]
                    )
                    for feature in ["Pred", "Contrib profile", "Obs"]
                }

                ylim = []
                for k in viz_dict:
                    f = k.split(" ", 1)[1]
                    if "Contrib" in f:
                        ylim.append((-fmax[f], fmax[f]))
                    else:
                        ylim.append((0, fmax[f]))
            else:
                ylim = None
            fig = plot_tracks(
                filter_tracks(viz_dict, xlim),
                seqlets=shifted_seqlets,
                title=title,
                fig_height_per_track=fig_height_per_track,
                rotate_y=rotate_y,
                fig_width=fig_width,
                ylim=ylim,
                legend=True,
            )
            figs.append(fig)
        return figs

    def export_bw(
        self,
        regions,
        output_prefix,
        fasta_file=None,
        contrib_method="grad",
        pred_summaries=None,
        batch_size=512,
        scale_contribution=False,
        flip_negative_strand=False,
        chromosomes=None,
    ):
        """Export predictions and model contributions to big-wig files

        Args:
          regions: list of genomic regions
          output_prefix: output file prefix
          fasta_file:
          contrib_method:
          pred_summaries:
          batch_size:
          scale_contribution: if True, multiply the contribution scores by the predicted count value
          flip_negative_strand:
          chromosomes: a list of chromosome names consisting a genome
        """
        if pred_summaries is None:
            pred_summaries = ["profile/wn", "counts/pre-act"]
        from pysam import FastaFile

        #          pred_summary: which operation to use for the profile gradients
        logger.info("Get model predictions and contribution scores")
        out = self.predict_regions(
            regions,
            contrib_method=contrib_method,
            pred_summaries=pred_summaries,
            fasta_file=fasta_file,
            batch_size=batch_size,
        )

        # Determine how many strands to write in export-bw
        n_tracks = out[0]["pred"][self.tasks[0]].shape[1]
        assert (
            n_tracks <= 2
        ), "More than 2 tracks predicted...please evaluate application of exporting bigwig tracks..."
        if n_tracks == 1:
            output_feats = ["preds", "contrib.profile", "contrib.counts"]
        elif n_tracks == 2:
            output_feats = [
                "preds.pos",
                "preds.neg",
                "contrib.profile",
                "contrib.counts",
            ]

        logger.info("Setup bigWigs for writing")
        # Get the genome lengths
        if fasta_file is None:
            fasta_file = self.fasta_file

        fa = FastaFile(fasta_file)
        if chromosomes is None:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths)])
        else:
            genome = OrderedDict(
                [(c, l) for c, l in zip(fa.references, fa.lengths) if c in chromosomes]
            )
        fa.close()

        # make sure the regions are in the right order
        first_chr = list(np.unique(np.array([interval.chrom for interval in regions])))
        last_chr = [c for c, l in genome.items() if c not in first_chr]
        genome = [(c, genome[c]) for c in first_chr + last_chr]

        # open bigWigs for writing
        bws = {}
        for task in self.tasks:
            bws[task] = {}
            for feat in output_feats:
                delim = "." if not output_prefix.endswith("/") else ""
                bw_preds_pos = pyBigWig.open(
                    f"{output_prefix}{delim}{task}.{feat}.bw", "w"
                )
                bw_preds_pos.addHeader(genome)
                bws[task][feat] = bw_preds_pos

        def add_entry(bw, arr, interval, start_idx=0):
            """Macro for adding an entry to the bigwig file

            Args:
              bw: pyBigWig file handle
              arr: 1-dimensional numpy array
              interval: genomic interval pybedtools.Interval
              start_idx: how many starting values in the array to skip
            """
            assert arr.ndim == 1
            assert start_idx < len(arr)

            if interval.stop - interval.start != len(arr):
                logger.warning(
                    f"interval.stop - interval.start ({interval.stop - interval.start})!= len(arr) ({len(arr)})"
                )
                logger.warning(f"Skipping the entry: {interval}")
                return
            bw.addEntries(
                interval.chrom,
                interval.start + start_idx,
                values=arr[start_idx:],
                span=1,
                step=1,
            )

        def to_1d_contrib(hyp_contrib, seq):
            # mask the hyp_contrib + add them up
            return (hyp_contrib * seq).sum(axis=-1)

        # interval logic to handle overlapping intervals
        #   assumption: all intervals are sorted w.r.t the start coordinate
        #   strategy: don't write values at the same position twice (skip those)
        #
        # graphical representation:
        # ...     ]    - prev_stop
        #      [     ]   - new interval 1
        #         [  ]   - added chunk from interval 1
        #   [  ]         - new interval 2 - skip
        #          [   ] - new interval 3, fully add

        logger.info("Writing to bigWigs")
        prev_stop = None  # Keep track of what the previous interval already covered
        prev_chrom = None
        for i in tqdm(range(len(out))):
            interval = out[i]["interval"]

            if prev_chrom != interval.chrom:
                # Encountered a new chromosome
                prev_stop = 0  # Restart the end-counter
                prev_chrom = interval.chrom

            if prev_stop >= interval.stop:
                # Nothing new to add to that range
                continue
            start_idx = max(prev_stop - interval.start, 0)

            for tid, task in enumerate(self.tasks):
                # Write predictions
                preds = out[i]["pred"][task]
                if n_tracks == 1:
                    add_entry(bws[task]["preds"], preds[:, 0], interval, start_idx)
                elif n_tracks == 2:
                    add_entry(bws[task]["preds.pos"], preds[:, 0], interval, start_idx)
                    if flip_negative_strand:
                        add_entry(
                            bws[task]["preds.neg"],
                            preds[:, 1] * -1,
                            interval,
                            start_idx,
                        )
                    else:
                        add_entry(
                            bws[task]["preds.neg"], preds[:, 1], interval, start_idx
                        )

                # Get the contribution scores
                seq = out[i]["seq"]
                hyp_contrib = out[i]["contrib_score"]

                if scale_contribution:
                    si_profile = preds.sum()  # Total number of counts in the region
                    si_counts = preds.sum()
                else:
                    si_profile = 1
                    si_counts = 1

                # Assertion to prevent multiple nucleotides being encoded at a genomic position.
                if not np.all(seq.astype(bool).sum(axis=-1).max() == 1):
                    continue

                if "profile/wn" in pred_summaries:
                    add_entry(
                        bws[task]["contrib.profile"],
                        to_1d_contrib(hyp_contrib[f"{task}/profile"], seq) * si_profile,
                        interval,
                        start_idx,
                    )
                if "counts/pre-act" in pred_summaries:
                    add_entry(
                        bws[task]["contrib.counts"],
                        to_1d_contrib(hyp_contrib[f"{task}/count"], seq) * si_counts,
                        interval,
                        start_idx,
                    )

            prev_stop = max(interval.stop, prev_stop)

        logger.info("Done writing. Closing bigWigs")
        # Close all the big-wig files
        for task in self.tasks:
            for feat in output_feats:
                bws[task][feat].close()
        logger.info(f"Done! Output files stored as: {output_prefix}{delim}*")
