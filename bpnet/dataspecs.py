"""
Schemas describing the following configuration YAML files: dataspec.yml
"""
from __future__ import absolute_import
from __future__ import print_function
from copy import deepcopy
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Optional, Union
import related
from kipoi_utils.external.related.mixins import RelatedConfigMixin, RelatedLoadSaveMixin
from kipoi_utils.external.related.fields import AnyField
from kipoi_utils.external.related.fields import StrSequenceField
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@related.mutable(strict=False)
class TaskSpec(RelatedConfigMixin):
    task = related.StringField()
    # Bigwig file paths to tracks (e.g. ChIP-nexus read counts for positive and negative strand)
    tracks = StrSequenceField(str)
    peaks = related.StringField(None, required=False)

    # if True the tracks will be simply added together
    # instead of predicting them separately
    sum_tracks = related.BooleanField(False, required=False)

    # One could in the future add the assay type
    # assay = related.StringField(None, required=False)

    def load_counts(self, intervals, use_strand=True, progbar=False, return_torch=True):
        """Load counts from bigwig files
        
        Args:
            intervals: genomic intervals to extract
            use_strand: whether to use strand information
            progbar: show progress bar
            return_torch: if True, return torch tensors instead of numpy arrays
        """
        try:
            from bpnet.extractors import StrandedBigWigExtractor
        except ImportError:
            # Fallback to genomelake if bpnet extractor not available
            from genomelake.extractors import BigwigExtractor
            logger.warning("Using genomelake BigwigExtractor as fallback")
            
            tracks = []
            for track in self.tracks:
                extractor = BigwigExtractor(track)
                track_data = extractor(intervals, nan_as_zero=True)
                tracks.append(track_data)
        else:
            tracks = []
            for track in self.tracks:
                extractor = StrandedBigWigExtractor(track,
                                                  use_strand=use_strand,
                                                  nan_as_zero=True)
                track_data = extractor.extract(intervals, progbar=progbar)
                tracks.append(track_data)

        if self.sum_tracks:
            result = sum(tracks)[..., np.newaxis]  # keep the same dimension
        else:
            if use_strand and len(tracks) >= 2:
                neg_strand = np.array([getattr(s, 'strand', '*') == '-' for s in intervals]).reshape((-1, 1))
                # NOTE: this assumes that there are exactly 2 strands
                if len(tracks) != 2:
                    logger.warning(f"use_strand is True. However, there are {len(tracks)} "
                                 f"tracks and not 2 as expected. Using first two tracks.")
                pos_counts = tracks[0]
                neg_counts = tracks[1] if len(tracks) > 1 else tracks[0]
                result = np.stack([np.where(neg_strand, neg_counts, pos_counts),
                                 np.where(neg_strand, pos_counts, neg_counts)], axis=-1)
            else:
                result = np.stack(tracks, axis=-1)
        
        # Convert to PyTorch tensor if requested
        if return_torch and isinstance(result, np.ndarray):
            result = torch.from_numpy(result).float()
            
        return result

    def list_all_files(self, include_peaks=False):
        """List all file paths specified
        """
        files = list(self.tracks)  # Create a copy to avoid modifying original
        if include_peaks and self.peaks is not None:
            files.append(self.peaks)
        return files

    def touch_all_files(self, verbose=True):
        """Check if all files exist and are accessible"""
        from bpnet.utils import touch_file
        for f in self.list_all_files(include_peaks=False):
            try:
                touch_file(f, verbose)
            except Exception as e:
                logger.error(f"Could not access file {f}: {e}")
                if verbose:
                    print(f"Warning: Could not access file {f}: {e}")

    def abspath(self):
        """Use absolute filepaths
        """
        obj = deepcopy(self)
        obj.tracks = [os.path.abspath(track) for track in self.tracks]
        if self.peaks is not None:
            obj.peaks = os.path.abspath(self.peaks)
        return obj

    def validate_files(self):
        """Validate that all required files exist"""
        missing_files = []
        for track in self.tracks:
            if not os.path.exists(track):
                missing_files.append(track)
        
        if self.peaks is not None and not os.path.exists(self.peaks):
            missing_files.append(self.peaks)
            
        if missing_files:
            raise FileNotFoundError(f"Missing files for task {self.task}: {missing_files}")
        
        return True


@related.immutable(strict=True)
class BiasSpec(TaskSpec):
    # specifies for which tasks does this bias track apply
    tasks = related.SequenceField(str, required=False, default=[])

    def __post_init__(self):
        """Ensure tasks is always a list"""
        if self.tasks is None:
            self.tasks = []


@related.immutable(strict=False)
class DataSpec(RelatedLoadSaveMixin):
    """Dataset specification
    """
    # Dictionary of different bigwig files
    task_specs = related.MappingField(TaskSpec, "task",
                                      required=True,
                                      repr=True)

    # Path to the reference genome fasta file
    fasta_file = related.StringField(required=True)

    # Bias track specification
    bias_specs = related.MappingField(BiasSpec, "task",
                                      required=False,
                                      repr=True,
                                      default={})

    # Original path to the file
    path = related.StringField(required=False)

    def __post_init__(self):
        """Initialize default values"""
        if self.bias_specs is None:
            self.bias_specs = {}

    def abspath(self):
        """Convert all paths to absolute paths"""
        return DataSpec(
            task_specs={k: v.abspath() for k, v in self.task_specs.items()},
            fasta_file=os.path.abspath(self.fasta_file),
            bias_specs={k: v.abspath() for k, v in self.bias_specs.items()},
            path=self.path)

    def get_bws(self):
        """Get bigwig files organized by task"""
        return OrderedDict([(task, task_spec.tracks)
                            for task, task_spec in self.task_specs.items()])

    def list_all_files(self, include_peaks=False):
        """List all file paths specified
        """
        files = []
        files.append(self.fasta_file)
        
        for ts in self.task_specs.values():
            files.extend(ts.list_all_files(include_peaks=include_peaks))

        for ts in self.bias_specs.values():
            files.extend(ts.list_all_files(include_peaks=include_peaks))
            
        return files

    def touch_all_files(self, verbose=True):
        """Check accessibility of all files"""
        from bpnet.utils import touch_file
        
        try:
            touch_file(self.fasta_file, verbose)
        except Exception as e:
            logger.error(f"Could not access fasta file {self.fasta_file}: {e}")
            if verbose:
                print(f"Warning: Could not access fasta file {self.fasta_file}: {e}")

        for ts in self.task_specs.values():
            ts.touch_all_files(verbose=verbose)

        for ts in self.bias_specs.values():
            ts.touch_all_files(verbose=verbose)

    def validate_all_files(self):
        """Validate that all required files exist"""
        if not os.path.exists(self.fasta_file):
            raise FileNotFoundError(f"Fasta file not found: {self.fasta_file}")
        
        for task_name, task_spec in self.task_specs.items():
            try:
                task_spec.validate_files()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Task {task_name}: {e}")
        
        for bias_name, bias_spec in self.bias_specs.items():
            try:
                bias_spec.validate_files()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Bias {bias_name}: {e}")
        
        return True

    def load_counts(self, intervals, use_strand=True, progbar=False, return_torch=True):
        """Load counts for all tasks
        
        Args:
            intervals: genomic intervals to extract
            use_strand: whether to use strand information
            progbar: show progress bar
            return_torch: if True, return torch tensors instead of numpy arrays
        """
        return {task: ts.load_counts(intervals, 
                                   use_strand=use_strand, 
                                   progbar=progbar,
                                   return_torch=return_torch)
                for task, ts in self.task_specs.items()}

    def load_bias_counts(self, intervals, use_strand=True, progbar=False, return_torch=True):
        """Load bias counts for all bias specs
        
        Args:
            intervals: genomic intervals to extract
            use_strand: whether to use strand information
            progbar: show progress bar
            return_torch: if True, return torch tensors instead of numpy arrays
        """
        return {task: ts.load_counts(intervals, 
                                   use_strand=use_strand, 
                                   progbar=progbar,
                                   return_torch=return_torch)
                for task, ts in self.bias_specs.items()}

    def get_all_regions(self):
        """Get all the regions from peak files
        """
        try:
            from pybedtools import BedTool
        except ImportError:
            logger.error("pybedtools not available. Cannot load regions.")
            return []
        
        regions = []
        for task, task_spec in self.task_specs.items():
            if task_spec.peaks is not None:
                try:
                    if os.path.exists(task_spec.peaks):
                        bed_regions = list(BedTool(task_spec.peaks))
                        regions.extend(bed_regions)
                    else:
                        logger.warning(f"Peak file not found for task {task}: {task_spec.peaks}")
                except Exception as e:
                    logger.error(f"Error loading peaks for task {task}: {e}")
        return regions

    def get_task_names(self):
        """Get list of all task names"""
        return list(self.task_specs.keys())

    def get_bias_names(self):
        """Get list of all bias names"""
        return list(self.bias_specs.keys())

    def get_summary(self):
        """Get a summary of the dataspec"""
        summary = {
            'fasta_file': self.fasta_file,
            'num_tasks': len(self.task_specs),
            'num_bias_specs': len(self.bias_specs),
            'tasks': {}
        }
        
        for task_name, task_spec in self.task_specs.items():
            summary['tasks'][task_name] = {
                'num_tracks': len(task_spec.tracks),
                'has_peaks': task_spec.peaks is not None,
                'sum_tracks': task_spec.sum_tracks
            }
        
        return summary

    @classmethod
    def from_config_file(cls, config_path):
        """Load DataSpec from a configuration file with better error handling"""
        try:
            return cls.load(config_path)
        except Exception as e:
            logger.error(f"Failed to load DataSpec from {config_path}: {e}")
            raise

    def save_config(self, output_path):
        """Save DataSpec to a configuration file"""
        try:
            self.dump(output_path)
        except Exception as e:
            logger.error(f"Failed to save DataSpec to {output_path}: {e}")
            raise

    def copy(self):
        """Create a deep copy of the DataSpec"""
        return deepcopy(self)

    def subset_tasks(self, task_names):
        """Create a new DataSpec with only specified tasks"""
        if not isinstance(task_names, (list, tuple, set)):
            task_names = [task_names]
        
        # Validate that all requested tasks exist
        available_tasks = set(self.task_specs.keys())
        requested_tasks = set(task_names)
        missing_tasks = requested_tasks - available_tasks
        
        if missing_tasks:
            raise ValueError(f"Requested tasks not found: {missing_tasks}. "
                           f"Available tasks: {available_tasks}")
        
        # Create new task_specs with only requested tasks
        new_task_specs = {task: self.task_specs[task] for task in task_names}
        
        # Filter bias_specs to only include those relevant to requested tasks
        new_bias_specs = {}
        for bias_name, bias_spec in self.bias_specs.items():
            if any(task in task_names for task in bias_spec.tasks):
                new_bias_specs[bias_name] = bias_spec
        
        return DataSpec(
            task_specs=new_task_specs,
            fasta_file=self.fasta_file,
            bias_specs=new_bias_specs,
            path=self.path
        )