\
import json
import numpy as np
import os
import six

# Placeholder for bcolz and tiledb imports
try:
    import bcolz
except ImportError:
    bcolz = None

try:
    import tiledb
except ImportError:
    tiledb = None

NUM_SEQ_CHARS = 4 # Also needed by one_hot_encode_sequence

def char_to_index(ch):
    """Convert DNA character to one-hot index"""
    if ch in 'Aa':
        return 0
    elif ch in 'Cc':
        return 1
    elif ch in 'Gg':
        return 2
    elif ch in 'Tt':
        return 3
    else:  # N or other
        return -1

def one_hot_encode_sequence(seq, encoded):
    """One-hot encode a DNA sequence into a pre-allocated array"""
    # Expects encoded to be pre-shaped: (len(seq), NUM_SEQ_CHARS)
    for i, base in enumerate(seq):
        col_idx = char_to_index(base)
        if col_idx >= 0:
            encoded[i, col_idx] = 1.0
        else:
            encoded[i, :] = 0.25 # For N or unknown bases

def nan_to_zero(arr):
    """Convert NaN values to zero in place"""
    arr[np.isnan(arr)] = 0

def makedirs(path, mode=0o777, exist_ok=False):
    try:
        os.makedirs(path, mode)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise

if bcolz:
    _blosc_params = bcolz.cparams(clevel=5, shuffle=bcolz.SHUFFLE, cname="lz4")
else:
    _blosc_params = None

def write_numpy(arr, path):
    np.save(path, arr)

def write_bcolz(arr, path):
    if not bcolz:
        raise ImportError("bcolz is not installed. Cannot write bcolz array.")
    carray = bcolz.carray(arr, rootdir=path, cparams=_blosc_params, mode="w")
    carray.flush()

def write_tiledb_impl(arr, path, overwrite=True): # Renamed to avoid conflict
    if not tiledb:
        raise ImportError("tiledb is not installed. Cannot write tiledb array.")
    
    # This is a simplified implementation based on genomelake's tiledb_array.py
    # It might need adjustments based on the exact version and usage of tiledb.
    if os.path.exists(path) and os.path.isdir(path) and overwrite:
        import shutil
        shutil.rmtree(path)

    if os.path.exists(path):
        raise FileExistsError("Output path {} already exists".format(path))

    ctx = tiledb.Ctx()
    
    GENOME_DOMAIN_NAME = "genome_coord"
    SECONDARY_DOMAIN_NAME = "signal_coord"
    GENOME_VALUE_NAME = "v"
    DEFAULT_GENOME_TILE_EXTENT = 9000

    n = arr.shape[0]
    n_tile_extent = min(DEFAULT_GENOME_TILE_EXTENT, n)
    d1 = tiledb.Dim(ctx, GENOME_DOMAIN_NAME, domain=(0, n - 1), tile=n_tile_extent, dtype="uint32")

    if arr.ndim == 1:
        domain = tiledb.Domain(ctx, d1)
    elif arr.ndim == 2:
        m = arr.shape[1]
        d2 = tiledb.Dim(ctx, SECONDARY_DOMAIN_NAME, domain=(0, m - 1), tile=m, dtype="uint32")
        domain = tiledb.Domain(ctx, d1, d2)
    else:
        raise ValueError("tiledb backend only supports 1D or 2D arrays")

    v_attr = tiledb.Attr(ctx, GENOME_VALUE_NAME, compressor=("blosc-lz", -1), dtype="float32")
    schema = tiledb.ArraySchema(ctx, domain=domain, attrs=(v_attr,), cell_order="row-major", tile_order="row-major")
    tiledb.DenseArray.create(path, schema)

    values = arr.astype(np.float32)
    with tiledb.DenseArray(ctx, path, mode="w") as A:
        A[:] = {GENOME_VALUE_NAME: values}


_array_writer = {
    "numpy": write_numpy,
    "bcolz": write_bcolz,
    "tiledb": write_tiledb_impl,
}

def load_tiledb_impl(path): # Renamed to avoid conflict
    if not tiledb:
        raise ImportError("tiledb is not installed. Cannot load tiledb array.")

    class TDBDenseArray:
        def __init__(self, array_path):
            self.ctx = tiledb.Ctx()
            self._arr = tiledb.DenseArray(self.ctx, array_path, mode="r")
            self.GENOME_VALUE_NAME = "v" # As defined in write_tiledb_impl

        def __getitem__(self, key):
            return self._arr[key][self.GENOME_VALUE_NAME]

        @property
        def shape(self):
            return self._arr.shape

        @property
        def ndim(self):
            return self._arr.ndim
            
    return TDBDenseArray(path)

def load_directory(base_dir, in_memory=False):
    metadata_path = os.path.join(base_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {base_dir}")
        
    with open(metadata_path, "r") as fp:
        metadata = json.load(fp)

    array_type = metadata.get("type")
    file_shapes = metadata.get("file_shapes")

    if not array_type or not file_shapes:
        raise ValueError("metadata.json is missing 'type' or 'file_shapes'")

    data = {}
    if array_type == "array_numpy":
        mmap_mode = None if in_memory else "r"
        for chrom in file_shapes:
            data[chrom] = np.load(
                "{}.npy".format(os.path.join(base_dir, chrom)), mmap_mode=mmap_mode
            )
    elif array_type == "array_bcolz":
        if not bcolz:
            raise ImportError("bcolz is not installed. Cannot load bcolz array.")
        for chrom in file_shapes:
            data[chrom] = bcolz.open(os.path.join(base_dir, chrom), mode="r")
        if in_memory:
            data = {k: data[k].copy() for k in data.keys()}
    elif array_type == "array_tiledb":
        if not tiledb:
            raise ImportError("tiledb is not installed. Cannot load tiledb array.")
        for chrom in file_shapes:
            data[chrom] = load_tiledb_impl(os.path.join(base_dir, chrom))
    else:
        raise ValueError(f"Unsupported array type: {array_type}. Can only load from array_numpy, array_bcolz, and array_tiledb")

    for chrom, shape_tuple in six.iteritems(file_shapes):
        if data[chrom].shape != tuple(shape_tuple): # Ensure shape is a tuple
            raise ValueError(
                "Inconsistent shape found in metadata file: "
                "{} - {} vs {}".format(chrom, tuple(shape_tuple), data[chrom].shape)
            )
    return data

# Functions to extract data to file (similar to genomelake.backend)
# These would be used to pre-process data into the array formats.
def extract_fasta_to_file(fasta_path, output_dir, mode="bcolz", overwrite=False):
    from pysam import FastaFile # Moved import here
    if mode not in _array_writer:
        raise ValueError(f"Unsupported mode: {mode}")

    makedirs(output_dir, exist_ok=overwrite)
    fasta_file = FastaFile(fasta_path)
    file_shapes = {}
    for chrom, size in zip(fasta_file.references, fasta_file.lengths):
        data = np.zeros((size, NUM_SEQ_CHARS), dtype=np.float32)
        seq = fasta_file.fetch(chrom)
        one_hot_encode_sequence(seq, data) # Assuming this function is available
        file_shapes[chrom] = data.shape
        _array_writer[mode](data, os.path.join(output_dir, chrom))

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(
            {
                "file_shapes": file_shapes,
                "type": f"array_{mode}",
                "source": fasta_path,
            },
            fp,
        )

def extract_bigwig_to_file(
    bigwig_path,
    output_dir,
    mode="bcolz",
    dtype=np.float32,
    overwrite=False,
    nan_as_zero=True,
):
    import pyBigWig # Moved import here
    if mode not in _array_writer:
        raise ValueError(f"Unsupported mode: {mode}")

    makedirs(output_dir, exist_ok=overwrite)
    bw = pyBigWig.open(bigwig_path)
    chrom_sizes = bw.chroms()
    file_shapes = {}
    for chrom, size in six.iteritems(chrom_sizes):
        # Ensure size is not None and is an integer
        if size is None:
            print(f"Warning: Chromosome {chrom} has no defined size in {bigwig_path}. Skipping.")
            continue
        try:
            size = int(size)
        except ValueError:
            print(f"Warning: Chromosome {chrom} has non-integer size '{size}' in {bigwig_path}. Skipping.")
            continue

        values = bw.values(chrom, 0, size)
        if values is None: # pyBigWig can return None if no values
             data = np.zeros(size, dtype=np.float32)
        else:
            data = np.array(values, dtype=np.float32) # Ensure it's an array

        if data.shape[0] != size: # Check if fetched data matches expected size
            # This can happen if the chromosome is in the header but has no actual values
            # Or if bw.values doesn't return a full array for some reason.
            # Fill with zeros or handle as an error, depending on desired behavior.
            # For now, creating an array of zeros of the expected size.
            print(f"Warning: Fetched data size {data.shape[0]} for chrom {chrom} (expected {size}) doesn't match. Using zeros.")
            data = np.zeros(size, dtype=np.float32)


        if nan_as_zero:
            nan_to_zero(data) # Assuming this function is available
        _array_writer[mode](data.astype(dtype), os.path.join(output_dir, chrom))
        file_shapes[chrom] = data.shape
    bw.close()

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(
            {
                "file_shapes": file_shapes,
                "type": f"array_{mode}",
                "source": bigwig_path,
            },
            fp,
        )
