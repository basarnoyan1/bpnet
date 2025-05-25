#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "argh<0.28",
    "attr",
    "related",
    "cloudpickle>=1.0.0",

    "concise @git+https://github.com/basarnoyan1/concise.git#egg=concise",
	"shapely<=1.8.5.post1",

    # PyTorch ecosystem
    "torch",
    "torchvision",
    "torchaudio",
    "captum",

    # ml
    "scikit-learn",

    # numerics
    "h5py<3",
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",

    # Plotting
    "matplotlib>=3.0.2,<3.4.0", # Keep matplotlib version constraint for now
    "plotnine", # Review if still needed with PyTorch for plotting or can be replaced
    "seaborn", # Review if still needed

    # genomics
    "pybigwig",
    "pybedtools",
    "modisco==0.5.3.0", # Review compatibility with PyTorch
    # "pyranges", # Consider if needed

    "joblib",
    "kipoi>=0.6.8", # Review compatibility
    "kipoi-utils>=0.3.0", # Review compatibility
    "kipoiseq>=0.2.2", # Review compatibility

    "papermill", # For notebook-based workflows, keep if relevant
    "jupyter_client>=6.1.2", # For Jupyter, keep if relevant
    "ipykernel", # For Jupyter, keep if relevant
    "nbconvert>=5.5.0", # For Jupyter, keep if relevant
    "vdom>=0.6", # For rich display in Jupyter, keep if relevant

    # utils
    "ipython", # For interactive work, keep if relevant
    "tqdm",

    # Remove or ensure compatibility
    "genomelake @git+https://github.com/pauldrinn/genomelake.git#egg=genomelake", # Check PyTorch compatibility
    "pysam",  # Check if still needed or can be replaced by pyfaidx or other alternatives
]

optional = [
    "comet_ml", # Experiment tracking, keep if used
    "wandb==0.8.7", # Experiment tracking, keep if used
    "fastparquet", # For Parquet file format, keep if used
    "python-snappy", # For Snappy compression, keep if used with Parquet
    "ipywidgets",  # For interactive widgets in Jupyter, keep if used
]

test_requirements = [
    "pytest>=3.3.1",
    "pytest-cov>=2.6.1",
    # "pytest-xdist", # Consider for parallel testing
    "gdown",   # download files from google drive
    "virtualenv", # For isolated test environments
]

setup(
    name="bpnet",
    version='0.0.23', # Consider updating version number for PyTorch migration
    python_requires='>=3.9',
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Ziga Avsec",
    author_email="avsec@in.tum.de",
    url="https://github.com/kundajelab/bpnet",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": test_requirements,
        "extras": optional,
    },
    license="MIT license",
    entry_points={'console_scripts': ['bpnet = bpnet.__main__:main']},
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    package_data={'bpnet': ['logging.conf']},
    include_package_data=True,
    tests_require=test_requirements
)
