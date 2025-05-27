#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "argh>=0.28",  # updated
    "attr",
    "related",
    "cloudpickle>=2.0.0",

    # PyTorch ecosystem
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "captum>=0.6.0",

    # ML
    "scikit-learn>=1.2.0",

    # Numerics  
    "h5py>=3.7.0",
    "numpy>=1.23.0",
    "pandas>=1.5.0",
    "scipy>=1.9.0",
    "statsmodels>=0.13.0",

    # Plotting
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",

    # Genomics
    "pybigwig>=0.3.22",
    "pybedtools>=0.9.0",
    "pysam>=0.21.0",
    "joblib>=1.2.0",
    "tqdm>=4.64.0",
    "kipoi[torch]>=0.7.0",  # minimal install, avoids keras/tensorflow
    "kipoiseq>=0.7.0",      # latest version, check compatibility
    "gin-config>=0.5.0",

    # Optional: for better sequence logos
    "logomaker>=0.8.0",

    # Notebooks
    "papermill>=2.4.0",
    "jupyter_client>=7.4.0",
    "ipykernel>=6.17.0",
    "nbconvert>=7.0.0",
    "vdom>=0.7",
    "ipython>=8.0.0"
    # "modisco",  # REMOVE from main requirements, add to optional if needed
]
optional = [
    "modisco",  # Only if you need it, as it depends on TensorFlow
    "comet_ml",
    "wandb>=0.15.0",
    "fastparquet>=2023.2.0",
    "python-snappy>=0.6.1",
    "ipywidgets>=8.0.0",
    "pyarrow>=12.0.0"  # For Parquet, as an alternative to fastparquet
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
