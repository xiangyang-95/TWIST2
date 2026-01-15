#!/bin/bash

# conda check if lerobot env is available
CONDA_ENV="lerobot"

if conda info --envs | grep -q $CONDA_ENV; then
    echo "Conda environment $CONDA_ENV found. Activating..."
    source ~/miniforge3/bin/activate $CONDA_ENV
else
    echo "Conda environment '$CONDA_ENV' not found. Creating..."
    conda create -n $CONDA_ENV python=3.10 -y
    source ~/miniforge3/bin/activate $CONDA_ENV
    conda install ffmpeg -c conda-forge -y
    pip install -r requirements.txt
fi

echo "Conda environment $CONDA_ENV is now active."