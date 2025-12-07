#!/bin/bash
#SBATCH --job-name=install_recon
#SBATCH --account=users
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%j_install.out
#SBATCH --error=%j_install.err

# Configuration
OVERLAY_FILE="overlay-25GB-500K.ext3"
SINGULARITY_FILE="cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
WORK_DIR=$(pwd)

echo "Installation job started on $(hostname)"

# We need to run in RW mode to install things
singularity exec --bind /scratch --nv \
    --overlay "$WORK_DIR/$OVERLAY_FILE:rw" \
    "$WORK_DIR/$SINGULARITY_FILE" \
    /bin/bash -c "
    
    set -e
    
    # 1. Install Miniconda if missing
    if [ ! -d /ext3/miniconda3 ]; then
        echo 'Installing Miniconda...'
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p /ext3/miniconda3
        rm miniconda.sh
    else
        echo 'Miniconda already installed.'
    fi
    
    source /ext3/miniconda3/etc/profile.d/conda.sh
    
    # 2. Create Environment if missing
    if ! conda env list | grep -q 'acv2'; then
        echo 'Creating acv2 environment...'
        conda create -n acv2 python=3.10 -y
    fi
    
    conda activate acv2
    
    # 3. Install Dependencies
    echo 'Installing dependencies...'
    cd $WORK_DIR
    pip install -r requirements.txt
    
    # 4. Clone Pi3 if missing
    if [ ! -d 'Pi3' ] || [ -z \"$(ls -A Pi3)\" ]; then
        echo 'Cloning Pi3...'
        rm -rf Pi3
        git clone https://github.com/yyfz/Pi3.git Pi3
        pip install -r Pi3/requirements.txt
    fi
    
    echo 'Installation Complete!'
    "
