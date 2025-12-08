#!/bin/bash
# HPC Interactive Execution Helper (Manual Steps)

# 1. SSH to Greene
ssh sr7431@greene.hpc.nyu.edu

# 2. SSH to Burst
ssh burst

# 3. Request GPU Interactive Session
# (Account proven valid on burst)
srun --account=ece_gy_9193-2025fa --partition=c12m85-a100-1 --gres=gpu:1 --time=01:00:00 --pty /bin/bash

# 4. Once inside session:
cd 2Pass-3D-Reconstruction-of-Urban-Scene
git pull origin cleanup

# 5. Run Install Script (Inside Session)
# This installs miniconda/env/deps inside the overlay
./setup_hpc.sh
# Follow instructions printed by setup_hpc.sh to `singularity exec` and verify installs.
# Or if `setup_hpc.sh` only downloads overlay, we need to manually run install steps if not done.
# The `install_hpc.slurm` logic can be run manually:

singularity exec --bind /scratch --nv --overlay overlay-25GB-500K.ext3:rw cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

# Inside container:
# bash miniconda.sh -b -p /ext3/miniconda3
# source /ext3/miniconda3/etc/profile.d/conda.sh
# conda create -n acv2 python=3.10 -y
# conda activate acv2
# pip install -r requirements.txt
# git clone https://github.com/yyfz/Pi3.git Pi3
# pip install -r Pi3/requirements.txt

# 6. Run Pipeline
python run_pipeline.py --clean
