#!/bin/bash
# HPC Setup Script for NYU Greene Cluster
# Automates the setup of Singularity overlay and environment

set -e

# Configuration (match HPC.md)
OVERLAY_SRC="/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz"
SINGULARITY_SRC="/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY_FILE="overlay-25GB-500K.ext3"
SINGULARITY_FILE="cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"

echo "================================================="
echo "  NYU Greene HPC Setup for 3D Reconstruction"
echo "================================================="

# 1. Check/Download Singularity Image
if [ -f "$SINGULARITY_FILE" ]; then
    echo "✓ Singularity image found: $SINGULARITY_FILE"
else
    echo "Downloading Singularity image..."
    # Check if we are on Greene (can scp from greene-dtn or cp if mounted)
    # The HPC.md says scp from greene-dtn.
    if [[ $(hostname) == *"greene"* || $(hostname) == *"burst"* || $(hostname) == *"log-"* ]]; then
        # Try local copy first (if /scratch/work is mounted)
        if [ -f "$SINGULARITY_SRC" ]; then
             cp "$SINGULARITY_SRC" .
        else
             # Fallback to SCP (might need password if keys not set, so warn user)
             echo "⚠️  /scratch/work not accessible directly. Trying SCP from greene-dtn..."
             scp greene-dtn:"$SINGULARITY_SRC" .
        fi
    else
        echo "❌ You must run this script on the Greene cluster (login or burst node)."
        exit 1
    fi
fi

# 2. Check/Download Overlay
if [ -f "$OVERLAY_FILE" ]; then
    echo "✓ Overlay file found: $OVERLAY_FILE"
elif [ -f "$OVERLAY_FILE.gz" ]; then
    echo "Unzipping existing overlay..."
    gunzip -v "$OVERLAY_FILE.gz"
else
    echo "Downloading and setting up overlay (25GB)..."
    if [ -f "$OVERLAY_SRC" ]; then
        cp "$OVERLAY_SRC" .
    else
        echo "⚠️  Trying SCP from greene-dtn..."
        scp greene-dtn:"$OVERLAY_SRC" .
    fi
    
    echo "Unzipping overlay (this may take a moment)..."
    gunzip -v "$(basename "$OVERLAY_SRC")"
fi

echo ""
echo "================================================="
echo "  Setup Complete! Next Steps:"
echo "================================================="
echo "1. Start an interactive GPU session to install dependencies:"
echo "   srun --account=ece_gy_9193-2025fa --partition=c12m85-a100-1 --gres=gpu:1 --time=01:00:00 --pty /bin/bash"
echo ""
echo "2. Inside the session, launch Singularity (RW mode):"
echo "   singularity exec --bind /scratch --nv --overlay $OVERLAY_FILE:rw $SINGULARITY_FILE /bin/bash"
echo ""
echo "3. Inside Singularity, install Miniconda and Environment:"
echo "   # Install Miniconda (if not done)"
echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh"
echo "   bash miniconda.sh -b -p /ext3/miniconda3"
echo "   source /ext3/miniconda3/etc/profile.d/conda.sh"
echo ""
echo "   # Create Env and Install Deps"
echo "   conda create -n acv2 python=3.10 -y"
echo "   conda activate acv2"
echo "   pip install -r requirements.txt"
echo "   # Don't forget to clone Pi3 if needed: git clone https://github.com/yyfz/Pi3.git Pi3"
echo ""
echo "4. Exit and submit jobs using 'sbatch run_hpc.slurm'"
