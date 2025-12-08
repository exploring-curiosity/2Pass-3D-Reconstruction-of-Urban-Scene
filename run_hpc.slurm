#!/bin/bash
#SBATCH --job-name=urban_3d_recon
#SBATCH --account=ece_gy_9193-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%j_recon.out
#SBATCH --error=%j_recon.err
#SBATCH --requeue

# Configuration
OVERLAY_FILE="overlay-25GB-500K.ext3"
SINGULARITY_FILE="cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
CONDA__PATH="/ext3/miniconda3"
ENV_NAME="acv2"

echo "Job started on $(hostname) at $(date)"

# Ensure we are in the right directory
# Slurm starts in the submission directory
WORK_DIR=$(pwd)
echo "Working directory: $WORK_DIR"

# Launch Singularity
# We bind /scratch and current directory
module purge
module load singularity

singularity exec --bind /scratch --nv \
    --overlay "$WORK_DIR/$OVERLAY_FILE:ro" \
    "$WORK_DIR/$SINGULARITY_FILE" \
    /bin/bash -c "
    
    # Setup Conda
    source $CONDA_PATH/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    echo 'Environment activated: \$(which python)'
    
    # Go to project root (mounted in /scratch or similar)
    cd $WORK_DIR
    
    # Run the Pipeline
    # 1. Pipeline Pass 1 (Static) + Pass 2 (Dynamic)
    python run_pipeline.py --clean
    
    # Note: If you want to skip pass 1 or 2, verify arguments:
    # python run_pipeline.py --skip-pass1
    
    "

echo "Job finished at $(date)"
