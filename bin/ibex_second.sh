#!/bin/bash 

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

# creates a separate directory for each job
JOB_NAME=SAT4
mkdir -p "$PROJECT_DIR"/results/"$JOB_NAME"

# launch the training job
CPUS_PER_GPU=6
sbatch --job-name "$JOB_NAME" --cpus-per-gpu $CPUS_PER_GPU \
    bin/train.sbatch referit3d/scripts/train_referit3d.py -scannet-file referit3d/data/scannet/save_dir/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl -referit3D-file referit3d/data/language/nr3d/csv/nr3d.csv --log-dir results/"$JOB_NAME" --n-workers $CPUS_PER_GPU --patience 100 --max-train-epochs 100 --init-lr 1e-4 --augment-with-sr3d referit3d/data/language/sr3d/csv/sr3d_train.csv  --unit-sphere-norm True --feat2d clsvecROI --context_2d unaligned --mmt_mask train2d --warmup --batch-size 32  --transformer --model mmt_referIt3DNet --margin 1 --ce 1 --triplet 0 --contrastive 1 --ce2 1 --triplet2 0 --contrastive2 1

