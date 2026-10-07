#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpumem:10g
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="Feature extraction"
#SBATCH --output=/home/paulverhoeven/logs/feat_extr-%j.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology,/local_temporary:/local_temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#paulusboskabouter/uv-package:latest"
#SBATCH --requeue


# Directories
export TEMPORARY=/data/pathology/projects/paul
export DATASET=/data/pa_cpgarchive/archives/toxicology/open-tg-gates/images/
export CODE_FOLDER=${TEMPORARY}/code
export BATCH_LOCATION=/local_temporary/dataset_${SLURM_JOB_ID}
mkdir -p $BATCH_LOCATION

#UV-related stuff
export UV_PROJECT_ENVIRONMENT=/tmp/venv
export UV_CACHE_DIR=/local_temporary/paul/uv-cache
export UV_LINK_MODE=copy
export HF_HOME=/local_temporary/hf # For storing the downloaded models.
export PYTHONUNBUFFERED=1 # For debugging


cd $CODE_FOLDER  # contains pyproject.toml + uv.lock

uv sync --frozen

# batch_extraction has input parameters code_directory, remote_dir(Location of datasets on cpgarchives), batch_directory(where to (temporarily) locally store the copied data)
uv run --frozen batch_extraction.py . $DATASET $BATCH_LOCATION
