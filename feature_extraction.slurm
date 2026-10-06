#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name="uv-test"
#SBATCH --output=/home/paulverhoeven/logs/uv-test-%j.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#paulusboskabouter/uv-package:latest"

#source /data/temporary/paul/.virchow
export CODE_FOLDER=/data/pathology/projects/paul/code
export UV_PROJECT_ENVIRONMENT=/data/pathology/projects/paul/code/venv
export UV_CACHE_DIR=/data/pathology/projects/paul/code/uv-cache
export UV_LINK_MODE=copy

cd CODE_FOLDER  # contains pyproject.toml + uv.lock

uv sync --frozen
uv run --frozen python -c "import torch, openslide; print(torch.cuda.is_available(), openslide.__version__)" >> FOLDER_PAUL/test.txt
