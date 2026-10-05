#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --job-name="uv-test"
#SBATCH --output=/home/paulverhoeven/logs/uv-test-%j.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#paulusboskabouter/uv-package:latest"

#source /data/temporary/paul/.virchow
export UV_PROJECT_ENVIRONMENT=/data/temporary/paul/venv
export UV_CACHE_DIR=/data/temporary/paul/uv-cache
export UV_LINK_MODE=copy

cd /data/temporary/paul/kidney-anomaly-detection   # contains pyproject.toml + uv.lock

uv sync --frozen
uv run --frozen python -c "import torch, openslide; print(torch.cuda.is_available(), openslide.__version__)" >> ./test.txt
