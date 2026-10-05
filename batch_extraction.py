#!/usr/bin/env python3
import os, shutil, subprocess, sys
from pathlib import Path

SRC = Path("/data/pa_cpgarchive/PATH/TO/SLIDES")        # adapt
MODEL = "virchow2"
BASE = Path("/data/temporary/paul")                     # adapt
OUT = BASE / "output" / MODEL
TMP = BASE / f"tmp_{os.environ.get('SLURM_JOB_ID', 'local')}"
FAILED = OUT / "failed.txt"
BATCH = 100

OUT.mkdir(parents=True, exist_ok=True)

failed_before = set(FAILED.read_text().split()) if FAILED.exists() else set()
done = {p.stem for p in OUT.glob("*.pt")}
todo = sorted(p for p in SRC.rglob("*.svs")
              if p.stem not in done and p.name not in failed_before)
print(f"{len(todo)} slides left", flush=True)

for i in range(0, len(todo), BATCH):
    batch = todo[i:i + BATCH]
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)

    for p in batch:
        shutil.copy2(p, TMP / p.name)

    r = subprocess.run(["uv", "run", "feature_extraction.py", "--virchow2",
                        "--input", str(TMP), "--output", str(OUT)])  # adapt args

    # anything in the batch without a .pt after the run counts as failed
    missing = [p.name for p in batch if not (OUT / f"{p.stem}.pt").exists()]
    if missing:
        with FAILED.open("a") as f:
            f.write("\n".join(missing) + "\n")
        print(f"batch {i // BATCH}: {len(missing)} slides produced no output "
              f"(exit code {r.returncode})", file=sys.stderr, flush=True)

    shutil.rmtree(TMP)
