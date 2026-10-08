from pathlib import Path
import os, shutil, subprocess, sys
from typing_extensions import Dict


WORK_DIR = Path(sys.argv[1]) # The directory where the code lives
REMOTE_DIR = Path(sys.argv[2])
BATCH_DIR = Path(sys.argv[3])
OUT_DIR = WORK_DIR/'features'

BATCH_SIZE = 50
MODELS = ['virchow2', 'gigapath', 'conchv15', 'h-optimus-1']
TOKENS = {}

# Used for failed attempts (so we don't get stuck in the while True loop.)
ATTEMPTS = {}
MAX_ATTEMPTS = 2


def list_slides() -> list[Path]:
    """
    Short function for fetching all the dataset files. Used for checking completion
    Returns
    ----------
    list : slides
        list containing ALL the Paths of the dataset.
    """
    slides = []
    for folder in REMOTE_DIR.iterdir():
        slide_location = folder / 'Kidney'
        if slide_location.exists():
            slides.extend(slide_location.glob("*.svs"))
    return slides


def check_completion() -> None:
    """
    Short function for checking if all the models have fully processed all dataset files.
    """
    slides = list_slides()
    todo = {}
    for model in MODELS:
        done = {p.stem for p in (OUT_DIR / model / 'tile_embeddings').glob("*.pt")}
        todo[model] = [s.name for s in slides if s.stem not in done]

    print(f"{len(slides)} slides in total")
    for model, files in todo.items():
        if len(files) == 0:
             print(f"{model} is done!")
        else:
            print(f"{model} has to finish the following file(s) ({len(files)}):")
            for file in files:
                print("\t"+file)
            print('\t'+'-'*9)

def fetch_processed() -> Dict:
    """
    Short function to get a dictionary for keeping track of which WSI's have yet-to-be-embedded.

    Returns
    ----------
    dict : processed
        Dictionary containing file: #number of models that have processed it.
    """
    processed = {}
    for model in MODELS:
        for file in (OUT_DIR/model/'tile_embeddings').glob('*.pt'):
            if processed.get(file.stem) is None:
                processed[file.stem] = 1
            else:
                processed[file.stem] += 1
    return processed


def fetch_batch(processed:dict) -> list[Path]:
    """
    Short function to get a list of all the yet-to-be-embedded WSI files.

    Parameters
    ----------
    processed : dict
        Dictionary containing file: number of models that have processed it, used for keeping track on whether a file is already completed.

    Returns
    ----------
    list : current_batch
        a list of file Paths that have yet to be embedded of size BATCH_SIZE
    """
    current_batch = []
    for folder in REMOTE_DIR.iterdir():
        slide_location = folder / 'Kidney'
        if not slide_location.exists():
            continue
        for file in slide_location.glob("*.svs"):
            if ATTEMPTS.get(file.stem, 0) >= MAX_ATTEMPTS:
                continue                              # given up, keep looking
            if processed.get(file.stem, 0) < len(MODELS):
                current_batch.append(file)
                if len(current_batch) >= BATCH_SIZE:
                    return current_batch
    return current_batch




def main():
    """
    Main loop, see comments throughout code.
    But tl;dr is fetches batches and runs feature_extraction.py as sub process.
    """
    while True:

        # 1. fetch all the batches
        processed = fetch_processed()
        current_batch = fetch_batch(processed)

        if not current_batch: # If we don't have any batches left we must be done!
            print("WORKS DONE! :-)")
            sys.exit(0)

        # 2. Copy the files for this batch to the local node
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        for file in current_batch:
            shutil.copy2(file, BATCH_DIR / file.name)


        # 3. Fetch the HF tokens
        with open("TOKENS.csv", 'r') as file:
            for line in file:
                line = line.split(',')
                TOKENS[line[0]] = line[1].rstrip('\n')

        # 4. For each model; start the sub-process of feature extraction
        for model in MODELS:
            env = os.environ.copy()
            env["HF_TOKEN"] = TOKENS[model]

            # Start feature extraction as a subprocess (that way vram gets cleared properly when the code is done)
            try:
                subprocess.run([sys.executable, WORK_DIR / 'feature_extraction.py', model, BATCH_DIR.resolve(), (OUT_DIR/model).resolve()], env=env, check=True)
            except subprocess.CalledProcessError as oopsie:
                print(f"{model} failed on this batch", flush=True)
                print(oopsie)

        # 5. keep track of whether slide files failed (exceeding MAX_ATTEMPTS cull the file (which will need manual checking at some point.))
        for slide in current_batch:
            ATTEMPTS[slide.stem] = ATTEMPTS.get(slide.stem, 0) + 1

        # 6. Clear the local folder to save space.
        shutil.rmtree(BATCH_DIR)

if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4] == "check":
        check_completion()
    else:
        main()
