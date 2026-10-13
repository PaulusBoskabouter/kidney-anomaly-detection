import torch
from pathlib import Path
import os, shutil, subprocess, sys
from hs2p import SlideSpec, TilingConfig, SegmentationConfig, PreviewConfig, tile_slides
import pandas as pd
from datetime import datetime


WORK_DIR = Path(sys.argv[1]) # The directory where the code lives
REMOTE_DIR = Path(sys.argv[2]) # The directory where the dataset lives
BATCH_DIR = Path(sys.argv[3]) # The directory where the dataset (batch) gets pulled to
MODEL = sys.argv[4] # Model name
ID = int(sys.argv[5]) # either '0' or '1'
FILES = pd.read_csv(WORK_DIR / 'assigned.csv')

BATCH_SIZE = 600
MODELS =  ['h-optimus-1', 'conchv15', 'gigapath', 'virchow2']
TOKENS = {}
assert MODEL in MODELS or MODEL=='check', f"{MODEL} doesn't exist, try {MODELS} or 'check'. {type(MODEL)}"

# Used for failed attempts (so we don't get stuck in the while True loop.)
ATTEMPTS = {}
MAX_ATTEMPTS = 3



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


def fetch_batch() -> list[Path]:
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
            # Check whether output already exists:
            if (WORK_DIR/ 'embeddings' / MODEL / 'tile_embeddings'/ f'{file.stem}.pt').exists():
                continue

            # Check if file is assigned to this jobs 'ID'.
            a = FILES.loc[
                    (FILES["file"] == file.name) & (FILES["assigned"] == ID),
                    "assigned"
                ]
            if a.empty:
                continue

            # File has had too many attempts, keep looking
            if ATTEMPTS.get(file.stem, 0) >= MAX_ATTEMPTS:
                # Print this debug statement only once
                if ATTEMPTS.get(file.stem) == MAX_ATTEMPTS:
                    time = datetime.now()
                    print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\t{file.name} has too many attempts, skipping...")
                    ATTEMPTS[file.stem] += 1
                continue

            # Else populate the current batch with this file.
            current_batch.append(file)


            # And of course limit our batch to the batch size.
            if len(current_batch) == BATCH_SIZE:
                return current_batch

    return current_batch


def main():
    """
    Main loop, see comments throughout code.
    But tl;dr is fetches batches and runs feature_extraction.py as sub process.
    """
    time = datetime.now()
    print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tStarting the features extraction on {'GPU' if torch.cuda.is_available() else 'CPU'}.")
    while True:
        # 1. fetch a batch
        current_batch = fetch_batch()


        if not current_batch: # If we don't have any batches left we must be done!
            time = datetime.now()
            print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tWorks done!")
            sys.exit(0)

        # 2. Copy the files for this batch to the local node
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        time = datetime.now()
        print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tCopying {len(current_batch)} file(s) over...")
        for file in current_batch:
            shutil.copy2(file, BATCH_DIR / file.name)
        time = datetime.now()
        print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tAll files copied!")


        # 3. Fetch the HF tokens
        with open("TOKENS.csv", 'r') as file:
            for line in file:
                line = line.split(',')
                TOKENS[line[0]] = line[1].rstrip('\n')



        # 4. Save the mask for later inspection.
        slides = [SlideSpec(sample_id=file.stem, image_path=file) for file in BATCH_DIR.rglob("*.svs") if file.stem not in [prev.stem for prev in (WORK_DIR / 'segmentation' / 'preview'/ 'mask').glob("*.jpg")]]
        if slides:
            time = datetime.now()
            print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tGenerating masks for slides...")
            tile_slides(
                slides,
                tiling=TilingConfig(
                    requested_spacing_um=0.5,
                    requested_tile_size_px=224,
                    tolerance=0.0,
                    overlap=0.0,
                    min_coverage={"tissue": 0.3},
                ),
                segmentation=SegmentationConfig(method="hsv", downsample=64, sthresh_up=2),
                preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=False, downsample=64),
                output_dir=Path(WORK_DIR/'segmentation'),
            )

        else:
            time = datetime.now()
            print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tNo masks needed for current slides.")


        # 5. Start the sub-process of feature extraction
        env = os.environ.copy()
        env["HF_TOKEN"] = TOKENS[MODEL]

        # 6. Start feature extraction as a subprocess (that way vram gets cleared properly when the code is done)
        time = datetime.now()
        print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tStarting embedding of {len(current_batch)} slides.")
        try:
            subprocess.run([sys.executable, WORK_DIR / 'feature_extraction' / 'feature_extraction.py', MODEL, BATCH_DIR.resolve(), WORK_DIR.resolve()], env=env, check=True)
        except subprocess.CalledProcessError as oopsie: # this should no longer be needed, TODO CHECK!
            print('='*10)
            print(oopsie)
            print('='*10)


        # 7. keep track of whether slide files failed (exceeding MAX_ATTEMPTS cull the file (which will need manual checking at some point.))
        for slide in current_batch:
            ATTEMPTS[slide.stem] = ATTEMPTS.get(slide.stem, 0) + 1


        # 8. Simple statement for keeping track of the progress:
        time = datetime.now()
        print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tFinished batch, completed {len(list(BATCH_DIR.rglob('*.svs')))}.")

        # 9. Clear the local folder to save space.
        shutil.rmtree(BATCH_DIR, ignore_errors=True)

if __name__ == "__main__":
    shutil.rmtree(BATCH_DIR, ignore_errors=True) # Make sure it's already empty
    main()
