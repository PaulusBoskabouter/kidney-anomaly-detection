#!/usr/bin/env python3
import sys
from pathlib import Path
from slide2vec import ExecutionOptions, Model, PreprocessingConfig
import slide2vec.runtime.tiling_pipeline as tiling_pipeline
import transformers.modeling_utils as _modeling_utils
import pandas as pd
from datetime import datetime
import torch


MODEL = sys.argv[1]
INPUT_DIR = Path(sys.argv[2])
WORK_DIR = Path(sys.argv[3])
OUT_DIR = WORK_DIR / 'embeddings' / MODEL

SLIDES = [file for file in INPUT_DIR.rglob("*.svs")]

def _load_tiling_process_df_str_id(path) -> pd.DataFrame:
    """ Work around a slide2vec bug: process_list.csv's sample_id column gets read back as int64 when filenames are purely numeric,
    but slide.sample_id is always a str (Path(slide).stem), so the equality check in prepare_tiled_slides() never matches and
    raises "No process-list entry found" even when tiling succeeded. Therefore this short function forces the column back to str type.
    Parameters
    ----------
    path : str | Path
        Same parameter as used in original function called  `process_list_path`

    Returns
    -------
    DataFrame
        DataFrame of pre-processed WSI's.
    """
    df = _orig_load_tiling_process_df(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df

# This is a short work-around to get conchv15 to work
if not hasattr(_modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
    _modeling_utils.PreTrainedModel.all_tied_weights_keys = {}

_orig_load_tiling_process_df = tiling_pipeline.load_tiling_process_df
tiling_pipeline.load_tiling_process_df = _load_tiling_process_df_str_id


# def main() -> None:
#     # 1. Get the list of slides that need processing (and catch the slides that didn't pre-process well and skip them.)
#     if not SLIDES:
#         time = datetime.now()
#         print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tNo unprocessed slides in this batch.")
#         return

#     # 2. Set the (pre-)processing parameters
#     model = Model.from_preset(MODEL)

#     preprocessing = PreprocessingConfig(
#         requested_spacing_um=0.5,
#         tolerance=0.0,
#         overlap=0.0,
#         masks={
#             'min_coverage':{"tissue": 0.3}
#         },
#         preview={
#             "save_mask_preview": False,
#             "save_tiling_preview": True
#         },
#         segmentation={
#             'method':'hsv',
#             'downsample':64,
#             'sthresh_up':2
#         }
#     )
#     execution = ExecutionOptions(output_dir=OUT_DIR)

#     # 3. Poor GPU
#     time = datetime.now()
#     print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tEmbedding {len(SLIDES)} slides using {MODEL}.")
#     while SLIDES:
#         try:
#             model.embed_slides(SLIDES, preprocessing=preprocessing, execution=execution)
#         except Exception as e:
#             time = datetime.now()
#             print("DEBUG:", SLIDES)
#             print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}):{time.minute:02}):\tSKIPPING bad slide {SLIDES[0]}: ({e!r})")

#             # Delete the file from the temporary folder.
#             SLIDES[0].unlink()
#             SLIDES = SLIDES[1:]
#             torch.cuda.empty_cache()

#     else:
#         time = datetime.now()
#         print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}:{time.minute:02}):\tCompleted {len(SLIDES)} slide(s)!")

# if __name__ == "__main__":
#     SLIDES = [file for file in INPUT_DIR.rglob("*.svs")]
#     main()



import re

def main() -> None:
    model = Model.from_preset(MODEL)
    preprocessing = PreprocessingConfig(
            requested_spacing_um=0.5,
            tolerance=0.0,
            overlap=0.0,
            masks={
                'min_coverage':{"tissue": 0.3}
            },
            preview={
                "save_mask_preview": False,
                "save_tiling_preview": True
            },
            segmentation={
                'method':'hsv',
                'downsample':64,
                'sthresh_up':2
            }
        )    # as before
    execution = ExecutionOptions(output_dir=OUT_DIR)
    emb_dir = OUT_DIR / "tile_embeddings"

    def pending():
        done = {p.stem for p in emb_dir.glob("*.pt")}
        return [f for f in sorted(INPUT_DIR.rglob("*.svs")) if f.stem not in done]

    while True:
        slides = pending()
        if not slides:
            break
        try:
            model.embed_slides(slides, preprocessing=preprocessing, execution=execution)
        except Exception as e:
            msg = str(e)
            named = [f for f in slides
                     if re.search(rf"(?<![\w.]){re.escape(f.stem)}(?!\w)", msg)]
            bad = named
            time = datetime.now()
            print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}):{time.minute:02}):\tSKIPPING {[f.name for f in bad]}: {e!r}")

            # Delete the bad files.
            for f in bad:
                f.unlink()
            torch.cuda.empty_cache()
            continue

        # no exception: anything still without a .pt was skipped silently (e.g. failed preprocessing)
        missing = pending()
        if missing:
            time = datetime.now()
            print(f"({time.day:2}-{time.month:2}-{time.year:4} @ {time.hour:02}):{time.minute:02}):\t Silently skipped by slide2vec: {[f.name for f in missing]}")
            for f in missing:
                f.unlink()
        break

if __name__ == "__main__":
    main()
