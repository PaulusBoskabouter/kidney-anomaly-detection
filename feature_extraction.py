#!/usr/bin/env python3
import sys
from pathlib import Path
from slide2vec import ExecutionOptions, Model, PreprocessingConfig
import slide2vec.runtime.tiling_pipeline as tiling_pipeline
import transformers.modeling_utils as _modeling_utils
import pandas as pd


MODEL = sys.argv[1]
INPUT_DIR = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])

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



def get_unprocessed_slides() -> list:
    """ Short function to get a list of all the yet-to-be-embedded WSI files.
    Returns
    -------
    list
        a list of file paths that have yet to be embedded.
    """
    processed = [p.stem for p in (OUT_DIR / 'tile_embeddings').glob("*.pt")]
    slides = [str(file.resolve()) for file in INPUT_DIR.rglob("*.svs") if file.stem not in processed]
    return slides


def main() -> None:

    # 1. Get the list of slides that need processing
    slides = get_unprocessed_slides()
    if not slides:
        print(f"No unprocessed slides for {MODEL}, skipping.")
        return

    # 2. Set the (pre-)processing parameters
    model = Model.from_preset(MODEL)
    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        tolerance=0.07,
        overlap=0.0,
        masks={
            'min_coverage':{"tissue": 0.3}
        },
        preview={
            "save_mask_preview": False,
            "save_tiling_preview": False
        },
        segmentation={
            'method':'hsv',
            'downsample':64,
            'sthresh_up':2
        }
    )
    execution = ExecutionOptions(output_dir=OUT_DIR) # num_workers_per_gpu=1, num_preprocessing_workers= 16

    # 3. Poor GPU
    model.embed_slides(
        slides,
        preprocessing=preprocessing,
        execution=execution,
    )



if __name__ == "__main__":
    main()
