#!/usr/bin/env python3
import sys
from pathlib import Path
from slide2vec import ExecutionOptions, Model, PreprocessingConfig
import slide2vec.runtime.tiling_pipeline as tiling_pipeline
import transformers.modeling_utils as _modeling_utils
import pandas as pd


# This is a short work-around to get conchv15 to work
if not hasattr(_modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
    _modeling_utils.PreTrainedModel.all_tied_weights_keys = {}


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

def get_unprocessed_slides(output_dir:str, dataset_dir:str="./dataset") -> list:
    """ Short function to get a list of all the yet-to-be-embedded WSI files.
    Parameters
    ----------
    output_dir : str
        The output directory of the embedding
    dataset_dir : str
        The location of the dataset

    Returns
    -------
    list
        a list of file paths that have yet to be embedded.
    """
    output_dir = Path(output_dir)
    processed = {p.stem for p in output_dir.glob("*.pt")}

    slides = []
    for folder in Path(dataset_dir).iterdir():
        slides += [
            str(file.resolve())
            for file in folder.rglob("*.svs")
            if file.stem not in processed
        ]
    return slides


def main():
    model_name = sys.argv[1]
    slides = get_unprocessed_slides(output_dir=f"embeddings/{model_name}/tile_embeddings")
    if not slides:
        print(f"No unprocessed slides for {model_name}, skipping.")
        return

    model = Model.from_preset(model_name)
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
    execution = ExecutionOptions(output_dir=str(Path(f"embeddings/{model_name}").resolve()))

    data = model.embed_slides(
        slides,
        preprocessing=preprocessing,
        execution=execution,
    )

_orig_load_tiling_process_df = tiling_pipeline.load_tiling_process_df
tiling_pipeline.load_tiling_process_df = _load_tiling_process_df_str_id

if __name__ == "__main__":
    main()
