import sys
from pathlib import Path
from slide2vec import ExecutionOptions, Model, PreprocessingConfig
import slide2vec.runtime.tiling_pipeline as tiling_pipeline
import transformers.modeling_utils as _modeling_utils


# This is a short work-around to get conv15 to work
if not hasattr(_modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
    _modeling_utils.PreTrainedModel.all_tied_weights_keys = {}


# Work around a slide2vec bug: process_list.csv's sample_id column gets
# read back as int64 when filenames are purely numeric, but slide.sample_id
# is always a str (Path(slide).stem), so the equality check in
# prepare_tiled_slides() never matches and raises "No process-list entry
# found" even when tiling succeeded. Force the column back to str.
_orig_load_tiling_process_df = tiling_pipeline.load_tiling_process_df


def _load_tiling_process_df_str_id(path):
    df = _orig_load_tiling_process_df(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


tiling_pipeline.load_tiling_process_df = _load_tiling_process_df_str_id


def get_unprocessed_slides(output_dir, dataset_dir="./dataset"):
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
    if len(sys.argv) < 2:
        print("Usage: python run_single_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    # HF_TOKEN is expected to already be set in the environment by the caller

    slides = get_unprocessed_slides(output_dir=f"embeddings/{model_name}/tile_embeddings")
    slides = slides[:1]

    if not slides:
        print(f"No unprocessed slides for {model_name}, skipping.")
        return

    model = Model.from_preset(model_name)
    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        preview={"save_mask_preview": False, "save_tiling_preview": False},
    )
    execution = ExecutionOptions(output_dir=str(Path(f"embeddings/{model_name}").resolve()))

    data = model.embed_slides(
        slides,
        preprocessing=preprocessing,
        execution=execution,
    )
    print(f"Done with {model_name}, embedded {len(slides)} slide(s).")


if __name__ == "__main__":
    main()
