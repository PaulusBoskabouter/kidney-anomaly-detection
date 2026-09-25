from pathlib import Path
from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig


# def main():
#     model = Model.from_preset("virchow2")
#     preprocessing = PreprocessingConfig(
#         requested_spacing_um=0.5,
#         segmentation={"method":'otsu', 'downsample':64, 'sthresh':254, 'sthresh_up':255, 'mthresh':7, 'close':4})
#     embedded = model.embed_slide("dataset/wsi/adapin/30838.svs", preprocessing=preprocessing)

#     tile_embeddings = embedded.tile_embeddings  # shape (N, 2560)
#     x, y = embedded.x, embedded.y   # shape (N,), level-0 pixels



from pathlib import Path
from slide2vec import ExecutionOptions, Model, PreprocessingConfig

def main():
    model = Model.from_preset("virchow2")
    preprocessing = PreprocessingConfig(requested_spacing_um=0.5)
    execution = ExecutionOptions(
        output_dir=str(Path("outputs/run").resolve()),
        num_preprocessing_workers=1,
    )

    embedded = model.embed_slide(
        # TODO: There exists a bug in slide2vec.runtime.tiling_pipeline where it loads the process csv like pd.read_csv(process_list_path) but if all .svs files contain a number as name (e.g. 500.svs),
        # pandas infers sample_id as int64 which downstream becomes process_df["sample_id"] == slide.sample_id, becomes 500 == "500" which is False. Causing a value error "raise ValueError(f"No process-list entry found for sample_id={slide.sample_id}")"
        "dataset/wsi/adapin/ding.svs",
        preprocessing=preprocessing,
        execution=execution,
    )

    tile_embeddings = embedded.tile_embeddings
    x, y = embedded.x, embedded.y

if __name__ == "__main__":
    main()



# TOKENS = {}

# with open("TOKENS.csv", 'r') as file:
#     for line in file:
#         line = line.split(',')
#         TOKENS[line[0]] = line[1].rstrip('\n')


# print(TOKENS)
