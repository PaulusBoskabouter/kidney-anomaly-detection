from pathlib import Path
from hs2p import SlideSpec, TilingConfig, SegmentationConfig, PreviewConfig, tile_slides


for method in ["hsv", "otsu", "threshold"]:
    slides = [SlideSpec(sample_id=method, image_path=Path("./dataset/wsi/48193.tiff"))]



# for method in ["hsv", "otsu", "threshold"]:
    tile_slides(
        slides,
        tiling=TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tolerance=0.07,
            overlap=0.0,
            min_coverage={"tissue": 0.1},
        ),
        segmentation=SegmentationConfig(method=method, downsample=64, sthresh=254, sthresh_up=255, mthresh=7, close=4),


        preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=False, downsample=64),
        output_dir=Path(f"output/stresh"),
    )
