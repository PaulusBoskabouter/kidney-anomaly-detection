from pathlib import Path
from hs2p import SlideSpec, TilingConfig, SegmentationConfig, PreviewConfig, tile_slides

slides = []

root = Path("./dataset/wsi")
for disease in root.iterdir():
    for file in disease.rglob("*"):
        slides.append(SlideSpec(sample_id=file.name.rstrip(".svs"), image_path=file))


tile_slides(
    slides,
    tiling=TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        min_coverage={"tissue": 0.3},
    ),
    # base params:
    segmentation=SegmentationConfig(method='hsv', downsample=64, sthresh_up=2),

    preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True, downsample=64),
    output_dir=Path("segmentation"),
)
