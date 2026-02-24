# rasteret.integrations.torchgeo

TorchGeo `GeoDataset` adapter for Rasteret collections.

`RasteretGeoDataset` wraps a Rasteret `Collection` as a standard TorchGeo
`GeoDataset`. It fetches COG tiles on-the-fly via async HTTP range reads and
returns samples as `{"image": Tensor, "bounds": Tensor, "transform": Tensor}`.
Compatible with all
TorchGeo samplers (`RandomGeoSampler`, `GridGeoSampler`, etc.), collation
helpers, and transforms.

This adapter provides **pipeline-level interop** (a TorchGeo dataset object).
It does not replace TorchGeo's rasterio/GDAL-backed `RasterDataset` backend.

## Typical usage

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)

sampler = RandomGeoSampler(dataset, size=256, length=100)
loader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
```

## Output contract

- Keys always include `bounds` and `transform`.
- If `is_image=True` (default), samples include `image: Tensor` with shape `[C, H, W]` (or `[T, C, H, W]` when `time_series=True`).
- If `is_image=False`, samples include `mask: Tensor` and follow TorchGeo `RasterDataset` conventions:
  - Single-scene: `[H, W]` when `C == 1` (channel dimension squeezed).
  - Time series: `[T, H, W]` when `C == 1`.

Rasteret's low-level read APIs return a `valid_mask` for ML-safe workflows, but it
is intentionally **not** included in TorchGeo samples by default to preserve
TorchGeo dataset composition behavior (`dataset1 & dataset2`, `dataset1 | dataset2`).

## Notes / limitations

- When `chip_size` is set, Rasteret guarantees fixed chip output shape even when floating point bounds would otherwise cause off-by-one rounding.
- When `time_series=False` and the requested slice overlaps multiple records, Rasteret selects the first record and logs a warning (it does not mosaic/merge overlapping scenes in the adapter).
- Rasteret requires all requested bands to share the same resolution for TorchGeo sampling. To opt into resampling bands onto a common grid, pass `allow_resample=True` to `Collection.to_torchgeo_dataset(...)`.

For train/val/test splits, see [ML Training with Splits](../../how-to/ml-training-splits.md).

::: rasteret.integrations.torchgeo.RasteretGeoDataset
    options:
      members:
        - __init__
        - __getitem__
        - close
      show_source: true
