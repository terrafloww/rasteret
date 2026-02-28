# rasteret.integrations.torchgeo

TorchGeo `GeoDataset` adapter for Rasteret collections.

`RasteretGeoDataset` is a standard TorchGeo `GeoDataset` subclass. It
replaces the I/O backend (instead of rasterio/GDAL) while
honoring the full GeoDataset contract: `index`, `crs`, `res`,
`__getitem__(GeoSlice) -> Sample`. Compatible with all TorchGeo samplers,
collation helpers (`stack_samples`, `concat_samples`), transforms, and
dataset composition (`IntersectionDataset`, `UnionDataset`).

## Typical usage

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)

sampler = RandomGeoSampler(dataset, size=256, length=100)
loader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
```

## GeoDataset contract (what TorchGeo requires)

Rasteret honors all of these:

- **`__getitem__(GeoSlice) -> Sample`**: returns a `dict[str, Any]`
- **`index`**: GeoPandas GeoDataFrame with `IntervalIndex` named `"datetime"` and Shapely footprint geometry
- **`crs`**: set from the collection's EPSG code
- **`res`**: derived from the first record's COG metadata transform
- **Dataset composition**: `IntersectionDataset(rasteret_ds, other_ds)` and `UnionDataset` work correctly

## Sample dict keys

**Standard keys** (always present):

- `bounds`: `Tensor` of spatial bounds
- `transform`: `Tensor` of affine transform coefficients
- `image`: `Tensor` with shape `[C, H, W]` (or `[T, C, H, W]` when `time_series=True`), when `is_image=True`
- `mask`: `Tensor` with shape `[H, W]` (or `[T, H, W]`), when `is_image=False` (channel dim squeezed when `C == 1`, matching TorchGeo `RasterDataset` conventions)

**Rasteret additions** (optional, do not break interop):

- `label`: scalar or tensor label from a metadata column, when `label_field` is set. TorchGeo's collate functions handle arbitrary keys, so this passes through `stack_samples` and `concat_samples` without issue.

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
