# Collection Management

Build, discover, filter, and manage Rasteret collections.

Rasteret has two CLI groups: `rasteret cache` manages local Collections
(build, import, inspect, delete); `rasteret datasets` manages the
dataset catalog (list, search, build from registry).

This page focuses on **Collections** and the local cache lifecycle.
For dataset catalog usage, see
[Dataset Catalog](dataset-catalog.md).

If you are new, start with: `cache build` -> `cache list` -> `cache info`.

## CLI

### Build from STAC

```bash
rasteret cache build bangalore \
  --stac-api https://earth-search.aws.element84.com/v1 \
  --collection sentinel-2-l2a \
  --bbox 77.55,13.01,77.58,13.08 \
  --date-range 2024-01-01,2024-06-30
```

### Import from a Parquet file

Works with any Parquet that has COG URLs: Source Cooperative exports,
STAC GeoParquet, or your own catalog. See [Build from Parquet](build-from-parquet.md) for details.

```bash
# Source Cooperative (public, no credentials needed)
AWS_NO_SIGN_REQUEST=YES rasteret cache import maxar-opendata \
  --record-table s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet

# Any Parquet with required columns (id, datetime, geometry, assets)
rasteret cache import my-collection \
  --record-table s3://my-bucket/stac-items.parquet
```

### List cached collections

```bash
rasteret cache list
```

### Inspect a collection

```bash
rasteret cache info bangalore
```

### Delete a collection

```bash
rasteret cache delete bangalore
```

All CLI commands accept `--workspace-dir` (default: `~/rasteret_workspace`)
and `--json` for machine-readable output.

### Dataset catalog (optional)

```bash
# List built-in catalog entries
rasteret datasets list

# Show one catalog entry
rasteret datasets info earthsearch/sentinel-2-l2a

# Build from a catalog entry
rasteret datasets build earthsearch/sentinel-2-l2a bangalore \
  --bbox 77.55,13.01,77.58,13.08 \
  --date-range 2024-01-01,2024-06-30
```

See [Dataset Catalog](dataset-catalog.md) for `register-local`,
`export-local`, and `unregister-local`.

### Naming and registry behavior

- STAC builds use standardized names: `{custom}_{date-range}_{source-prefix}`.
- `custom` is normalised for filesystem safety (underscores become dashes).
- `build_from_table()` and CLI `cache import` write collections as `{name}_records`.
- Local Collections built/imported in a workspace are cached locally;
  they are not auto-added to the dataset catalog.
- Dataset catalog workflows are documented in
  [Dataset Catalog](dataset-catalog.md).

To make a local Collection appear in `rasteret datasets list`, see
[Dataset Catalog](dataset-catalog.md) for registration workflows.

---

## Python API

### Build from STAC

See [`build_from_stac()`](../reference/rasteret.md) API reference.

```python
import rasteret

collection = rasteret.build_from_stac(
    name="bangalore",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection="sentinel-2-l2a",
    bbox=(77.55, 13.01, 77.58, 13.08),
    date_range=("2024-01-01", "2024-06-30"),
)
```

### Build from Parquet

See [`build_from_table()`](../reference/rasteret.md) and the full
[Build from Parquet](build-from-parquet.md) guide.

```python
collection = rasteret.build_from_table(
    "s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet",
    name="maxar-opendata",
)
```

When `name` is provided, the collection is cached to
`~/rasteret_workspace/{name}_records/` -- same behavior as `build_from_stac()`.

### Export and reload

See [`export()`](../reference/core/collection.md) and [`load()`](../reference/rasteret.md).

```python
collection.export("./my_collection")
reloaded = rasteret.load("./my_collection")
```

### Discover cached collections

See [`Collection.list_collections()`](../reference/core/collection.md).

```python
from pathlib import Path
from rasteret import Collection

for c in Collection.list_collections(Path.home() / "rasteret_workspace"):
    print(f"  {c['name']} ({c['kind']}), {c['size']} rows")
```

### Filter with [`subset()`](../reference/core/collection.md)

Criteria are combined with AND:

```python
filtered = collection.subset(
    cloud_cover_lt=15,
    date_range=("2024-03-01", "2024-04-30"),
    bbox=(77.55, 13.03, 77.57, 13.06),
    split="train",
)
```

### Arrow expression filtering

For arbitrary filters, use [`where()`](../reference/core/collection.md) with `pyarrow.dataset` expressions:

```python
import pyarrow.dataset as ds

clear = collection.where(ds.field("eo:cloud_cover") < 5.0)
```

---

## Add custom columns

The Parquet schema is extensible. Common additions: splits, labels,
quality flags.

### Add a train/val/test split

See [ML Training with Splits](ml-training-splits.md#2-assign-splits) for the
full workflow. Once the `split` column exists, filter with:

```python
train = collection.subset(split="train")
```

### Add a label column

```python
labels = pa.array([0, 1, 0, 1], type=pa.int32())  # one per record
table = table.append_column("label", labels)
```

Then use `label_field="label"` in [`to_torchgeo_dataset()`](../reference/integrations/torchgeo.md) to include
labels in TorchGeo samples.

---

## Register a custom cloud config

See [`CloudConfig`](../reference/cloud.md) API reference.

```python
from rasteret import CloudConfig

CloudConfig.register(
    "my-private-collection",
    CloudConfig(
        provider="aws",
        requester_pays=True,
        region="eu-central-1",
        url_patterns={
            "https://my-cdn.example.com/": "s3://my-private-bucket/",
        },
    ),
)
```

## Register a custom band mapping

For catalog datasets (`build()`), band mappings are handled automatically
via the `DatasetDescriptor.band_map` field - you never need to touch
`BandRegistry` directly.

For custom STAC APIs via `build_from_stac()`, **you choose the band keys**.
`band_map` maps those band keys to STAC asset keys. For example, if you want
to use short codes like `B04`, provide `band_map={"B04": "red", ...}`.

If you prefer to keep your code generic (and reuse the same band keys across
catalog + BYO datasets), register a mapping:

```python
from rasteret.constants import BandRegistry

BandRegistry.register(
    "my-private-collection",
    {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir"},
)
# Now you can use bands=["B04"] instead of bands=["red"]
```

For multi-band GeoTIFF assets where multiple logical bands live in one file
(e.g. NAIP `image` contains R/G/B/NIR), you must also provide
`band_index_map={"R": 0, "G": 1, "B": 2, "NIR": 3}` so Rasteret can select the
correct sample plane.

---

## Advanced: Catalog registration

If you want a local Collection to appear in `rasteret datasets list`,
register it as a catalog entry. This is optional; `export()` / `load()`
work without registration.

### Register a local Collection

```python
rasteret.register_local("local/my_collection", "./my_collection")
```

### Runtime registration (process-local)

```python
from rasteret.catalog import DatasetDescriptor

rasteret.register(
        DatasetDescriptor(
            id="acme/field-survey-2024",
            name="ACME Field Survey",
            geoparquet_uri="/path/to/collection_parquet",
            description="Drone mosaics for 2024 survey.",
            band_map={"R": "red", "G": "green", "B": "blue"},
        )
    )
```

See [Dataset Catalog](dataset-catalog.md) for full catalog workflows
including CLI commands and persistence.

The full runnable script is at `examples/collection_management.py`.
