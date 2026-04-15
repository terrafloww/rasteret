# AlphaEarth Foundation Embeddings

AlphaEarth Foundation (AEF) embeddings are 64-band foundation-model features
published as cloud-native GeoTIFFs. Rasteret provides a ready-to-read annual AEF
Rasteret Collection, so most users should load the catalog entry instead of rebuilding
it.

```python
import rasteret

collection = rasteret.load("aef/v1-annual")
collection.describe()
```

This opens Rasteret's maintained AEF collection on Source Cooperative. The AEF
Rasteret artifact is also published on Hugging Face for discovery and sharing.
The collection already contains the COG metadata needed for byte-range reads.
The catalog descriptor also keeps the narrow `index.parquet` attached as a
sidecar index, so spatial and temporal filters can be applied before Rasteret
reads the wider collection data.

You do not need to call `rasteret.build()` for the public AEF collection.

## Small First Read

AEF has one record per year and 64 embedding bands named `A00` through `A63`.
Start with a small area and a single year:

```python
bbox = (11.3, -0.002, 11.5, 0.001)
bands = ["A00", "A01", "A31", "A63"]

sub = collection.subset(
    bbox=bbox,
    date_range=("2023-01-01", "2023-12-31"),
)

cube = sub.get_xarray(
    geometries=bbox,
    bands=bands,
)
```

`get_xarray(...)` is convenient for small windows, plotting, and local analysis.
For many points or training chips, prefer `sample_points(...)` or
`to_torchgeo_dataset(...)` so you do not materialize a large 64-band area at
once.

## Pick The Right API

| Task | Recommended API |
| --- | --- |
| Inspect or plot a small area | `get_xarray(...)` |
| Read embedding vectors at points | `sample_points(...)` |
| Scan or train over many chips | `to_torchgeo_dataset(...)` |

Point sampling:

```python
import pyarrow as pa

points = pa.table(
    {
        "name": ["sample-1"],
        "lon": [11.4],
        "lat": [-0.001],
    }
)

samples = sub.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=[f"A{i:02d}" for i in range(64)],
    geometry_crs=4326,
)
```

The returned Arrow table includes the input `name`, `lon`, and `lat` columns
beside the sampled embedding values.

TorchGeo handoff:

```python
dataset = sub.to_torchgeo_dataset(
    bands=[f"A{i:02d}" for i in range(64)],
    chip_size=256,
)
```

## Bands, Dtype, And Nodata

AEF bands are named `A00` through `A63`:

```python
all_bands = [f"A{i:02d}" for i in range(64)]
```

AEF embeddings are stored as signed 8-bit integers. Nodata is `-128`. Rasteret
preserves the source dtype and carries nodata metadata through xarray assembly.
If you need model-space float embeddings, de-quantize at the point where your
analysis needs floats:

```python
import numpy as np

raw = cube["A00"].values
embedding = np.where(raw == -128, np.nan, raw.astype(np.float32) / 127.0)
```

Keeping the source `int8` values as long as possible reduces memory pressure
when reading many AEF bands.

## Descriptor Details

The AEF catalog entry has three relevant paths:

| Field | Meaning |
| --- | --- |
| `record_table_uri` | The narrow published AEF `index.parquet` record table. |
| `index_uri` | The same `index.parquet`, used by `load()` as the runtime sidecar index. |
| `collection_uri` | The read-ready Rasteret collection under the published `data/` directory. |

The descriptor maps source columns into Rasteret's schema:

| Field | Meaning |
| --- | --- |
| `field_roles` / `column_map` | Aliases source columns such as `fid`, `geom`, `year`, `path`, and `crs`. |
| `href` role | Identifies the source COG URL column. |
| `band_index_map` | Maps `A00` through `A63` to band positions in the multi-band COG. |
| `surface_fields` | Controls which fields are exposed from the index and collection layers. |

AEF-specific raster details are captured in the descriptor and parsed COG
metadata:

- AEF COGs store their geotransform as a ModelTransformationTag rather than the
  more common PixelScale + Tiepoint tags.
- AEF tiles use a south-up orientation with positive Y pixel scale.
- AEF stores the 64 bands in a planar-separate multi-band TIFF layout.

Rasteret's header parser and band index metadata handle those details during
reads. For the general record table contract, see
[Schema Contract](../explanation/schema-contract.md).
