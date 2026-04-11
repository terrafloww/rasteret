# AlphaEarth Foundation Embeddings (AEF)

[AlphaEarth Foundation (AEF)](https://source.coop/repositories/tge-labs/aef)
embeddings are 64-band foundation-model features published as cloud-native
GeoTIFFs. Rasteret also publishes a ready-to-read AEF collection for annual
embeddings, so most users should **load** it instead of rebuilding it.

Use this path first:

```python
import rasteret

collection = rasteret.load("aef/v1-annual")
collection.describe()
```

This opens the maintained AEF Rasteret collection on Source Cooperative. The
Rasteret Collection is already built for annual AEF embeddings and already contains the
COG metadata Rasteret needs for fast byte-range reads. `load()` also keeps the
published `index.parquet` attached as a sidecar index, so spatial and temporal
filters are applied before Rasteret reads the wider collection data.

You do **not** need to run `rasteret.build()` for the public AEF collection.

---

## A Small First Read

AEF has one record per year and 64 embedding bands named `A00` through `A63`.
Start with a small area and one year:

```python
bbox = (-83.86, 39.54, -83.82, 39.58)  # small WGS84 bbox
bands = ["A00", "A01", "A31", "A63"]

sub = collection.subset(
    bbox=bbox,
    date_range=("2024-01-01", "2024-12-31"),
)

cube = sub.get_xarray(
    geometries=bbox,
    bands=bands,
)

cube
```

`get_xarray(...)` materializes the requested area into an xarray Dataset. That
is convenient for small windows, plotting, and local analysis. For large areas,
prefer `sample_points(...)` or `to_torchgeo_dataset(...)` so you do not load a
large 64-band mosaic all at once.

---

## Pick The Right API

| Task | Recommended API | Why |
|---|---|---|
| Inspect or plot a small area | `get_xarray(...)` | Returns a familiar xarray Dataset. |
| Read embedding vectors at points | `sample_points(...)` | Reads only the pixels under your points. |
| Scan or train over many chips | `to_torchgeo_dataset(...)` | Streams chips through TorchGeo/DataLoader instead of materializing the whole area. |

Example point sampling:

```python
import geopandas as gpd
from shapely.geometry import Point

points = gpd.GeoDataFrame(
    {"name": ["sample-1"]},
    geometry=[Point(-83.84, 39.56)],
    crs="EPSG:4326",
)

samples = sub.sample_points(
    points,
    bands=[f"A{i:02d}" for i in range(64)],
    geometry_crs=4326,
)

samples.to_pandas().head()
```

Example TorchGeo handoff:

```python
tg_dataset = sub.to_torchgeo_dataset(
    bands=[f"A{i:02d}" for i in range(64)],
)
```

See the AEF notebooks for larger end-to-end examples:

- `notebooks/07_aef_similarity_search.ipynb`
- `notebooks/08_aef_fire_lancedb_torchgeo.ipynb`

---

## Bands, Dtype, And Nodata

AEF COGs contain 64 bands: `A00` through `A63`.

```python
all_bands = [f"A{i:02d}" for i in range(64)]
```

AEF embeddings are stored as signed 8-bit integers. Nodata is `-128`. Rasteret
preserves the source dtype and carries nodata metadata through xarray assembly.
If you need model-space float embeddings, de-quantize explicitly at the point
where your analysis needs floats:

```python
import numpy as np

raw = cube["A00"].values
embedding = np.where(raw == -128, np.nan, raw.astype(np.float32) / 127.0)
```

Do not de-quantize earlier than needed. Keeping the source `int8` values reduces
memory pressure, especially when reading many AEF bands.

---


## How The Built-In AEF Descriptor Works

The AEF entry in Rasteret's catalog has three relevant paths:

| Descriptor field | What it points to |
|---|---|
| `record_table_uri` | The narrow published AEF `index.parquet` record table. |
| `index_uri` | The same `index.parquet`, used by `load()` as the runtime sidecar index. |
| `collection_uri` | The read-ready Rasteret Collection under the published `data/` directory. |

The descriptor also maps source columns into Rasteret's schema:

| Descriptor field | Purpose |
|---|---|
| `field_roles` / `column_map` | Aliases source columns such as `fid`, `geom`, `year`, and `path` into Rasteret's schema. |
| `href` role | Identifies the source COG URL column. |
| `band_index_map` | Maps `A00` through `A63` to band positions in the multi-band COG. |
| `surface_fields` | Controls which fields are exposed from the index and collection layers. |

Rasteret reads AEF COGs with the same reader used for other collections. The
AEF-specific details are captured in the catalog descriptor and the published
collection metadata:

- **ModelTransformationTag**: AEF COGs store their geotransform as a 4x4 matrix
  (TIFF tag 34264) instead of the more common PixelScale + Tiepoint tags.
  Rasteret's header parser handles both forms.
- **South-up orientation**: AEF tiles use a positive Y pixel scale. Rasteret's
  tile math handles both north-up and south-up rasters.
- **Planar-separate multi-band layout**: AEF stores the 64 bands separately in
  the TIFF. Rasteret uses the band index metadata to read only the requested
  bands and tile byte ranges.

For the asset schema details, see
[Schema Contract - asset dict](../explanation/schema-contract.md).
