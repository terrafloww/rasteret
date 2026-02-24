"""Collection management: all pure-rasteret features in one script.

Demonstrates:
- Building a collection from STAC
- Persisting and reloading from Parquet
- Discovering cached collections
- Filtering with subset(), where(), and raw Arrow expressions
- Registering custom cloud configs and band mappings
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds

import rasteret
from rasteret import CloudConfig, Collection
from rasteret.constants import BandRegistry

# ---------------------------------------------------------------------------
# 1. Build a collection from STAC
# ---------------------------------------------------------------------------

workspace = Path.home() / "rasteret_workspace"
BBOX = (77.55, 13.01, 77.58, 13.08)

collection = rasteret.build_from_stac(
    name="bangalore",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection="sentinel-2-l2a",
    bbox=BBOX,
    date_range=("2024-01-01", "2024-06-30"),
    workspace_dir=workspace,
)

print(f"Built collection: {collection.name}")
print(f"Rows: {collection.dataset.count_rows()}")

# ---------------------------------------------------------------------------
# 2. Persist and reload
# ---------------------------------------------------------------------------

save_path = workspace / "demo_saved"
collection.export(save_path)
print(f"\nExported to {save_path}")

reloaded = rasteret.load(save_path, name="demo_saved")
print(f"Reloaded: {reloaded.name}, rows={reloaded.dataset.count_rows()}")

# ---------------------------------------------------------------------------
# 3. Discover cached collections
# ---------------------------------------------------------------------------

cached = Collection.list_collections(workspace)
print(f"\nCached collections ({len(cached)}):")
for c in cached:
    print(f"  {c['name']} ({c['kind']}), {c['size']} rows, source={c['data_source']}")

# ---------------------------------------------------------------------------
# 4. Filtering: subset()
# ---------------------------------------------------------------------------

# Cloud cover + date range
filtered = collection.subset(
    cloud_cover_lt=15,
    date_range=("2024-03-01", "2024-04-30"),
)
print(f"\nCloud < 15% + Mar-Apr: {filtered.dataset.count_rows()} rows")

# Bbox filter (requires scalar bbox columns, produced by normalize layer)
try:
    bbox_filtered = collection.subset(bbox=(77.55, 13.03, 77.57, 13.06))
    print(f"Bbox filtered: {bbox_filtered.dataset.count_rows()} rows")
except ValueError as e:
    print(f"Bbox filter skipped: {e}")

# ---------------------------------------------------------------------------
# 5. Filtering: where() with raw Arrow expressions
# ---------------------------------------------------------------------------

expr = ds.field("eo:cloud_cover") < 5.0
clear_scenes = collection.where(expr)
print(f"\nCloud < 5% (raw Arrow): {clear_scenes.dataset.count_rows()} rows")

# ---------------------------------------------------------------------------
# 6. Register custom cloud config
# ---------------------------------------------------------------------------

CloudConfig.register(
    "my-private-collection",
    CloudConfig(
        provider="aws",
        requester_pays=True,
        region="eu-central-1",
        url_patterns={"https://my-cdn.example.com/": "s3://my-private-bucket/"},
    ),
)
print(f"\nRegistered cloud config: {CloudConfig.get_config('my-private-collection')}")

# ---------------------------------------------------------------------------
# 7. Custom band mapping (for build_from_stac with custom STAC APIs)
#    For catalog datasets, band_map in DatasetDescriptor handles this
#    automatically - you only need BandRegistry for non-catalog sources.
# ---------------------------------------------------------------------------

BandRegistry.register(
    "my-private-collection",
    {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir"},
)
print(f"Registered bands: {BandRegistry.get('my-private-collection')}")
print(f"All registered: {BandRegistry.list_registered()}")
