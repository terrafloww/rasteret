# Compatibility matrix

Rasteret supports multiple catalogs and auth mechanisms. This page is a
release-time checklist: it makes the supported combinations explicit and
points to the tests that cover them.

## What gets tested

- **Default CI** runs unit/integration tests plus a small set of **public,
  no-auth network smoke tests** (anonymous range reads + a GeoParquet ingest
  path).
- **Live smoke** (`pytest -m network`) runs end-to-end against STAC endpoints
  and range reads. These tests are skipped automatically
  when optional dependencies or credentials are missing.

## Matrix (v0.3.x)

| Source | Catalog IDs | Auth mechanism | Build-time enrichment I/O | Covered by default CI | Covered by live smoke |
|---|---|---|---|---|---|
| Earth Search (Element84) | `earthsearch/*` | Public | HTTPS / S3 anonymous | `test_compat_matrix.py` + `test_public_network_smoke.py` | `test_network_smoke.py` |
| Planetary Computer | `pc/*` | SAS-signed HTTPS | HTTPS (signed) | `test_compat_matrix.py` | `test_network_smoke.py` (needs `rasteret[azure]`) |
| Requester-pays S3 (Landsat, etc.) | `earthsearch/landsat-c2-l2` | AWS creds + requester-pays | Pre-signed HTTPS | Unit-level only (no creds) | Not enabled by default (requires AWS creds) |

## Output path test coverage

| Path | Tested against | Data tested | Test file |
|---|---|---|---|
| `get_xarray()` | rasterio windowed read | Sentinel-2 uint16, AEF int8 | `test_execution.py` |
| `get_gdf()` | rasterio windowed read | Sentinel-2 uint16 | `test_execution.py` |
| `to_torchgeo_dataset()` | Pure TorchGeo GeoDataset | Sentinel-2 uint16 | `test_torchgeo_network.py` |

## Running live smoke locally

```bash
uv sync --extra dev --extra azure
uv run pytest -m network -q
```

To require that every live check runs (fail on skips):

```bash
uv run pytest -m network -q --network-strict
```

In strict mode, skips become failures so you can treat the live suite as
a release gate.
