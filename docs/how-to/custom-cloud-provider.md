# Invisible Connectivity: Multi-Cloud Auth

In Rasteret, connectivity is designed to be invisible. Our I/O engine automatically routes your requests to the correct cloud store (S3, GCS, Azure) based on the URL pattern and handles standard public access without any configuration.

This guide covers what to do in the cases where you **do** need to manage connectivity: custom requester-pays buckets, private buckets, and custom credential providers.

---

This guide covers three layers, from simplest to most advanced:

1. **Requester-pays / URL rewriting** - `CloudConfig` (URL patterns + auto-resolved credentials)
2. **Dynamic credentials** - `create_backend()` with obstore credential providers
3. **Custom I/O** - implement `StorageBackend` (or wrap a store) and pass `backend=`

## Register a cloud config

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

The `url_patterns` dict maps HTTP URL prefixes to S3 URL prefixes.
When Rasteret encounters a COG URL starting with the HTTP pattern, it
rewrites it to the S3 pattern for authenticated access.

## Use the config

Once registered, the config is picked up automatically when
`data_source` matches:

```python
import rasteret

collection = rasteret.build_from_stac(
    name="private-data",
    stac_api="https://my-stac.example.com/v1",
    collection="my-private-collection",
    # Optional: namespace conventions to avoid collisions across providers.
    # data_source="acme/my-private-collection",
    bbox=(-0.2, 51.4, 0.2, 51.7),
    date_range=("2024-01-01", "2024-06-30"),
)
```

Or pass the config explicitly via `cloud_config` when you need requester-pays
or URL rewriting:

```python
from rasteret import CloudConfig

config = CloudConfig.get_config("my-private-collection")

ds = collection.get_xarray(
    geometries=(-0.1, 51.45, 0.1, 51.65),  # bbox tuple
    bands=["B04", "B08"],
    cloud_config=config,
)
```

Rasteret auto-creates a backend from the config at read time.
For requester-pays buckets, AWS credentials are resolved automatically
from environment variables or `~/.aws/credentials` via boto3.

## Built-in configs

Rasteret ships with a few pre-registered configs for common sources, and the
catalog also registers per-dataset configs when a `DatasetDescriptor` includes
`cloud_config`.

The key thing: configs are looked up by **Collection `data_source`**. When in
doubt, print `collection.data_source` and register/lookup under that value.

Check what's registered:

```python
CloudConfig.get_config("sentinel-2-l2a")
# CloudConfig(provider='aws', requester_pays=False, region='us-west-2', ...)
```

For catalog datasets, the `data_source` is the catalog ID (e.g.
`earthsearch/landsat-c2-l2`, `earthsearch/naip`), so those are the keys you
should use with `CloudConfig.get_config(...)` if you want to inspect or
override built-in behavior.

## Multi-cloud URL routing (via obstore)

Rasteret's IO layer uses obstore as the HTTP transport and natively routes
URLs to the correct cloud store:

| URL pattern | Store type |
|---|---|
| `s3://bucket/...` | `S3Store` |
| `*.s3.*.amazonaws.com/...` | `S3Store` |
| `gs://bucket/...` | `GCSStore` |
| `storage.googleapis.com/bucket/...` | `GCSStore` |
| `*.blob.core.windows.net/container/...` | `AzureStore` |
| Pre-signed / SAS-signed URLs (query params) | `HTTPStore` |
| Other HTTPS | `HTTPStore` |

This happens automatically -- no configuration needed for public data.

## Authenticated cloud reads

Use `create_backend()` when your data source provides its own credential
mechanism (Planetary Computer SAS tokens, Earthdata-style temporary S3 credentials).
This passes the credential provider to the underlying cloud store
(S3Store, AzureStore, GCSStore):

### Planetary Computer

Built-in `pc/*` datasets work via `build()` when `rasteret[azure]` is installed.
Rasteret signs STAC assets during the build so COG header enrichment can read
bytes from Azure.

!!! note "Rate limits"

    Planetary Computer has two different network surfaces:

    - **SAS signing** (calling the Planetary Computer API to obtain short-lived SAS URLs) is **rate-limited** and can return **HTTP 429**.
    - **COG reads** (range requests to Azure Blob URLs that already include SAS tokens) go **directly to Azure Blob Storage**, and do not depend on the signing API.

    If you hit signing rate limits, reduce query size (e.g. `query={"max_items": 1}`), retry later, or configure a subscription key via
    `PC_SDK_SUBSCRIPTION_KEY` (or `planetarycomputer configure`) for less restrictive rate limits.

    Separately, Azure Blob Storage itself can throttle very high request rates (e.g. **429/503**). If you see those while reading tiles/headers,
    lower `max_concurrent` and retry.

If you want **long-lived cached Collections** (without embedding SAS tokens in
the Parquet index), create a backend and pass it to both `build()` and reads:

```python
import rasteret
from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider

pc_asset_url = "https://naipeuwest.blob.core.windows.net/naip/v002/"
backend = rasteret.create_backend(
    credential_provider=PlanetaryComputerCredentialProvider(pc_asset_url)
)
collection = rasteret.build(
    "pc/sentinel-2-l2a",
    name="pc-s2",
    bbox=(-122.45, 37.74, -122.35, 37.84),
    date_range=("2024-06-01", "2024-07-15"),
    backend=backend,
)
ds = collection.get_xarray(
    geometries=(-122.45, 37.74, -122.35, 37.84),
    bands=["B04"],
    backend=backend,
)
```

### Earthdata (temporary S3 credentials)

```python
import rasteret
from obstore.auth.earthdata import NasaEarthdataCredentialProvider

credentials_url = "https://data.lpdaac.earthdata.nasa.gov/s3credentials"
backend = rasteret.create_backend(
    credential_provider=NasaEarthdataCredentialProvider(credentials_url),
    region="us-west-2",
)
ds = collection.get_xarray(
    geometries=(-105.5, 40.0, -105.0, 40.5),
    bands=["B04"],
    backend=backend,
)
```
Use the DAAC (Distributed Active Archive Center) specific `s3credentials`
endpoint for the assets you query.

Rasteret does **not** prompt for credentials. Configure Earthdata auth
via one of:

- `~/.netrc` (recommended), or
- `EARTHDATA_TOKEN`, or
- `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

For custom Earthdata-backed datasets, set `s3_credentials_url` on your
dataset descriptor (or pass a pre-built backend) so Rasteret can fetch
temporary credentials as needed.

### Custom ObstoreBackend

For advanced use cases (e.g. wrapping a pre-configured store with custom
client options), use [`ObstoreBackend`](../reference/cloud.md) directly:

```python
from rasteret.cloud import ObstoreBackend
from obstore.store import S3Store

store = S3Store(
    bucket="my-private-bucket",
    config={"region": "eu-central-1"},
)
backend = ObstoreBackend(store, url_prefix="s3://my-private-bucket/")

ds = collection.get_xarray(
    geometries=(10.0, 48.0, 10.5, 48.5),
    bands=["B04"],
    backend=backend,
)
```

This is only needed for store configurations that `create_backend()` does
not cover. For most use cases, `create_backend()` with a credential
provider is sufficient.
