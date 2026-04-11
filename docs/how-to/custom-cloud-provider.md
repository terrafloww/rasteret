# Custom Cloud Provider

Rasteret can read public HTTP, S3, GCS, and Azure Blob COG URLs without custom
setup. Use this page when your data needs requester-pays handling, URL
rewriting, temporary credentials, or a preconfigured storage backend.

Most users only need one of these:

| Need | Use |
| --- | --- |
| Rewrite provider URLs and configure requester-pays S3 | `CloudConfig` |
| Use temporary cloud credentials from an obstore provider | `rasteret.create_backend(...)` |
| Wrap a custom obstore store yourself | `ObstoreBackend` |

## Public Data

For public COG URLs, no configuration is usually needed. Rasteret routes common
URL shapes to the appropriate store:

| URL pattern | Store |
| --- | --- |
| `s3://bucket/...` | S3 |
| `*.s3.*.amazonaws.com/...` | S3 |
| `gs://bucket/...` | GCS |
| `storage.googleapis.com/bucket/...` | GCS |
| `*.blob.core.windows.net/container/...` | Azure Blob |
| pre-signed or SAS URLs | HTTP |
| other HTTPS URLs | HTTP |

If public reads fail, first check whether the bucket is requester-pays, private,
or behind provider-specific signing.

## CloudConfig For URL Rewriting

Use `CloudConfig` when URLs in metadata should be rewritten before range reads.
This is common when a catalog exposes HTTPS URLs but authenticated reads should
go through `s3://...`.

```python
import rasteret
from rasteret import CloudConfig

config = CloudConfig(
    provider="aws",
    requester_pays=True,
    region="eu-central-1",
    url_patterns={
        "https://my-cdn.example.com/": "s3://my-private-bucket/",
    },
)

CloudConfig.register("acme/private-scenes", config)
```

When a collection has `data_source="acme/private-scenes"`, Rasteret can look up
the config automatically. You can also pass it explicitly:

```python
ds = collection.get_xarray(
    geometries=(-0.1, 51.45, 0.1, 51.65),
    bands=["B04", "B08"],
    cloud_config=config,
)
```

For requester-pays S3, Rasteret configures `request_payer=true` on the backend.
Make sure your AWS credentials are available in the environment or standard AWS
credential files.

## Use A Config While Building

Pass the same config during build when COG header enrichment needs authenticated
range reads:

```python
collection = rasteret.build_from_stac(
    name="private-scenes",
    stac_api="https://my-stac.example.com/v1",
    collection="private-scenes",
    data_source="acme/private-scenes",
    bbox=(-0.2, 51.4, 0.2, 51.7),
    date_range=("2024-01-01", "2024-06-30"),
    cloud_config=config,
)
```

If you build through a registered dataset catalog entry, Rasteret can register
the entry's `cloud_config` for you. When debugging, check:

```python
collection.data_source
CloudConfig.get_config(collection.data_source)
```

## Temporary Credentials With `create_backend()`

Use `create_backend()` when the provider supplies an obstore credential provider
for temporary credentials or request signing. Pass the backend to both build and
read calls when both phases need the same authenticated access.

### Planetary Computer

```python
import rasteret
from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider

asset_prefix = "https://naipeuwest.blob.core.windows.net/naip/v002/"
backend = rasteret.create_backend(
    credential_provider=PlanetaryComputerCredentialProvider(asset_prefix)
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

Planetary Computer signing can be rate-limited. If you see signing HTTP 429s,
try a smaller query, retry later, or configure a Planetary Computer subscription
key. Tile reads against already signed Azure URLs are a different network path.

### Earthdata S3 Credentials

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

Use the DAAC-specific `s3credentials` endpoint for the assets you query.
Rasteret does not prompt for credentials; configure Earthdata auth with
`~/.netrc`, `EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` /
`EARTHDATA_PASSWORD`.

## Advanced: Wrap An Obstore Store

Use `ObstoreBackend` when you already have a configured obstore store and want
Rasteret to use it directly:

```python
from obstore.store import S3Store

from rasteret.cloud import ObstoreBackend

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

Prefer `CloudConfig` or `create_backend()` unless you really need direct control
over the underlying store.
