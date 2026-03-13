# Configuring Custom Collections[¶](#configuring-custom-collections)

Rasteret ships with a catalog of 12 datasets across Earth Search, Planetary Computer, and AEF. You can register any additional collection. This notebook covers:

1. `CloudConfig.register()`: URL signing / requester-pays for custom collections
1. `BandRegistry.register()`: map band codes to asset names
1. Backend usage patterns: `create_backend()` (recommended) and `ObstoreBackend` (advanced)

No live cloud credentials are required for the runnable cells in this notebook.

In \[ \]:

Copied!

```
from rasteret import CloudConfig
from rasteret.constants import BandRegistry
```

from rasteret import CloudConfig from rasteret.constants import BandRegistry

## 1. Register a custom cloud config[¶](#1-register-a-custom-cloud-config)

`CloudConfig` tells Rasteret how to resolve and sign URLs for a collection. Built-in configs cover datasets that need URL rewriting or requester-pays access (Landsat, NAIP, Planetary Computer). See `rasteret datasets list` for the full catalog. Register your own for private or non-standard collections.

Fields:

- `provider`: `"aws"` (only provider with built-in URL rewriting today)
- `requester_pays`: if `True`, Rasteret authenticates requests with your credentials
- `region`: AWS region for the S3 bucket
- `url_patterns`: rewrite rules from HTTP CDN URLs to `s3://` paths

In \[ \]:

Copied!

```
# Register a hypothetical private Sentinel-1 collection
CloudConfig.register(
    "my-sentinel-1-grd",
    CloudConfig(
        provider="aws",
        requester_pays=True,
        region="eu-central-1",
        url_patterns={
            "https://my-cdn.example.com/sar/": "s3://my-sar-bucket/",
        },
    ),
)

# Verify it's registered
config = CloudConfig.get_config("my-sentinel-1-grd")
print(f"Provider:       {config.provider}")
print(f"Region:         {config.region}")
print(f"Requester pays: {config.requester_pays}")
print(f"URL patterns:   {config.url_patterns}")
```

# Register a hypothetical private Sentinel-1 collection

CloudConfig.register( "my-sentinel-1-grd", CloudConfig( provider="aws", requester_pays=True, region="eu-central-1", url_patterns={ "https://my-cdn.example.com/sar/": "s3://my-sar-bucket/", }, ), )

# Verify it's registered

config = CloudConfig.get_config("my-sentinel-1-grd") print(f"Provider: {config.provider}") print(f"Region: {config.region}") print(f"Requester pays: {config.requester_pays}") print(f"URL patterns: {config.url_patterns}")

### Built-in configs[¶](#built-in-configs)

You can inspect what's already registered:

In \[ \]:

Copied!

```
for name in ["sentinel-2-l2a", "landsat-c2-l2"]:
    c = CloudConfig.get_config(name)
    if c:
        print(f"{name}: region={c.region}, requester_pays={c.requester_pays}")
```

for name in \["sentinel-2-l2a", "landsat-c2-l2"\]: c = CloudConfig.get_config(name) if c: print(f"{name}: region={c.region}, requester_pays={c.requester_pays}")

## 2. Register custom band mappings[¶](#2-register-custom-band-mappings)

`BandRegistry` maps user-facing band codes (like `"VV"`) to the asset key names stored in the STAC catalog. For built-in collections (Sentinel-2, Landsat) the mapping is identity, but other collections use different keys.

Example: Sentinel-1 GRD assets are stored under lowercase keys (`vv`, `vh`, `local_incidence_angle`), but you want to reference them as `VV`, `VH`, `angle` in your code.

In \[ \]:

Copied!

```
BandRegistry.register(
    "my-sentinel-1-grd",
    {
        "VV": "vv",
        "VH": "vh",
        "angle": "local_incidence_angle",
    },
)

print(f"Bands: {BandRegistry.get('my-sentinel-1-grd')}")
print(f"All registered: {BandRegistry.list_registered()}")
```

BandRegistry.register( "my-sentinel-1-grd", { "VV": "vv", "VH": "vh", "angle": "local_incidence_angle", }, ) print(f"Bands: {BandRegistry.get('my-sentinel-1-grd')}") print(f"All registered: {BandRegistry.list_registered()}")

## 3. Backend options for cloud I/O[¶](#3-backend-options-for-cloud-io)

Rasteret uses obstore for all remote byte-range reads. For most authenticated workflows, use `create_backend()` to pass an obstore credential provider.

```
import rasteret
from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider

pc_asset_url = "https://naipeuwest.blob.core.windows.net/naip/v002/"
backend = rasteret.create_backend(
    credential_provider=PlanetaryComputerCredentialProvider(pc_asset_url)
)

ds = collection.get_xarray(geometries=[aoi], bands=["B04"], backend=backend)
```

For advanced cases, you can still build a custom `ObstoreBackend` from a pre-configured store.

```
from obstore.store import S3Store
from rasteret.cloud import ObstoreBackend

store = S3Store.from_url("s3://my-sar-bucket", config={"region": "eu-central-1"})
backend = ObstoreBackend(store, url_prefix="s3://my-sar-bucket/")

ds = collection.get_xarray(geometries=[aoi], bands=["B04"], backend=backend)
```

The `StorageBackend` protocol requires only two methods:

- `get_range(url, start, length) -> bytes`
- `get_ranges(url, ranges) -> list[bytes]`

Implement this protocol to plug in any I/O backend (fsspec, mocked readers for tests, etc.).

## Summary[¶](#summary)

| Feature                   | What it does                                        |
| ------------------------- | --------------------------------------------------- |
| `CloudConfig.register()`  | URL rewriting + signing for any collection          |
| `BandRegistry.register()` | Map band codes to asset keys                        |
| `create_backend()`        | Recommended way to pass credential providers        |
| `ObstoreBackend`          | Advanced custom store wrapper                       |
| `StorageBackend` protocol | Implement `get_range` / `get_ranges` for custom I/O |

Next: [TorchGeo Benchmark](https://terrafloww.github.io/rasteret/tutorials/05_torchgeo_comparison/index.md)
