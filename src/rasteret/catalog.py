# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Dataset registry: spec-aligned descriptors for known COG collections.

Each :class:`DatasetDescriptor` is a proto-spec-descriptor that captures
identity, access, and band-mapping metadata for a cloud-native GeoTIFF
collection.  The :class:`DatasetRegistry` stores them in-memory and
auto-populates :class:`~rasteret.constants.BandRegistry` and
:class:`~rasteret.cloud.CloudConfig` keyed by STAC collection id.

Users can register custom datasets at runtime::

    import rasteret
    from rasteret.catalog import DatasetDescriptor

    rasteret.register(DatasetDescriptor(
        id="acme/field-survey-2024",
        name="ACME Field Survey",
        stac_api="https://acme.example.com/stac/v1",
        stac_collection="field-survey-2024",
        band_map={"RGB": "image"},
        separate_files=False,
        license="proprietary",
        license_url="https://acme.example.com/license",
    ))
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


def _local_registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv("RASTERET_LOCAL_DATASETS_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".rasteret" / "datasets.local.json"


@dataclass(frozen=True)
class DatasetDescriptor:
    """A dataset descriptor: identity + access + band mapping.

    Proto-spec-descriptor.  Each entry will migrate to YAML format
    when the spec ships.  Fields map to spec axes:

        id, name, description             -> dataset identity
        stac_api, stac_collection         -> access (stac_query)
        geoparquet_uri                    -> access (parquet_record_table)
        band_map                          -> field roles (input bands)
        spatial_coverage, temporal_range  -> coverage metadata
        license, license_url,
        commercial_use                    -> licensing
        static_catalog                    -> static STAC catalog traversal
        column_map, href_column,
        band_index_map, bbox_columns      -> normalisation hints

    Parameters
    ----------
    id : str
        Namespaced identifier (e.g. ``"earthsearch/sentinel-2-l2a"``).
    name : str
        Human-readable name.
    description : str
        One-liner description.
    stac_api : str, optional
        STAC API endpoint URL.  For static STAC catalogs (no ``/search``
        endpoint), this is the URL to the root ``catalog.json`` file and
        ``static_catalog`` must be ``True``.
    stac_collection : str, optional
        STAC collection identifier.  May be ``None`` for static catalogs
        that should be traversed from the root.
    geoparquet_uri : str, optional
        URI to a GeoParquet record table.
    column_map : dict, optional
        ``{source: contract}`` alias map for GeoParquet normalisation.
        Source columns are preserved; contract-name columns are added
        as zero-copy aliases.
    href_column : str, optional
        Column in the GeoParquet containing COG URLs.  When set and
        ``assets`` is absent, the normalisation layer builds the
        ``assets`` struct from this column and ``band_index_map``.
    band_index_map : dict, optional
        ``{band_code: sample_index}`` for multi-band COGs.  Used with
        ``href_column`` to construct per-band asset references with
        ``band_index``.
    bbox_columns : dict, optional
        ``{"minx": col, "miny": col, "maxx": col, "maxy": col}``
        mapping source column names for spatial filtering on the
        GeoParquet index.  Used by ``build()`` to construct a
        ``filter_expr`` so only relevant rows are enriched.
    band_map : dict, optional
        Mapping of band code to STAC asset name.
    separate_files : bool
        ``True`` when each band is a separate COG file (default).
    spatial_coverage : str
        Geographic coverage hint (e.g. ``"global"``).
    temporal_range : tuple of str, optional
        ``(start, end)`` ISO date strings.
    requires_auth : bool
        Whether credentials are needed to access the data.
    license : str
        License identifier.  Use the value reported by the STAC API
        (typically an SPDX id like ``"CC-BY-4.0"`` or ``"proprietary"``
        for bespoke open-access licenses).
    license_url : str
        URL to the full license text.  Sourced from the STAC collection's
        ``rel=license`` link.
    commercial_use : bool
        ``True`` (default) when the license permits commercial use.
        ``False`` for licenses like ``CC-BY-NC-4.0``.
    static_catalog : bool
        ``True`` when ``stac_api`` points to a static STAC catalog
        (a ``catalog.json`` on S3) rather than a queryable STAC API
        with a ``/search`` endpoint.  Static catalogs are traversed
        with ``pystac.Catalog.from_file()`` and filtered client-side.
    s3_credentials_url : str, optional
        Endpoint for obtaining temporary S3 credentials for auth-gated datasets.
        When set, ``build()`` can auto-construct a backend using ``obstore``
        credential providers and the user's ``.netrc`` / environment variables.
    example_bbox : tuple of float, optional
        Example bounding box (minx, miny, maxx, maxy) known to return data.
        Used in docs and live smoke tests.
    example_date_range : tuple of str, optional
        Example ISO date range (start, end) known to return data. Used in docs
        and live smoke tests.
    cloud_config : dict, optional
        Cloud provider configuration for URL resolution.
    torchgeo_class : str, optional
        Equivalent TorchGeo class name (reference only, not a dependency).
    torchgeo_verified : bool
        ``True`` when the underlying data source has been confirmed to be
        the same files that the TorchGeo class reads.
    """

    # --- Identity (spec: dataset block) ---
    id: str
    name: str
    description: str = ""

    # --- Access (spec: record_source via parquet_record_table) ---
    stac_api: str | None = None
    stac_collection: str | None = None
    geoparquet_uri: str | None = None
    column_map: dict[str, str] | None = None

    # --- GeoParquet normalisation hints ---
    href_column: str | None = None
    band_index_map: dict[str, int] | None = None
    bbox_columns: dict[str, str] | None = None

    # --- Band mapping (spec: fields with role=input) ---
    band_map: dict[str, str] | None = None
    separate_files: bool = True

    # --- Coverage metadata ---
    spatial_coverage: str = ""
    temporal_range: tuple[str, str] | None = None
    requires_auth: bool = False
    license: str = ""  # SPDX id or "proprietary" (matches STAC collection metadata)
    license_url: str = ""  # Link to full license text
    commercial_use: bool = True  # False when license prohibits commercial use

    # --- Static STAC catalog support ---
    static_catalog: bool = False  # True for static STAC catalogs (no /search endpoint)

    # --- Auth / Cloud configuration ---
    s3_credentials_url: str | None = None
    cloud_config: dict[str, str] | None = None
    example_bbox: tuple[float, float, float, float] | None = None
    example_date_range: tuple[str, str] | None = None

    # --- Cross-references ---
    torchgeo_class: str | None = None
    torchgeo_verified: bool = False


class DatasetRegistry:
    """Registry of dataset descriptors.  Proto-spec catalog.

    Built-in datasets are registered at module import time.
    Users can add entries via :meth:`register` or the top-level
    :func:`rasteret.register` helper.
    """

    _descriptors: ClassVar[dict[str, DatasetDescriptor]] = {}

    @classmethod
    def register(cls, descriptor: DatasetDescriptor) -> None:
        """Register a dataset descriptor.

        Also populates :class:`~rasteret.constants.BandRegistry` and
        :class:`~rasteret.cloud.CloudConfig` keyed by the descriptor id so that
        provider-specific conventions do not collide (e.g. Planetary Computer
        vs Earth Search for ``sentinel-2-l2a``).
        """
        cls._descriptors[descriptor.id] = descriptor

        # Populate BandRegistry keyed by descriptor id (namespaced).
        # Skip if an entry already exists (first-write-wins).
        if descriptor.band_map:
            from rasteret.constants import BandRegistry

            if not BandRegistry.get(descriptor.id):
                BandRegistry.register(descriptor.id, descriptor.band_map)

        # Populate CloudConfig keyed by descriptor id (namespaced).
        if descriptor.cloud_config:
            from rasteret.cloud import CloudConfig

            CloudConfig.register(
                descriptor.id,
                CloudConfig(
                    provider=descriptor.cloud_config.get("provider", "aws"),
                    requester_pays=descriptor.cloud_config.get("requester_pays", False),
                    region=descriptor.cloud_config.get("region", "us-west-2"),
                    url_patterns=descriptor.cloud_config.get("url_patterns", {}),
                ),
            )

    @classmethod
    def unregister(cls, dataset_id: str) -> DatasetDescriptor | None:
        """Remove a descriptor from the in-memory registry."""
        return cls._descriptors.pop(dataset_id, None)

    @classmethod
    def get(cls, dataset_id: str) -> DatasetDescriptor | None:
        """Look up a descriptor by namespaced ID.

        Parameters
        ----------
        dataset_id : str
            Full namespaced id (e.g. ``"earthsearch/sentinel-2-l2a"``).
        """
        return cls._descriptors.get(dataset_id)

    @classmethod
    def list(cls) -> list[DatasetDescriptor]:
        """Return all registered descriptors."""
        return list(cls._descriptors.values())

    @classmethod
    def search(cls, keyword: str) -> list[DatasetDescriptor]:
        """Search descriptors by keyword in id, name, or description.

        Parameters
        ----------
        keyword : str
            Case-insensitive search term.
        """
        kw = keyword.lower()
        return [
            d
            for d in cls._descriptors.values()
            if kw in d.id.lower() or kw in d.name.lower() or kw in d.description.lower()
        ]


def load_local_descriptors(
    path: str | Path | None = None,
) -> list[DatasetDescriptor]:
    """Load persisted local dataset descriptors from JSON.

    Invalid entries are skipped with a warning.
    """
    registry_path = _local_registry_path(path)
    if not registry_path.exists():
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to read local dataset registry %s: %s", registry_path, exc
        )
        return []

    if not isinstance(payload, list):
        logger.warning(
            "Local dataset registry %s must contain a JSON list", registry_path
        )
        return []

    descriptors: list[DatasetDescriptor] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            descriptors.append(DatasetDescriptor(**entry))
        except TypeError as exc:
            dataset_id = entry.get("id", "<missing-id>")
            logger.warning(
                "Skipping invalid local dataset descriptor %s: %s", dataset_id, exc
            )
    return descriptors


def _write_local_descriptors(
    descriptors: list[DatasetDescriptor],
    registry_path: Path,
) -> None:
    payload: list[dict[str, Any]] = [
        asdict(descriptor) for descriptor in sorted(descriptors, key=lambda d: d.id)
    ]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_local_descriptor(
    descriptor: DatasetDescriptor,
    path: str | Path | None = None,
) -> None:
    """Persist a local dataset descriptor to JSON (upsert by id)."""
    registry_path = _local_registry_path(path)
    existing = {d.id: d for d in load_local_descriptors(registry_path)}
    existing[descriptor.id] = descriptor
    _write_local_descriptors(list(existing.values()), registry_path)


def remove_local_descriptor(
    dataset_id: str,
    path: str | Path | None = None,
) -> DatasetDescriptor | None:
    """Remove one persisted local descriptor (if present)."""
    registry_path = _local_registry_path(path)
    descriptors = load_local_descriptors(registry_path)
    removed: DatasetDescriptor | None = None
    kept: list[DatasetDescriptor] = []

    for descriptor in descriptors:
        if descriptor.id == dataset_id:
            removed = descriptor
        else:
            kept.append(descriptor)

    if removed is None:
        return None

    _write_local_descriptors(kept, registry_path)
    return removed


def unregister_local_descriptor(
    dataset_id: str,
    path: str | Path | None = None,
) -> DatasetDescriptor | None:
    """Unregister a local dataset from persisted and in-memory registries."""
    persisted = remove_local_descriptor(dataset_id, path=path)

    in_memory = DatasetRegistry.get(dataset_id)
    removed_in_memory: DatasetDescriptor | None = None
    if (
        in_memory is not None
        and in_memory.geoparquet_uri
        and in_memory.spatial_coverage == "local"
    ):
        removed_in_memory = DatasetRegistry.unregister(dataset_id)

    return persisted or removed_in_memory


def export_local_descriptor(
    dataset_id: str,
    output_path: str | Path,
    path: str | Path | None = None,
) -> Path:
    """Export one local descriptor as JSON for sharing."""
    descriptor = next(
        (entry for entry in load_local_descriptors(path) if entry.id == dataset_id),
        None,
    )

    if descriptor is None:
        runtime_descriptor = DatasetRegistry.get(dataset_id)
        if (
            runtime_descriptor is not None
            and runtime_descriptor.geoparquet_uri
            and runtime_descriptor.spatial_coverage == "local"
        ):
            descriptor = runtime_descriptor

    if descriptor is None:
        raise KeyError(f"Local dataset '{dataset_id}' not found.")

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(descriptor), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


# ---------------------------------------------------------------------------
# Built-in dataset descriptors
# ---------------------------------------------------------------------------

# --- Core satellite imagery (Earth Search, free, no auth) ---

DatasetRegistry.register(
    DatasetDescriptor(
        id="earthsearch/sentinel-2-l2a",
        name="Sentinel-2 Level-2A",
        description="Multi-spectral optical imagery, 10-60m, global",
        stac_api="https://earth-search.aws.element84.com/v1",
        stac_collection="sentinel-2-l2a",
        band_map={
            "B01": "coastal",
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B05": "rededge1",
            "B06": "rededge2",
            "B07": "rededge3",
            "B08": "nir",
            "B8A": "nir08",
            "B09": "nir09",
            "B11": "swir16",
            "B12": "swir22",
            "SCL": "scl",
        },
        separate_files=True,
        spatial_coverage="global",
        temporal_range=("2015-06-23", "present"),
        license="proprietary",
        license_url="https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2024-06-01", "2024-07-15"),
        torchgeo_class="Sentinel2",
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="earthsearch/landsat-c2-l2",
        name="Landsat Collection 2 Level-2",
        description="Multi-spectral optical + thermal, 30m, global",
        stac_api="https://earth-search.aws.element84.com/v1",
        stac_collection="landsat-c2-l2",
        band_map={
            "B1": "coastal",
            "B2": "blue",
            "B3": "green",
            "B4": "red",
            "B5": "nir08",
            "B6": "swir16",
            "B7": "swir22",
            "qa_aerosol": "qa_aerosol",
            "qa_pixel": "qa_pixel",
            "qa_radsat": "qa_radsat",
        },
        separate_files=True,
        spatial_coverage="global",
        temporal_range=("1982-07-16", "present"),
        requires_auth=True,
        license="proprietary",
        license_url="https://www.usgs.gov/core-science-systems/hdds/data-policy",
        cloud_config={
            "provider": "aws",
            "requester_pays": True,
            "region": "us-west-2",
            "url_patterns": {
                "https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"
            },
        },
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2024-06-01", "2024-07-15"),
        torchgeo_class="Landsat9",
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="earthsearch/naip",
        name="NAIP",
        description="Aerial imagery, 1m, CONUS (beta: single multi-band COG)",
        stac_api="https://earth-search.aws.element84.com/v1",
        stac_collection="naip",
        band_map={"R": "image", "G": "image", "B": "image", "NIR": "image"},
        band_index_map={"R": 0, "G": 1, "B": 2, "NIR": 3},
        separate_files=False,
        spatial_coverage="north-america",
        temporal_range=("2010-01-01", "2023-12-31"),
        requires_auth=True,
        license="proprietary",
        license_url="https://www.fsa.usda.gov/help/policies-and-links/",
        cloud_config={
            "provider": "aws",
            "requester_pays": True,
            "region": "us-west-2",
            "url_patterns": {},
        },
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2022-01-01", "2022-12-31"),
        torchgeo_class="NAIP",
    )
)

# --- Elevation (Earth Search, free) ---

DatasetRegistry.register(
    DatasetDescriptor(
        id="earthsearch/cop-dem-glo-30",
        name="Copernicus DEM 30m",
        description="Global digital elevation model, 30m",
        stac_api="https://earth-search.aws.element84.com/v1",
        stac_collection="cop-dem-glo-30",
        band_map={"DEM": "data"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2021-04-22", "2021-04-22"),
        license="proprietary",
        license_url="https://spacedata.copernicus.eu/documents/20126/0/CSCDA_ESA_Mission-specific+Annex.pdf",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2021-04-01", "2021-05-01"),
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="earthsearch/cop-dem-glo-90",
        name="Copernicus DEM 90m",
        description="Global digital elevation model, 90m",
        stac_api="https://earth-search.aws.element84.com/v1",
        stac_collection="cop-dem-glo-90",
        band_map={"DEM": "data"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2021-04-22", "2021-04-22"),
        license="proprietary",
        license_url="https://spacedata.copernicus.eu/documents/20123/121286/CSCDA_ESA_Mission-specific+Annex_31_Oct_22.pdf",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2021-04-01", "2021-05-01"),
    )
)

# --- Planetary Computer datasets (Azure Blob, public via SAS signing) ---

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/sentinel-2-l2a",
        name="Sentinel-2 Level-2A (Planetary Computer)",
        description="Multi-spectral optical imagery, 10-60m, global (Azure mirror)",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="sentinel-2-l2a",
        band_map={
            "B01": "B01",
            "B02": "B02",
            "B03": "B03",
            "B04": "B04",
            "B05": "B05",
            "B06": "B06",
            "B07": "B07",
            "B08": "B08",
            "B8A": "B8A",
            "B09": "B09",
            "B11": "B11",
            "B12": "B12",
            "SCL": "SCL",
        },
        separate_files=True,
        spatial_coverage="global",
        temporal_range=("2015-06-23", "present"),
        requires_auth=True,
        license="proprietary",
        license_url="https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2024-06-01", "2024-07-15"),
        torchgeo_class="Sentinel2",
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/io-lulc-annual-v02",
        name="ESRI 10m Land Use/Land Cover",
        description="Annual global land use/land cover, 10m, 11 classes",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="io-lulc-annual-v02",
        band_map={"LULC": "data"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2017-01-01", "present"),
        requires_auth=True,
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2022-01-01", "2022-12-31"),
    )
)

# Register user-local descriptors from JSON (if present).
for _descriptor in load_local_descriptors():
    DatasetRegistry.register(_descriptor)

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/alos-dem",
        name="ALOS World 3D 30m DEM",
        description="Global digital elevation model, 30m, JAXA",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="alos-dem",
        band_map={"DEM": "data"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2021-01-01", "2021-01-01"),
        requires_auth=True,
        license="proprietary",
        license_url="https://earth.jaxa.jp/policy/en.html",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2016-12-01", "2016-12-31"),
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/nasadem",
        name="NASADEM",
        description="NASA DEM (SRTM-derived), 30m, near-global",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="nasadem",
        band_map={"DEM": "elevation"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2000-02-20", "2000-02-20"),
        requires_auth=True,
        license="proprietary",
        license_url="https://lpdaac.usgs.gov/data/data-citation-and-policies/",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2000-02-20", "2000-02-21"),
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/esa-worldcover",
        name="ESA WorldCover",
        description="Global land cover, 10m, 2020-2021",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="esa-worldcover",
        band_map={"MAP": "map"},
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2020-01-01", "2021-12-31"),
        requires_auth=True,
        license="CC-BY-4.0",
        license_url="https://spdx.org/licenses/CC-BY-4.0.html",
        example_bbox=(-122.45, 37.74, -122.35, 37.84),
        example_date_range=("2021-01-01", "2021-12-31"),
    )
)

DatasetRegistry.register(
    DatasetDescriptor(
        id="pc/usda-cdl",
        name="USDA Cropland Data Layer",
        description="Annual cropland classification, 30m, CONUS",
        stac_api="https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_collection="usda-cdl",
        band_map={"CDL": "cultivated"},
        separate_files=False,
        spatial_coverage="conus",
        temporal_range=("2008-01-01", "2024-12-31"),
        requires_auth=True,
        license="proprietary",
        license_url="https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#Section3_5.0",
        example_bbox=(-93.8, 41.9, -93.6, 42.1),
        example_date_range=("2021-01-01", "2021-12-31"),
    )
)

# --- Foundation model embeddings (Source Cooperative, free, no auth) ---

DatasetRegistry.register(
    DatasetDescriptor(
        id="aef/v1-annual",
        name="AlphaEarth Foundation Embeddings (Annual)",
        description="64-band int8 foundation-model embeddings, 10m, global",
        geoparquet_uri=(
            "s3://us-west-2.opendata.source.coop/"
            "tge-labs/aef/v1/annual/aef_index.parquet"
        ),
        column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
        href_column="path",
        band_index_map={f"A{i:02d}": i for i in range(64)},
        bbox_columns={
            "minx": "wgs84_west",
            "miny": "wgs84_south",
            "maxx": "wgs84_east",
            "maxy": "wgs84_north",
        },
        separate_files=False,
        spatial_coverage="global",
        temporal_range=("2018-01-01", "2023-12-31"),
        requires_auth=False,
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        cloud_config={
            "provider": "aws",
            "region": "us-west-2",
            "url_patterns": {
                "s3://us-west-2.opendata.source.coop/": ("https://data.source.coop/"),
            },
        },
        example_bbox=(11.3, -0.002, 11.5, 0.001),
        example_date_range=("2023-01-01", "2023-12-31"),
    )
)
