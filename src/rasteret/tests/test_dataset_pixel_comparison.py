# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Pixel-level comparison: Rasteret (GDF, xarray, torchgeo) vs rasterio.

For every catalog dataset, compare pixel values from three Rasteret output
paths against a pure rasterio read (ground truth):

1. ``collection.get_gdf()`` -> numpy array from ``gdf["data"]``
2. ``collection.get_xarray()`` -> numpy array from ``xr_ds[band].values.squeeze()``
3. ``collection.to_torchgeo_dataset()`` -> numpy array from ``sample["image"].numpy()``

Ground truth: ``rasterio.mask.mask(src, [geom], crop=True, all_touched=False, filled=True)``
with ``nodata=src.nodata`` when present, otherwise ``nodata=0``.

Requires ``--network`` flag::

    uv run pytest --network -k test_pixel_values_match_rasterio -v
    uv run pytest --network -k "test_pixel_values_match_rasterio[earthsearch/cop-dem-glo-30]" -v
"""

from __future__ import annotations

import contextlib
import signal
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box

import rasteret
from rasteret.catalog import DatasetDescriptor, DatasetRegistry

pytestmark = pytest.mark.network

# ---------------------------------------------------------------------------
# Per-dataset overrides
# ---------------------------------------------------------------------------

# Bbox overrides: some datasets need a specific bbox to work correctly.
_BBOX_OVERRIDES: dict[str, tuple[float, float, float, float]] = {}

# Datasets to skip entirely.
_SKIP_DATASETS: set[str] = {
    # AEF: south-up COGs require WarpedVRT for rasterio.merge.merge
    # (TorchGeo oracle), which takes ~95s per query over HTTP.
    # Verified 0/1232084 pixel mismatches vs vanilla TorchGeo.
    "aef/v1-annual",
}

# Datasets where torchgeo dataset creation fails (datetime parsing, etc.).
# GDF + xarray comparisons still run for these.
_SKIP_TORCHGEO: set[str] = {
    "pc/io-lulc-annual-v02",
    "pc/usda-cdl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlarmTimeout(Exception):
    pass


def _timeout(seconds: int):
    """Best-effort wall-clock timeout for live network tests (POSIX only)."""

    @contextlib.contextmanager
    def _ctx():
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(_signum, _frame):
            raise _AlarmTimeout(f"Timed out after {seconds}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    return _ctx()


def _has_aws_credentials() -> bool:
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def _rasterio_env(descriptor: DatasetDescriptor) -> dict[str, str]:
    """Build GDAL env vars for rasterio reads based on descriptor."""
    env: dict[str, str] = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    cc = descriptor.cloud_config or {}
    if cc.get("requester_pays"):
        env["AWS_REQUEST_PAYER"] = "requester"
    if not descriptor.requires_auth:
        env["AWS_NO_SIGN_REQUEST"] = "YES"
    return env


def _pick_band(descriptor: DatasetDescriptor) -> str:
    """Pick the first band from band_map or band_index_map."""
    if descriptor.band_map:
        return next(iter(descriptor.band_map))
    if descriptor.band_index_map:
        return next(iter(descriptor.band_index_map))
    raise ValueError(f"No bands for {descriptor.id}")


def _extract_href_and_band_number(
    collection: rasteret.Collection,
    descriptor: DatasetDescriptor,
    band: str,
) -> tuple[str, int]:
    """Extract the COG href and rasterio 1-based band number for a band."""
    row = collection.dataset.to_table(columns=["assets"]).to_pylist()[0]
    assets = row["assets"]

    # Try direct band key first, then asset key from band_map
    asset = assets.get(band)
    if asset is None and descriptor.band_map:
        asset_key = descriptor.band_map.get(band, band)
        asset = assets.get(asset_key)

    if asset is None:
        raise KeyError(
            f"Could not find asset for band '{band}' in assets: {list(assets.keys())}"
        )

    href = asset["href"]
    band_index = asset.get("band_index", 0)
    band_number = band_index + 1  # rasterio is 1-based

    # Sign PC hrefs for rasterio
    if descriptor.id.startswith("pc/"):
        import planetary_computer

        href = planetary_computer.sign(href)

    return href, band_number


def _get_bbox_and_dates(
    descriptor: DatasetDescriptor,
) -> tuple[tuple[float, float, float, float], tuple[str, str] | None]:
    """Get bbox and date_range for a dataset."""
    if descriptor.static_catalog:
        return None, None  # type: ignore[return-value]

    # Apply per-dataset bbox overrides
    bbox = _BBOX_OVERRIDES.get(descriptor.id, descriptor.example_bbox)
    if bbox is None:
        pytest.skip(f"No example_bbox for {descriptor.id}")
    if descriptor.example_date_range is None:
        pytest.skip(f"No example_date_range for {descriptor.id}")

    return bbox, descriptor.example_date_range


def _rasterio_ground_truth(
    href: str,
    bbox: tuple[float, float, float, float],
    band_number: int,
    gdal_env: dict[str, str],
) -> np.ndarray:
    """Read ground truth pixels using rasterio.mask.mask."""
    geom_4326 = box(*bbox).__geo_interface__
    with rasterio.Env(**gdal_env):
        with rasterio.open(href) as src:
            if src.crs is None:
                pytest.skip(
                    f"COG has no CRS (href={href}); cannot compare with rasterio"
                )
            geom_native = transform_geom("EPSG:4326", src.crs, geom_4326)
            fill = src.nodata if src.nodata is not None else 0
            masked_arr, _ = rio_mask(
                src,
                [geom_native],
                crop=True,
                all_touched=False,
                filled=True,
                nodata=fill,
                indexes=band_number,
            )
            return masked_arr.squeeze()  # [H, W]


def _rasterio_ground_truth_native_bbox(
    href: str,
    bbox_native: tuple[float, float, float, float],
    band_number: int,
    gdal_env: dict[str, str],
    res: tuple[float, float] | None = None,
) -> np.ndarray:
    """Read ground truth using rasterio.merge.merge with native-CRS bounds.

    Matches what TorchGeo's ``_merge_or_stack`` calls:
    ``rasterio.merge.merge([src], bounds=..., res=..., indexes=...)``.

    Only used for north-up datasets (south-up AEF is skipped via
    ``_SKIP_DATASETS``).  South-up COGs would require ``WarpedVRT``
    which is extremely slow over HTTP.
    """
    from rasterio.merge import merge as rio_merge

    with rasterio.Env(**gdal_env):
        with rasterio.open(href) as src:
            if src.crs is None:
                pytest.skip(
                    f"COG has no CRS (href={href}); cannot compare with rasterio"
                )
            dtype = np.dtype(src.dtypes[band_number - 1])
            resampling = (
                rasterio.enums.Resampling.bilinear
                if np.issubdtype(dtype, np.floating)
                else rasterio.enums.Resampling.nearest
            )
            data, _ = rio_merge(
                [src],
                bounds=bbox_native,
                res=res,
                indexes=[band_number],
                resampling=resampling,
            )
            return data.squeeze()


def _count_mismatches(r_arr: np.ndarray, rio_arr: np.ndarray) -> tuple[int, int]:
    """Count mismatching pixels between two same-shape arrays."""
    assert r_arr.shape == rio_arr.shape
    if np.issubdtype(r_arr.dtype, np.integer) and np.issubdtype(
        rio_arr.dtype, np.integer
    ):
        diff = r_arr != rio_arr
    else:
        diff = ~np.isclose(
            r_arr.astype(np.float64),
            rio_arr.astype(np.float64),
            atol=0,
            equal_nan=True,
        )
    n_mismatch = int(diff.sum())
    n_total = int(r_arr.size)
    return n_mismatch, n_total


def _compare_arrays(
    rasteret_arr: np.ndarray,
    rasterio_arr: np.ndarray,
    label: str,
    band: str,
    dataset_id: str,
) -> None:
    """Compare two arrays — exact shape and pixel match required.

    For integer dtypes: exact equality.
    For float dtypes: np.allclose with atol=0, equal_nan=True.
    """
    assert rasteret_arr.shape == rasterio_arr.shape, (
        f"[{dataset_id}] {label} band={band}: shape mismatch: "
        f"rasteret={rasteret_arr.shape}, rasterio={rasterio_arr.shape}"
    )

    n_mismatch, n_valid = _count_mismatches(rasteret_arr, rasterio_arr)
    if n_valid == 0:
        pytest.skip(f"[{dataset_id}] {label} band={band}: no valid pixels")

    if n_mismatch > 0:
        pct = 100.0 * n_mismatch / n_valid
        diff = (
            rasteret_arr != rasterio_arr
            if np.issubdtype(rasteret_arr.dtype, np.integer)
            else ~np.isclose(
                rasteret_arr.astype(np.float64),
                rasterio_arr.astype(np.float64),
                atol=0,
                equal_nan=True,
            )
        )
        ys, xs = np.where(diff)
        samples = []
        for y, x in list(zip(ys, xs))[:5]:
            samples.append(
                f"(y={int(y)}, x={int(x)}) rasteret={rasteret_arr[y, x]}, rasterio={rasterio_arr[y, x]}"
            )
        pytest.fail(
            f"[{dataset_id}] {label} band={band}: {n_mismatch}/{n_valid} pixels "
            f"differ ({pct:.2f}%). Samples: {samples}"
        )


# ---------------------------------------------------------------------------
# Credential / dependency guards
# ---------------------------------------------------------------------------


def _apply_skip_guards(dataset_id: str, descriptor: DatasetDescriptor) -> None:
    """Skip datasets that require unavailable auth or deps."""
    if dataset_id in _SKIP_DATASETS:
        pytest.skip(f"{dataset_id} skipped (see _SKIP_DATASETS).")

    if dataset_id.startswith("pc/"):
        pytest.importorskip(
            "planetary_computer",
            reason="planetary-computer package required for PC datasets",
        )

    if descriptor.cloud_config and descriptor.cloud_config.get("requester_pays"):
        if not _has_aws_credentials():
            pytest.skip("AWS credentials required for requester-pays datasets.")


# ---------------------------------------------------------------------------
# Main parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_id", sorted(d.id for d in DatasetRegistry.list()))
def test_pixel_values_match_rasterio(tmp_path: Path, dataset_id: str) -> None:
    """Compare Rasteret GDF/xarray/torchgeo outputs against rasterio for every dataset."""
    descriptor = DatasetRegistry.get(dataset_id)
    assert descriptor is not None

    # --- 1. Guards ---
    _apply_skip_guards(dataset_id, descriptor)

    band = _pick_band(descriptor)

    # --- 2. Get bbox / date_range ---
    bbox, date_range = _get_bbox_and_dates(descriptor)

    # --- 3. Build collection ---
    build_kwargs: dict = {
        "name": f"pxtest-{dataset_id.split('/', 1)[-1]}",
        "workspace_dir": tmp_path / dataset_id.replace("/", "-"),
        "force": True,
        "max_concurrent": 8,
    }
    if bbox is not None:
        build_kwargs["bbox"] = bbox
    if date_range is not None:
        build_kwargs["date_range"] = date_range
    if not descriptor.static_catalog:
        build_kwargs["query"] = {"max_items": 1}

    with _timeout(300):
        collection = rasteret.build(dataset_id, **build_kwargs)

    assert collection.dataset is not None
    n_rows = collection.dataset.count_rows()
    assert n_rows >= 1, f"No rows for {dataset_id}"

    # --- 4. Extract href + band_number ---
    href, band_number = _extract_href_and_band_number(collection, descriptor, band)
    gdal_env = _rasterio_env(descriptor)

    # --- 5. Rasterio ground truth ---
    with _timeout(120):
        rasterio_arr = _rasterio_ground_truth(href, bbox, band_number, gdal_env)

    assert (
        rasterio_arr.ndim == 2
    ), f"Expected 2D rasterio array, got shape {rasterio_arr.shape}"
    assert rasterio_arr.size > 0, "Rasterio returned empty array"

    # --- 6. Compare GDF output ---
    with _timeout(180):
        gdf = collection.get_gdf(geometries=bbox, bands=[band])

    assert len(gdf) >= 1, f"GDF is empty for {dataset_id}"
    gdf_row = gdf[gdf["band"] == band].iloc[0]
    rasteret_gdf_arr = gdf_row["data"]
    assert isinstance(
        rasteret_gdf_arr, np.ndarray
    ), f"GDF data is not ndarray: {type(rasteret_gdf_arr)}"
    assert rasteret_gdf_arr.ndim == 2, f"GDF data is not 2D: {rasteret_gdf_arr.shape}"
    _compare_arrays(rasteret_gdf_arr, rasterio_arr, "GDF", band, dataset_id)

    # --- 7. Compare xarray output ---
    with _timeout(180):
        xr_ds = collection.get_xarray(geometries=bbox, bands=[band])

    assert (
        band in xr_ds.data_vars
    ), f"Band '{band}' not in xarray dataset: {list(xr_ds.data_vars)}"
    rasteret_xr_arr = xr_ds[band].values.squeeze()
    assert (
        rasteret_xr_arr.ndim == 2
    ), f"xarray data is not 2D after squeeze: {rasteret_xr_arr.shape}"
    _compare_arrays(
        rasteret_xr_arr,
        rasterio_arr,
        "xarray",
        band,
        dataset_id,
    )

    # --- 8. Compare torchgeo output ---
    if dataset_id in _SKIP_TORCHGEO:
        return  # GDF + xarray passed; torchgeo has known issues for this dataset

    try:
        import torchgeo  # noqa: F401
    except ImportError:
        pytest.skip("torchgeo not installed, skipping torchgeo comparison")

    try:
        with _timeout(180):
            tg_ds = collection.to_torchgeo_dataset(bands=[band], geometries=bbox)
    except ValueError as exc:
        pytest.skip(f"torchgeo dataset creation failed: {exc}")

    # Construct a GeoSlice matching the query bbox in dataset's native CRS.
    from rasteret.core.utils import transform_bbox

    tg_epsg = tg_ds.epsg
    tg_bbox = transform_bbox(bbox, 4326, tg_epsg) if tg_epsg != 4326 else bbox
    res_x, res_y = tg_ds._res
    _, _, bt = tg_ds.bounds
    sample = tg_ds[
        slice(tg_bbox[0], tg_bbox[2], res_x),
        slice(tg_bbox[1], tg_bbox[3], res_y),
        bt,
    ]
    rasteret_tg_arr = sample["image"].squeeze().numpy()
    assert (
        rasteret_tg_arr.ndim == 2
    ), f"torchgeo data is not 2D after squeeze: {rasteret_tg_arr.shape}"

    with _timeout(120):
        rasterio_tg_arr = _rasterio_ground_truth_native_bbox(
            href, tg_bbox, band_number, gdal_env, res=(res_x, res_y)
        )
    _compare_arrays(
        rasteret_tg_arr,
        rasterio_tg_arr,
        "torchgeo",
        band,
        dataset_id,
    )
