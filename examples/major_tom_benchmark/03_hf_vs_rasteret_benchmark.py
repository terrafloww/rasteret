#!/usr/bin/env python3
"""Throughput benchmark: HF Datasets (images-in-Parquet) vs Rasteret (COG reads).

Compare two ways to obtain the *same* Major TOM patches identified by
(`product_id`, `grid_cell`, `timestamp`):

1) Hugging Face `datasets` streaming Parquet reads, where each band column stores
   a small GeoTIFF blob
2) Rasteret reading pixels from source COGs using a prebuilt index
   (`Collection.get_numpy`)

This benchmark reports a stage breakdown for HF (example fetch vs GeoTIFF decode)
and an end-to-end time for Rasteret.

Methodology notes
-----------------
- HF mode used here is streaming-only:
  - `datasets_streaming`: `load_dataset(..., streaming=True)` and iterator access.
- Stage timings are split into:
  - `hf_dataset_init_s`: dataset creation/materialization setup
  - `hf_next_example_s`: row access from HF dataset object
  - `hf_tif_decode_s`: local GeoTIFF blob decode via rasterio `MemoryFile`
- CRS diversity:
  - Use `--min-unique-epsg` to force multi-UTM/multi-CRS sample sets.
  - Rasteret reads patch AOIs via `Collection.get_numpy` grouped by `product_id`.

References
----------
- HF Datasets streaming docs:
  https://huggingface.co/docs/datasets/dataset_streaming
- HF `load_dataset` reference (`streaming`, `columns`, `filters`):
  https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods
- Major TOM project (metadata + parquet row-group access pattern):
  https://github.com/ESA-PhiLab/Major-TOM
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow as pa
import pyarrow.dataset as pads
import shapely
from pyproj import Transformer
from rasterio.io import MemoryFile

import rasteret

META_HF_PATH = "hf://datasets/Major-TOM/Core-S2L2A/metadata.parquet"
PATCH_SIZE_PX = 1068
PATCH_RES_M = 10.0
HALF_PATCH_METERS = (PATCH_SIZE_PX * PATCH_RES_M) / 2.0
BAND_PRESETS: dict[str, list[str]] = {
    "s2_10m_rgbnir": ["B02", "B03", "B04", "B08"],
    "s2_20m_swirrededge": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "s2_60m_atmo": ["B01", "B09"],
}


def _resolve_hf_dataset_file(path: str) -> str:
    """Resolve a `hf://datasets/...` file path to a local cached file path.

    Using PyArrow directly against `hf://` remote files can be unstable across
    environments (e.g., depending on pyarrow/fsspec versions). For benchmark
    repeatability, we download/resolve HF-hosted Parquet files to a local path
    and then let PyArrow operate on the local file.
    """

    prefix = "hf://datasets/"
    if not path.startswith(prefix):
        return path

    parts = path.removeprefix(prefix).split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HF datasets path: {path!r}. Expected: hf://datasets/<org>/<name>/<file>"
        )

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing Hugging Face dependency. Install with: pip install datasets"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HF datasets (images-in-Parquet) vs Rasteret (COG reads)"
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=None,
        help=(
            "Path to prebuilt Rasteret collection directory "
            "(must include major_tom_product_id and requested band metadata columns)."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default=META_HF_PATH,
        help="Major TOM metadata parquet path (hf://... or local path)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Number of matched patches to benchmark",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["head", "random"],
        default="head",
        help="How to choose candidate metadata rows before geometry checks",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when --sample-strategy random",
    )
    parser.add_argument(
        "--min-unique-epsg",
        type=int,
        default=1,
        help="Require at least this many distinct EPSG codes in sampled patches.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B02", "B08"],
        help="Bands to decode in both paths",
    )
    parser.add_argument(
        "--max-nodata-frac",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed nodata fraction in Major TOM metadata (0..1). "
            "Use 0.0 for strict full-coverage patches."
        ),
    )
    parser.add_argument(
        "--band-preset",
        choices=sorted(BAND_PRESETS),
        default="",
        help=(
            "Optional Sentinel-2 preset for same-resolution band groups. "
            "Overrides --bands when set."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=150,
        help="Rasteret max_concurrent for get_numpy",
    )
    parser.add_argument(
        "--hf-range-size-mib",
        type=int,
        default=128,
        help="HF/Arrow Parquet prefetch range size in MiB (default: 128).",
    )
    parser.add_argument(
        "--hf-prefetch-limit",
        type=int,
        default=1,
        help="HF/Arrow Parquet prefetch limit (default: 1).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output JSON path for machine-readable benchmark results.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        default=False,
        help="Skip Hugging Face datasets path (run Rasteret only).",
    )
    parser.add_argument(
        "--skip-rasteret",
        action="store_true",
        default=False,
        help="Skip Rasteret path (run HF only).",
    )
    return parser.parse_args()


def _build_patch_geometries(
    centre_lon: np.ndarray,
    centre_lat: np.ndarray,
    crs: np.ndarray,
) -> np.ndarray:
    """Vectorized patch geometry construction, grouped by CRS.

    Returns geometries in WGS84 for containment checks and in the source
    projected CRS for Rasteret reads (avoids a WGS84 round-trip drift that
    can produce ragged pixel shapes at scale).
    """
    wgs84_result = np.empty(len(centre_lon), dtype=object)
    projected_result = np.empty(len(centre_lon), dtype=object)
    for crs_val in np.unique(crs):
        mask = crs == crs_val
        epsg = int(str(crs_val).split(":")[1])
        to_projected = Transformer.from_crs(4326, epsg, always_xy=True)
        to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
        x_center, y_center = to_projected.transform(centre_lon[mask], centre_lat[mask])
        # Snap to the patch pixel grid to avoid +/-1 pixel drift from floating
        # point projection math. Major TOM patches are defined on a fixed 10m
        # grid; snapping yields stable windows and avoids ragged shapes at N=1000+.
        x_center = (
            np.round(np.asarray(x_center, dtype=np.float64) / PATCH_RES_M) * PATCH_RES_M
        )
        y_center = (
            np.round(np.asarray(y_center, dtype=np.float64) / PATCH_RES_M) * PATCH_RES_M
        )
        projected_boxes = shapely.box(
            x_center - HALF_PATCH_METERS,
            y_center - HALF_PATCH_METERS,
            x_center + HALF_PATCH_METERS,
            y_center + HALF_PATCH_METERS,
        )
        wgs84_boxes = shapely.transform(
            projected_boxes,
            lambda coords, _t=to_wgs84: np.column_stack(
                _t.transform(coords[:, 0], coords[:, 1])
            ),
        )
        wgs84_result[mask] = wgs84_boxes
        projected_result[mask] = projected_boxes
    return wgs84_result, projected_result


def decode_tif_bytes(
    raw_bytes: bytes | memoryview,
) -> tuple[np.ndarray, object, object]:
    """Decode GeoTIFF bytes + capture georeferencing.

    Returns ``(array, transform, crs)``.
    """
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    with MemoryFile(raw_bytes) as mem_file:
        with mem_file.open() as dataset:
            array = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
    return array, transform, crs


def _decode_hf_example_bands(
    *,
    example: dict[str, object],
    bands: list[str],
) -> tuple[dict[str, object], int, int, float]:
    """Decode requested band blobs from one HF example.

    Returns:
        grid metadata, total pixels, total bytes, decode seconds.
    """
    total_pixels = 0
    total_bytes = 0
    decode_s = 0.0
    first_transform = None
    first_crs = None
    first_shape: tuple[int, int] | None = None
    first_res: tuple[float, float] | None = None

    for band in bands:
        t_decode = time.perf_counter()
        array, tf, crs = decode_tif_bytes(example[band])
        decode_s += time.perf_counter() - t_decode

        total_pixels += int(array.size)
        total_bytes += int(array.nbytes)
        if first_transform is None:
            first_transform = tf
            first_crs = crs
            first_shape = (int(array.shape[0]), int(array.shape[1]))
            first_res = (abs(float(tf.a)), abs(float(tf.e)))
            continue

        # Major TOM stores some bands at different native resolutions.
        # We require same-grid bands for fair per-pixel comparison.
        res = (abs(float(tf.a)), abs(float(tf.e)))
        shape = (int(array.shape[0]), int(array.shape[1]))
        if first_res is None or first_shape is None:
            raise RuntimeError("Internal error: missing first band grid info")
        if res != first_res or shape != first_shape:
            raise RuntimeError(
                "Requested bands have different grids in Major TOM. "
                f"First band grid res={first_res}, shape={first_shape}; "
                f"band '{band}' has res={res}, shape={shape}. "
                "Run the benchmark with same-resolution bands only."
            )

    epsg = None
    if first_crs is not None and hasattr(first_crs, "to_epsg"):
        epsg = first_crs.to_epsg()
    if epsg is None:
        raise RuntimeError("HF example CRS missing EPSG; cannot reproduce patch grid")
    if first_transform is None or first_shape is None:
        raise RuntimeError(
            "HF example missing transform/shape; cannot reproduce patch grid"
        )

    # Compute bounds from transform + shape to avoid floating drift.
    # rasterio Affine: a, b, c, d, e, f where north-up has b=d=0, e<0.
    a = float(first_transform.a)
    e = float(first_transform.e)
    left = float(first_transform.c)
    top = float(first_transform.f)
    height, width = first_shape
    right = left + width * a
    bottom = top + height * e
    bounds = (left, min(bottom, top), right, max(bottom, top))

    grid = {
        "epsg": int(epsg),
        "bounds": bounds,
        "res": (abs(a), abs(e)),
        "shape": (height, width),
    }
    return grid, total_pixels, total_bytes, decode_s


def _iter_hf_examples(
    *,
    dataset_or_stream: object,
    limit: int,
):
    """Yield `(example, access_seconds)` from HF streaming iterator."""
    iterator = iter(dataset_or_stream)
    for _ in range(limit):
        t_access = time.perf_counter()
        try:
            example = next(iterator)
        except StopIteration:
            return
        yield example, time.perf_counter() - t_access


def _resolve_bands(args: argparse.Namespace) -> list[str]:
    if args.band_preset:
        return BAND_PRESETS[str(args.band_preset)]
    return [str(b) for b in args.bands]


def load_matched_sample_rows(
    *,
    rasteret,
    collection_path: Path,
    metadata_path: str,
    samples: int,
    sample_strategy: str,
    random_seed: int,
    min_unique_epsg: int,
    max_nodata_frac: float,
) -> pd.DataFrame:
    collection = rasteret.load(collection_path)
    if "major_tom_product_id" not in collection.dataset.schema.names:
        raise ValueError(
            "Collection missing 'major_tom_product_id'. Rebuild with MTOM enrichment script."
        )

    required_scene_cols = ["major_tom_product_id", "geometry"]
    if "proj:epsg" in collection.dataset.schema.names:
        required_scene_cols.append("proj:epsg")
    if "B02_metadata" in collection.dataset.schema.names:
        required_scene_cols.append("B02_metadata")

    table = collection.dataset.to_table(columns=required_scene_cols)
    rows = table.to_pylist()

    scene_by_product: dict[str, object] = {}
    scene_epsg_by_product: dict[str, int | None] = {}
    raster_bounds_by_product: dict[str, tuple[float, float, float, float]] = {}

    for row in rows:
        product_id = row.get("major_tom_product_id")
        geometry_wkb = row.get("geometry")
        if not product_id or not geometry_wkb:
            continue
        product_key = str(product_id)
        if product_key in scene_by_product:
            continue

        scene_by_product[product_key] = shapely.from_wkb(geometry_wkb)
        scene_epsg_by_product[product_key] = (
            int(row["proj:epsg"]) if row.get("proj:epsg") is not None else None
        )
        meta = row.get("B02_metadata")
        if (
            isinstance(meta, dict)
            and meta.get("transform")
            and meta.get("image_width")
            and meta.get("image_height")
        ):
            sx, tx, sy, ty = (
                float(meta["transform"][0]),
                float(meta["transform"][1]),
                float(meta["transform"][2]),
                float(meta["transform"][3]),
            )
            width = int(meta["image_width"])
            height = int(meta["image_height"])
            xmin = tx
            ymax = ty
            xmax = tx + width * sx
            ymin = ty + height * sy
            raster_bounds_by_product[product_key] = (
                min(xmin, xmax),
                min(ymin, ymax),
                max(xmin, xmax),
                max(ymin, ymax),
            )

    product_ids = set(scene_by_product)

    # HF pushdown path:
    # - restrict to product ids present in the local Rasteret collection
    # - restrict nodata fraction
    # - require non-null key fields
    # This avoids a blind `head()` over global metadata ordering.
    required_cols = [
        "product_id",
        "grid_cell",
        "timestamp",
        "centre_lat",
        "centre_lon",
        "nodata",
        "crs",
        "parquet_url",
        "parquet_row",
    ]
    metadata_local = _resolve_hf_dataset_file(metadata_path)
    metadata_ds = pads.dataset(metadata_local, format="parquet")
    filter_expr = (pads.field("nodata") <= float(max_nodata_frac)) & pads.field(
        "product_id"
    ).isin(sorted(product_ids))
    for col in [
        "grid_cell",
        "timestamp",
        "centre_lat",
        "centre_lon",
        "crs",
        "parquet_url",
        "parquet_row",
    ]:
        filter_expr = filter_expr & pads.field(col).is_valid()

    meta = metadata_ds.to_table(columns=required_cols, filter=filter_expr).to_pandas()
    if meta.empty:
        raise RuntimeError(
            "No HF metadata rows matched product_ids from the provided Rasteret collection. "
            "This benchmark expects a Rasteret collection derived from Major TOM keys "
            "(collection must have a compatible 'major_tom_product_id')."
        )
    if sample_strategy == "random":
        meta = meta.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    else:
        meta = meta.reset_index(drop=True)

    crs_values = meta["crs"].astype(str)
    patch_geometries_wgs84, patch_geometries_proj = _build_patch_geometries(
        centre_lon=meta["centre_lon"].to_numpy(dtype=np.float64),
        centre_lat=meta["centre_lat"].to_numpy(dtype=np.float64),
        crs=crs_values.to_numpy(),
    )
    patch_wkb = shapely.to_wkb(patch_geometries_proj)

    selected_rows: list[dict[str, object]] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def _try_add_index(index: int) -> bool:
        row = meta.iloc[index]
        product_id = str(row.product_id)
        grid_cell = str(row.grid_cell)
        timestamp = str(row.timestamp)
        key = (product_id, grid_cell, timestamp)
        if key in selected_keys:
            return False

        scene = scene_by_product.get(product_id)
        if scene is None:
            return False

        if not scene.contains(patch_geometries_wgs84[index]):
            center = shapely.Point(float(row.centre_lon), float(row.centre_lat))
            if not scene.contains(center):
                return False
        # Enforce full pixel window inside the raster extent (prevents ragged
        # shapes caused by edge clipping when patches are near scene borders).
        raster_bounds = raster_bounds_by_product.get(product_id)
        if raster_bounds is not None:
            patch_bounds = patch_geometries_proj[index].bounds
            epsg = int(str(row.crs).split(":")[1])
            scene_epsg = scene_epsg_by_product.get(product_id)
            if scene_epsg is not None and int(scene_epsg) != epsg:
                return False
            x0, y0, x1, y1 = (
                float(patch_bounds[0]),
                float(patch_bounds[1]),
                float(patch_bounds[2]),
                float(patch_bounds[3]),
            )
            rb0, rb1, rb2, rb3 = raster_bounds
            margin = float(PATCH_RES_M)
            if not (
                x0 >= rb0 + margin
                and y0 >= rb1 + margin
                and x1 <= rb2 - margin
                and y1 <= rb3 - margin
            ):
                return False

        selected_keys.add(key)
        selected_rows.append(
            {
                "product_id": product_id,
                "grid_cell": grid_cell,
                "timestamp": timestamp,
                "centre_lat": float(row.centre_lat),
                "centre_lon": float(row.centre_lon),
                "crs": str(row.crs),
                "geometry_wkb": patch_wkb[index],
                "parquet_url": str(row.parquet_url),
                "parquet_row": int(row.parquet_row),
            }
        )
        return True

    # Phase 1: one valid sample per EPSG when CRS diversity is requested.
    required_epsg = max(1, int(min_unique_epsg))
    if required_epsg > 1:
        target_epsg = crs_values.drop_duplicates().tolist()[:required_epsg]
        for epsg in target_epsg:
            if len(selected_rows) >= samples:
                break
            epsg_indices = np.flatnonzero(crs_values.to_numpy() == epsg)
            for index in epsg_indices:
                if _try_add_index(int(index)):
                    break

    # Phase 2: fill remaining slots using global ordering.
    for index in range(len(meta)):
        if len(selected_rows) >= samples:
            break
        _try_add_index(index)

    if len(selected_rows) < samples:
        raise RuntimeError(
            f"Insufficient stable full-size patches after containment filter: "
            f"{len(selected_rows)} < requested {samples}. "
            "Increase available scenes in the collection."
        )
    if len({r["crs"] for r in selected_rows}) < required_epsg:
        raise RuntimeError(
            f"CRS diversity requirement not met: "
            f"found {len({r['crs'] for r in selected_rows})}, required {required_epsg}. "
            "Lower --min-unique-epsg or increase available scenes in the collection."
        )
    return pd.DataFrame(selected_rows)


def benchmark_hf_datasets_streaming(
    *,
    sample_rows: pd.DataFrame,
    bands: list[str],
    hf_range_size_mib: int,
    hf_prefetch_limit: int,
) -> dict[str, float | int]:
    """Benchmark HF parquet-backed reads with streaming mode only."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    data_files = {
        "train": sorted(set(sample_rows["parquet_url"].astype(str).to_list()))
    }
    filters = sample_rows.apply(
        lambda row: [
            ("product_id", "==", str(row["product_id"])),
            ("grid_cell", "==", str(row["grid_cell"])),
            ("product_datetime", "==", str(row["timestamp"])),
        ],
        axis=1,
    ).to_list()
    columns = list(
        dict.fromkeys([*bands, "product_id", "grid_cell", "product_datetime"])
    )
    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=int(hf_prefetch_limit),
            range_size_limit=int(hf_range_size_mib) << 20,
        ),
    )

    t_init0 = time.perf_counter()
    dataset_or_stream = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True,
        token=token,
        columns=columns,
        filters=filters,
        fragment_scan_options=fragment_scan_options,
    )
    hf_dataset_init_s = time.perf_counter() - t_init0

    total_bytes = 0
    total_pixels = 0
    matched = 0
    hf_next_example_s = 0.0
    hf_tif_decode_s = 0.0

    start = time.perf_counter()
    for example, access_s in _iter_hf_examples(
        dataset_or_stream=dataset_or_stream,
        limit=len(sample_rows),
    ):
        hf_next_example_s += access_s
        _grid, pixels, byte_count, decode_s = _decode_hf_example_bands(
            example=example,
            bands=bands,
        )

        total_pixels += pixels
        total_bytes += byte_count
        hf_tif_decode_s += decode_s
        matched += 1

    elapsed = time.perf_counter() - start
    if matched < len(sample_rows):
        raise RuntimeError(
            f"HF datasets filters returned fewer rows than requested: "
            f"{matched} < {len(sample_rows)}."
        )
    stats: dict[str, float | int] = {
        "elapsed_s": elapsed,
        "decoded_pixels": total_pixels,
        "decoded_bytes": total_bytes,
        "matched": len(sample_rows),
        "scanned": matched,
        "hf_dataset_init_s": hf_dataset_init_s,
        "hf_next_example_s": hf_next_example_s,
        "hf_tif_decode_s": hf_tif_decode_s,
        "hf_mode": "datasets_streaming",
        "hf_range_size_mib": int(hf_range_size_mib),
        "hf_prefetch_limit": int(hf_prefetch_limit),
    }
    return stats


def benchmark_rasteret_index_get_numpy(
    *,
    rasteret,
    collection_path: Path,
    sample_rows: pd.DataFrame,
    bands: list[str],
    max_concurrent: int,
) -> dict[str, float | int]:
    """Use the prebuilt Rasteret collection and read patch AOIs via get_numpy.

    This is the "fast path" that does not attempt to match HF's exact patch grid.
    """
    collection = rasteret.load(collection_path)
    geometry_array = pa.array(sample_rows["geometry_wkb"], type=pa.binary())
    product_id_values = sample_rows["product_id"].astype(str)

    start = time.perf_counter()
    total_bytes = 0
    total_pixels = 0
    grouped_indices = product_id_values.groupby(product_id_values, sort=False).indices
    for product_id, index_list in grouped_indices.items():
        scene_view = collection.where(
            pads.field("major_tom_product_id") == str(product_id)
        )
        crs_values = sample_rows.loc[index_list, "crs"].astype(str).unique().tolist()
        if len(crs_values) != 1:
            raise RuntimeError(
                "Expected a single CRS per product_id group, found: "
                f"product_id={product_id!r}, crs={crs_values}."
            )
        epsg = int(str(crs_values[0]).split(":")[1])
        geometries = pa.compute.take(
            geometry_array,
            pa.array(index_list, type=pa.int64()),
        )
        array = scene_view.get_numpy(
            geometries=geometries,
            bands=bands,
            max_concurrent=max_concurrent,
            geometry_crs=epsg,
        )
        total_pixels += int(array.size)
        total_bytes += int(array.size * array.dtype.itemsize)

    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "decoded_pixels": total_pixels,
        "decoded_bytes": total_bytes,
        "matched": len(sample_rows),
        "scanned": len(sample_rows),
    }


def _validate_collection_contract(
    *,
    rasteret,
    collection_path: Path,
    bands: list[str],
) -> None:
    """Fail fast when benchmark collection contract is not satisfied."""
    collection = rasteret.load(collection_path)
    if collection.dataset is None:
        raise RuntimeError("Invalid benchmark collection: missing dataset.")
    if "major_tom_product_id" not in collection.dataset.schema.names:
        raise RuntimeError(
            "Collection missing 'major_tom_product_id'. "
            "Use the Major TOM enrichment flow before benchmarking."
        )
    available = set(collection.bands)
    missing = [band for band in bands if band not in available]
    if missing:
        raise RuntimeError(
            "Requested bands are not present in the benchmark collection. "
            f"missing={missing}, available={sorted(available)}. "
            "Rebuild the collection with a richer band set before running this preset."
        )


def print_summary(name: str, stats: dict[str, float | int]) -> None:
    elapsed = float(stats["elapsed_s"])
    decoded_bytes = int(stats["decoded_bytes"])
    matched = int(stats["matched"])
    init_s = (
        float(stats.get("hf_dataset_init_s", 0.0)) if name.startswith("hf_") else 0.0
    )
    total_elapsed = elapsed + init_s
    mib_s = (decoded_bytes / (1024 * 1024)) / max(elapsed, 1e-9)
    patch_s = matched / max(elapsed, 1e-9)
    print(f"\n[{name}]")
    print(f"matched={matched}, scanned={int(stats['scanned'])}")
    print(f"elapsed_s={elapsed:.2f}")
    if name.startswith("hf_"):
        print(f"elapsed_total_s={total_elapsed:.2f}")
    print(f"decoded_pixels={int(stats['decoded_pixels']):,}")
    print(f"throughput_mib_s={mib_s:.2f}")
    print(f"patches_per_sec={patch_s:.2f}")
    if name.startswith("hf_"):
        print("stage_seconds:")
        print(f"  dataset_init={float(stats.get('hf_dataset_init_s', 0.0)):.2f}")
        print(f"  next_example={float(stats.get('hf_next_example_s', 0.0)):.2f}")
        print(f"  tif_decode={float(stats.get('hf_tif_decode_s', 0.0)):.2f}")
        # Conversion is a no-op in this script (arrays stay as numpy).


def main() -> None:
    args = parse_args()
    if args.collection_path is None:
        print(
            "Skipping benchmark: --collection-path is required for this example. "
            "Provide a prebuilt Major TOM Rasteret collection to run it."
        )
        return
    bands = _resolve_bands(args)

    # Benchmarks should be quiet by default.
    rasteret.set_options(progress=False)
    _validate_collection_contract(
        rasteret=rasteret,
        collection_path=args.collection_path,
        bands=bands,
    )

    sample_rows = load_matched_sample_rows(
        rasteret=rasteret,
        collection_path=args.collection_path,
        metadata_path=args.metadata_path,
        samples=args.samples,
        sample_strategy=args.sample_strategy,
        random_seed=args.random_seed,
        min_unique_epsg=args.min_unique_epsg,
        max_nodata_frac=args.max_nodata_frac,
    )
    epsg_counts = sample_rows["crs"].astype(str).value_counts().to_dict()
    print(f"sample_rows={len(sample_rows)}")
    print(f"bands={bands}")
    if args.band_preset:
        print(f"band_preset={args.band_preset}")
    print(f"unique_epsg={len(epsg_counts)} epsg_counts={epsg_counts}")
    print(f"collection_path={args.collection_path}")
    print("hf_mode=datasets_streaming")
    print("rasteret_mode=index_get_numpy")

    hf_stats = None
    rasteret_stats = None
    hf_name = "hf_datasets_streaming"
    rasteret_name = "rasteret_index_get_numpy"

    if not args.skip_hf:
        hf_stats = benchmark_hf_datasets_streaming(
            sample_rows=sample_rows,
            bands=bands,
            hf_range_size_mib=int(args.hf_range_size_mib),
            hf_prefetch_limit=int(args.hf_prefetch_limit),
        )
        print_summary(hf_name, hf_stats)

    if not args.skip_rasteret:
        rasteret_stats = benchmark_rasteret_index_get_numpy(
            rasteret=rasteret,
            collection_path=args.collection_path,
            sample_rows=sample_rows,
            bands=bands,
            max_concurrent=args.max_concurrent,
        )
        print_summary(rasteret_name, rasteret_stats)

    winner = None
    speedup = None
    if hf_stats is not None and rasteret_stats is not None:
        hf_elapsed = float(hf_stats["elapsed_s"])
        rasteret_elapsed = float(rasteret_stats["elapsed_s"])
        if rasteret_elapsed < hf_elapsed:
            speedup = hf_elapsed / max(rasteret_elapsed, 1e-9)
            print(f"\nrasteret_faster_by={speedup:.2f}x")
            winner = "rasteret"
        else:
            speedup = rasteret_elapsed / max(hf_elapsed, 1e-9)
            print(f"\nhf_streaming_faster_by={speedup:.2f}x")
            winner = "hf_streaming"

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "samples": int(len(sample_rows)),
            "bands": list(bands),
            "collection_path": str(args.collection_path),
            "metadata_path": str(args.metadata_path),
            "sample_strategy": str(args.sample_strategy),
            "random_seed": int(args.random_seed),
            "min_unique_epsg": int(args.min_unique_epsg),
            "unique_epsg": int(sample_rows["crs"].astype(str).nunique()),
            "rasteret_mode": "index_get_numpy",
            "hf": None
            if hf_stats is None
            else {
                "name": hf_name,
                "elapsed_s": float(hf_stats["elapsed_s"]),
                "matched": int(hf_stats["matched"]),
                "decoded_pixels": int(hf_stats["decoded_pixels"]),
                "decoded_bytes": int(hf_stats["decoded_bytes"]),
                "hf_dataset_init_s": float(hf_stats.get("hf_dataset_init_s", 0.0)),
                "hf_next_example_s": float(hf_stats.get("hf_next_example_s", 0.0)),
                "hf_tif_decode_s": float(hf_stats.get("hf_tif_decode_s", 0.0)),
                "hf_range_size_mib": int(hf_stats.get("hf_range_size_mib", 0)),
                "hf_prefetch_limit": int(hf_stats.get("hf_prefetch_limit", 0)),
            },
            "rasteret": None
            if rasteret_stats is None
            else {
                "name": rasteret_name,
                "elapsed_s": float(rasteret_stats["elapsed_s"]),
                "matched": int(rasteret_stats["matched"]),
                "decoded_pixels": int(rasteret_stats["decoded_pixels"]),
                "decoded_bytes": int(rasteret_stats["decoded_bytes"]),
            },
            "winner": winner,
            "speedup": None if speedup is None else float(speedup),
        }
        with args.json_out.open("w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, indent=2)
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
