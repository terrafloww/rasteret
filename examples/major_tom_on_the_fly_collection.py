# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Build a Major TOM-style Rasteret Collection directly from Sentinel-2 STAC.

This example shows a concise, production path:
1) Build a Sentinel-2 Collection from catalog/STAC
2) Add Major TOM-style columns (`major_tom_product_id`, `major_tom_grid_cell`, `split`)
3) Fetch tensors from Arrow geometry columns via `Collection.get_numpy()`

The goal is low code + easy metadata iteration/sharing + fast reads.
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import shapely
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

import rasteret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and query a Major TOM-style Collection on the fly"
    )
    parser.add_argument(
        "--dataset",
        default="earthsearch/sentinel-2-l2a",
        help="Rasteret catalog dataset id",
    )
    parser.add_argument(
        "--name",
        default="major-tom-on-the-fly",
        help="Output Collection name",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-122.55, 37.65, -122.30, 37.90),
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Bounding box",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        default=("2024-01-01", "2024-02-01"),
        metavar=("START", "END"),
        help="Date range",
    )
    parser.add_argument(
        "--grid-km",
        type=int,
        default=10,
        help="Major TOM grid size in km",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split to sample for retrieval",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of sample geometries to fetch",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B02", "B08"],
        help="Bands to fetch",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=256,
        help="Patch size in pixels for per-scene AOI chips",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=10.0,
        help="Patch resolution in meters for AOI chip construction",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=120,
        help="Concurrency for enrichment/retrieval",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "rasteret_workspace",
        help="Workspace root",
    )
    parser.add_argument(
        "--metadata-path",
        default="",
        help="Optional Major TOM metadata Parquet path to report key overlap",
    )
    return parser.parse_args()


def _split_from_grid(grid_cell: str) -> str:
    bucket = zlib.crc32(grid_cell.encode("utf-8")) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _append_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.schema.names:
        idx = table.schema.get_field_index(name)
        return table.set_column(idx, name, values)
    return table.append_column(name, values)


def _major_tom_grid_cells(table: pa.Table, grid_km: int) -> pa.Array:
    try:
        from majortom.grid import Grid
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Major TOM grid helper is not installed. Install with: "
            "pip install git+https://github.com/ESA-PhiLab/Major-TOM"
        ) from exc

    minx = table.column("bbox_minx").to_numpy(zero_copy_only=False)
    miny = table.column("bbox_miny").to_numpy(zero_copy_only=False)
    maxx = table.column("bbox_maxx").to_numpy(zero_copy_only=False)
    maxy = table.column("bbox_maxy").to_numpy(zero_copy_only=False)
    lats = (miny + maxy) / 2.0
    lons = (minx + maxx) / 2.0

    grid = Grid(grid_km, latitude_range=(-90, 90), longitude_range=(-180, 180))
    rows, cols = grid.latlon2rowcol(lats, lons)
    rows_np = np.asarray(rows).astype(str)
    cols_np = np.asarray(cols).astype(str)
    cells = np.char.add(np.char.add(rows_np, "_"), cols_np)
    return pa.array(cells, type=pa.string())


def _major_tom_product_ids(table: pa.Table) -> pa.Array:
    if "s2:product_uri" in table.schema.names:
        uris = table.column("s2:product_uri").combine_chunks()
        uris = pc.cast(uris, pa.string())
        cleaned = pc.replace_substring_regex(uris, pattern=r"\.SAFE$", replacement="")
        return pc.fill_null(cleaned, "")
    ids = pc.cast(table.column("id").combine_chunks(), pa.string())
    return pc.fill_null(ids, "")


def enrich_major_tom_columns(base: "rasteret.Collection", grid_km: int) -> pa.Table:
    schema_names = base.dataset.schema.names
    selected_columns: list[str] = []
    for name in [
        "id",
        "datetime",
        "geometry",
        "assets",
        "proj:epsg",
        "scene_bbox",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
        "s2:product_uri",
        "collection",
    ]:
        if name in schema_names:
            selected_columns.append(name)
    selected_columns.extend(name for name in schema_names if name.endswith("_metadata"))

    table = base.dataset.to_table(columns=selected_columns)
    product_ids = _major_tom_product_ids(table)
    grid_cells = _major_tom_grid_cells(table, grid_km)
    splits = pa.array(
        [_split_from_grid(cell) for cell in grid_cells.to_numpy(zero_copy_only=False)],
        type=pa.string(),
    )

    table = _append_or_replace_column(table, "major_tom_product_id", product_ids)
    table = _append_or_replace_column(table, "major_tom_grid_cell", grid_cells)
    table = _append_or_replace_column(table, "split", splits)
    return table


def report_metadata_overlap(
    enriched: "rasteret.Collection", metadata_path: str
) -> None:
    local_keys = enriched.dataset.to_table(
        columns=["major_tom_product_id", "major_tom_grid_cell"]
    )
    local_keys = pa.table(
        {
            "product_id": local_keys.column("major_tom_product_id"),
            "grid_cell": local_keys.column("major_tom_grid_cell"),
        }
    ).combine_chunks()
    local_valid = pc.and_(
        pc.is_valid(local_keys.column("product_id")),
        pc.is_valid(local_keys.column("grid_cell")),
    )
    local_keys = (
        local_keys.filter(local_valid)
        .group_by(["product_id", "grid_cell"])
        .aggregate([])
    )

    if local_keys.num_rows == 0:
        print("metadata_key_overlap=0 (no local keys)")
        return

    product_set = pc.unique(local_keys.column("product_id"))
    metadata_ds = ds.dataset(metadata_path, format="parquet")
    meta_filter = ds.field("product_id").isin(product_set)
    meta_keys = metadata_ds.to_table(
        columns=["product_id", "grid_cell"],
        filter=meta_filter,
    )
    meta_keys = meta_keys.combine_chunks()
    meta_valid = pc.and_(
        pc.is_valid(meta_keys.column("product_id")),
        pc.is_valid(meta_keys.column("grid_cell")),
    )
    meta_keys = (
        meta_keys.filter(meta_valid).group_by(["product_id", "grid_cell"]).aggregate([])
    )

    if meta_keys.num_rows == 0:
        print("metadata_key_overlap=0 (no metadata keys after filtering)")
        return

    overlap = local_keys.join(
        meta_keys, keys=["product_id", "grid_cell"], join_type="inner"
    )
    print(
        "metadata_key_overlap="
        f"{overlap.num_rows}/{local_keys.num_rows} local keys matched metadata "
        f"({meta_keys.num_rows} metadata keys scanned after product_id filter)"
    )


def _patch_wkb_from_center(
    center_lon: float,
    center_lat: float,
    epsg: int,
    chip_size: int,
    resolution_m: float,
) -> bytes:
    half_patch_meters = (chip_size * resolution_m) / 2.0
    to_projected = Transformer.from_crs(4326, epsg, always_xy=True)
    to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
    x_center, y_center = to_projected.transform(center_lon, center_lat)
    projected_geom = box(
        x_center - half_patch_meters,
        y_center - half_patch_meters,
        x_center + half_patch_meters,
        y_center + half_patch_meters,
    )
    wgs84_geom = shapely_transform(to_wgs84.transform, projected_geom)
    return shapely.to_wkb(wgs84_geom)


def fetch_numpy_scene_batch(
    collection: "rasteret.Collection",
    split: str,
    bands: list[str],
    samples: int,
    max_concurrent: int,
    chip_size: int,
    resolution_m: float,
) -> tuple[int, float]:
    subset = collection.subset(split=split)
    sample = subset.dataset.head(
        samples,
        columns=[
            "major_tom_product_id",
            "bbox_minx",
            "bbox_miny",
            "bbox_maxx",
            "bbox_maxy",
            "proj:epsg",
        ],
    )
    if sample.num_rows == 0:
        raise RuntimeError(f"No rows available for split='{split}'")

    centers_lon = (
        sample.column("bbox_minx").to_numpy(zero_copy_only=False)
        + sample.column("bbox_maxx").to_numpy(zero_copy_only=False)
    ) / 2.0
    centers_lat = (
        sample.column("bbox_miny").to_numpy(zero_copy_only=False)
        + sample.column("bbox_maxy").to_numpy(zero_copy_only=False)
    ) / 2.0
    epsg_values = sample.column("proj:epsg").to_numpy(zero_copy_only=False)
    product_ids = pc.fill_null(
        pc.cast(sample.column("major_tom_product_id").combine_chunks(), pa.string()), ""
    ).to_numpy(zero_copy_only=False)

    patch_wkbs = [
        _patch_wkb_from_center(
            center_lon=float(center_lon),
            center_lat=float(center_lat),
            epsg=int(epsg),
            chip_size=chip_size,
            resolution_m=resolution_m,
        )
        for center_lon, center_lat, epsg in zip(
            centers_lon, centers_lat, epsg_values, strict=False
        )
    ]
    geometry_array = pa.array(patch_wkbs, type=pa.binary())

    start = time.perf_counter()
    total_samples = 0
    grouped_indices: dict[str, list[int]] = {}
    for idx, product_id in enumerate(product_ids):
        pid = str(product_id)
        grouped_indices.setdefault(pid, []).append(idx)

    for product_id, index_list in grouped_indices.items():
        scene_view = subset.where(ds.field("major_tom_product_id") == product_id)
        geometries = pc.take(geometry_array, pa.array(index_list, type=pa.int64()))
        array = scene_view.get_numpy(
            geometries=geometries,
            bands=bands,
            max_concurrent=max_concurrent,
        )
        total_samples += int(array.shape[0])

    elapsed = time.perf_counter() - start
    return total_samples, elapsed


def main() -> None:
    args = parse_args()
    base_name = f"{args.name}-base"
    base = rasteret.build(
        args.dataset,
        name=base_name,
        bbox=tuple(args.bbox),
        date_range=tuple(args.date_range),
        workspace_dir=args.workspace,
        force=True,
        max_concurrent=args.max_concurrent,
    )
    print(f"base_rows={base.dataset.count_rows()}")

    enriched_table = enrich_major_tom_columns(base, args.grid_km)
    enriched = rasteret.build_from_table(
        enriched_table,
        name=args.name,
        data_source="sentinel-2-l2a",
        workspace_dir=args.workspace,
        enrich_cog=False,
        max_concurrent=args.max_concurrent,
        force=True,
    )
    print(f"enriched_rows={enriched.dataset.count_rows()}")
    print(f"collection_path={args.workspace / f'{args.name}_records'}")

    if args.metadata_path:
        report_metadata_overlap(enriched, args.metadata_path)

    fetched, elapsed = fetch_numpy_scene_batch(
        collection=enriched,
        split=args.split,
        bands=args.bands,
        samples=args.samples,
        max_concurrent=args.max_concurrent,
        chip_size=args.chip_size,
        resolution_m=args.resolution_m,
    )
    print(
        f"get_numpy_scene_batch split={args.split} samples={fetched} "
        f"elapsed_s={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
