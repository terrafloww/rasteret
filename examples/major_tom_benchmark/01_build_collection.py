# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Build a scene-level Rasteret cache for the Major TOM benchmark workflow.

This script is intentionally benchmark-focused:
1) Build Sentinel-2 collection from STAC/catalog
2) Add `major_tom_product_id` / `major_tom_grid_cell` / `split`
3) Persist as a reusable Rasteret collection

Use `02_run_benchmark_from_cache.py` to run benchmark scenarios from this cache.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import rasteret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene-level Rasteret collection for Major TOM benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="earthsearch/sentinel-2-l2a",
        help="Rasteret dataset id to index",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20_000,
        help="Max STAC items to index (scenes).",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B02", "B08"],
        help="Band codes to include in the collection (limits COG enrichment work).",
    )
    parser.add_argument(
        "--source-collection-path",
        type=Path,
        default=None,
        help=(
            "Optional existing scene-level Rasteret collection path to reuse "
            "instead of building from STAC."
        ),
    )
    parser.add_argument(
        "--name",
        default="major-tom-benchmark-scenes",
        help="Collection cache name",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "rasteret_workspace",
        help="Workspace root for persisted collection",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=(-180.0, -90.0, 180.0, 90.0),
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Index bbox in WGS84",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        default=("2020-01-01", "2024-02-06"),
        metavar=("START", "END"),
        help="Date range",
    )
    parser.add_argument(
        "--grid-km",
        type=int,
        default=10,
        help="Grid cell size in km for major_tom_grid_cell",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=120,
        help="Max concurrent requests during index build/enrichment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Rebuild cache even if collection already exists",
    )
    return parser.parse_args()


def _append_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.schema.names:
        idx = table.schema.get_field_index(name)
        return table.set_column(idx, name, values)
    return table.append_column(name, values)


def _major_tom_product_ids(table: pa.Table) -> pa.Array:
    if "s2:product_uri" in table.schema.names:
        uris = pc.cast(table.column("s2:product_uri").combine_chunks(), pa.string())
        cleaned = pc.replace_substring_regex(uris, pattern=r"\.SAFE$", replacement="")
        return pc.fill_null(cleaned, "")
    ids = pc.cast(table.column("id").combine_chunks(), pa.string())
    return pc.fill_null(ids, "")


def _major_tom_grid_cells(table: pa.Table, grid_km: int) -> pa.Array:
    try:
        from majortom.grid import Grid
    except ImportError as exc:
        raise RuntimeError(
            "Major TOM grid helper missing. Install with: "
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
    cells = np.char.add(
        np.char.add(np.asarray(rows).astype(str), "_"), np.asarray(cols).astype(str)
    )
    return pa.array(cells, type=pa.string())


def _split_from_cell(grid_cell: str) -> str:
    bucket = zlib.crc32(grid_cell.encode("utf-8")) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()

    if args.source_collection_path is not None:
        collection = rasteret.load(args.source_collection_path)
    else:
        from rasteret.catalog import DatasetRegistry

        descriptor = DatasetRegistry.get(str(args.dataset))
        if not descriptor.stac_api or not descriptor.stac_collection:
            raise ValueError(
                f"Dataset '{descriptor.id}' has no STAC API/collection configured."
            )

        requested_bands = [str(b) for b in args.bands]
        band_map = descriptor.band_map or {}
        filtered_band_map = {
            band: band_map[band] for band in requested_bands if band in band_map
        }
        if not filtered_band_map:
            raise ValueError(
                f"No requested bands found in dataset band_map: {requested_bands}. "
                f"Available bands: {sorted(band_map)}"
            )

        collection = rasteret.build_from_stac(
            name=args.name,
            stac_api=str(descriptor.stac_api),
            collection=str(descriptor.stac_collection),
            data_source=str(descriptor.id),
            band_map=filtered_band_map,
            band_index_map=descriptor.band_index_map,
            bbox=tuple(float(v) for v in args.bbox),
            date_range=(str(args.date_range[0]), str(args.date_range[1])),
            workspace_dir=args.workspace,
            max_concurrent=int(args.max_concurrent),
            force=bool(args.force),
            query={"max_items": int(args.max_items)},
        )

    schema_names = collection.dataset.schema.names
    selected = []
    for name in [
        "id",
        "datetime",
        "geometry",
        "assets",
        "scene_bbox",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
        "proj:epsg",
        "year",
        "month",
        "s2:product_uri",
        "collection",
        "major_tom_product_id",
        "major_tom_grid_cell",
        "split",
    ]:
        if name in schema_names:
            selected.append(name)
    selected.extend(name for name in schema_names if name.endswith("_metadata"))

    table = collection.dataset.to_table(columns=selected)
    if "major_tom_product_id" in table.schema.names:
        product_ids = pc.cast(table.column("major_tom_product_id"), pa.string())
    else:
        product_ids = _major_tom_product_ids(table)
    if "major_tom_grid_cell" in table.schema.names:
        grid_cells = pc.cast(table.column("major_tom_grid_cell"), pa.string())
    else:
        grid_cells = _major_tom_grid_cells(table, int(args.grid_km))
    if "split" in table.schema.names:
        splits = pc.cast(table.column("split"), pa.string())
    else:
        splits = pa.array(
            [_split_from_cell(cell) for cell in grid_cells.to_pylist()],
            type=pa.string(),
        )

    table = _append_or_replace_column(
        table, "major_tom_product_id", pc.fill_null(product_ids, "")
    )
    table = _append_or_replace_column(
        table, "major_tom_grid_cell", pc.fill_null(grid_cells, "")
    )
    table = _append_or_replace_column(table, "split", pc.fill_null(splits, ""))

    enriched = rasteret.as_collection(
        table,
        name=args.name,
        data_source=collection.data_source,
        description=collection.description,
        start_date=collection.start_date,
        end_date=collection.end_date,
    )

    out_path = Path(args.workspace) / f"{args.name}_records"
    enriched.export(out_path)
    print(f"collection_path={out_path}")
    print(f"rows={enriched.dataset.count_rows()}")


if __name__ == "__main__":
    main()
