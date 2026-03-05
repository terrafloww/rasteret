# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Build a ~20k-row scene cache for the Major TOM HF benchmark (sharded STAC).

Earth Search STAC can intermittently return server errors for large, global
queries. This script builds the cache by issuing many small bbox/time searches
and then deduplicating scenes by STAC item id.

Output is a Parquet-backed Rasteret collection directory suitable for:
`examples/major_tom_benchmark/03_hf_vs_rasteret_benchmark.py`.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import rasteret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sharded scene-level Rasteret collection for Major TOM benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="earthsearch/sentinel-2-l2a",
        help="Rasteret dataset id to index",
    )
    parser.add_argument(
        "--name",
        default="major-tom-benchmark-scenes-20k",
        help="Output collection cache name (written as <name>_records under --workspace)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "rasteret_workspace",
        help="Workspace root for persisted collection",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=20_000,
        help="Target number of unique scenes to include in output",
    )
    parser.add_argument(
        "--shard-max-items",
        type=int,
        default=500,
        help="Max STAC items to index per shard",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B02", "B08"],
        help="Band codes to include in the collection (limits COG enrichment work).",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        default=("2020-01-01", "2024-02-06"),
        metavar=("START", "END"),
        help="Date range (must overlap Major TOM metadata coverage)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=120,
        help="Max concurrent requests during index build/enrichment",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries per shard when STAC returns server errors",
    )
    parser.add_argument(
        "--retry-sleep-s",
        type=float,
        default=2.0,
        help="Base sleep seconds between shard retries (exponential backoff)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Rebuild shard caches even if they already exist",
    )
    return parser.parse_args()


def _major_tom_product_ids(table: pa.Table) -> pa.Array:
    if "s2:product_uri" in table.schema.names:
        uris = pc.cast(table.column("s2:product_uri").combine_chunks(), pa.string())
        cleaned = pc.replace_substring_regex(uris, pattern=r"\.SAFE$", replacement="")
        return pc.fill_null(cleaned, "")
    ids = pc.cast(table.column("id").combine_chunks(), pa.string())
    return pc.fill_null(ids, "")


def _default_shard_bboxes() -> list[tuple[float, float, float, float]]:
    """Small 1x1 degree bboxes around globally distributed locations."""
    centers = [
        (-122.4194, 37.7749),  # San Francisco
        (-74.0060, 40.7128),  # New York
        (-99.1332, 19.4326),  # Mexico City
        (-58.3816, -34.6037),  # Buenos Aires
        (-46.6333, -23.5505),  # São Paulo
        (-0.1276, 51.5072),  # London
        (2.3522, 48.8566),  # Paris
        (13.4050, 52.5200),  # Berlin
        (37.6173, 55.7558),  # Moscow
        (31.2357, 30.0444),  # Cairo
        (3.3792, 6.5244),  # Lagos
        (18.4241, -33.9249),  # Cape Town
        (55.2708, 25.2048),  # Dubai
        (77.5946, 12.9716),  # Bengaluru
        (72.8777, 19.0760),  # Mumbai
        (116.4074, 39.9042),  # Beijing
        (139.6917, 35.6895),  # Tokyo
        (126.9780, 37.5665),  # Seoul
        (106.8456, -6.2088),  # Jakarta
        (100.5018, 13.7563),  # Bangkok
        (151.2093, -33.8688),  # Sydney
        (144.9631, -37.8136),  # Melbourne
        (174.7633, -36.8485),  # Auckland
        (28.0473, -26.2041),  # Johannesburg
        (36.8219, -1.2921),  # Nairobi
        (90.4125, 23.8103),  # Dhaka
        (88.3639, 22.5726),  # Kolkata
        (-3.7038, 40.4168),  # Madrid
        (12.4964, 41.9028),  # Rome
        (19.0402, 47.4979),  # Budapest
        (35.2137, 31.7683),  # Jerusalem
        (29.9187, 31.2001),  # Alexandria
        (-43.1729, -22.9068),  # Rio
        (-70.6693, -33.4489),  # Santiago
        (-79.3832, 43.6532),  # Toronto
        (-123.1207, 49.2827),  # Vancouver
        (-87.6298, 41.8781),  # Chicago
        (-95.3698, 29.7604),  # Houston
        (-118.2437, 34.0522),  # Los Angeles
        (-157.8583, 21.3069),  # Honolulu
        (114.1694, 22.3193),  # Hong Kong
        (121.4737, 31.2304),  # Shanghai
        (103.8198, 1.3521),  # Singapore
        (73.0479, 33.6844),  # Islamabad
        (67.0099, 24.8615),  # Karachi
        (44.3661, 33.3152),  # Baghdad
        (35.5018, 33.8938),  # Beirut
        (30.5234, 50.4501),  # Kyiv
        (24.7536, 59.4370),  # Tallinn
        (10.7522, 59.9139),  # Oslo
        (18.0686, 59.3293),  # Stockholm
        (12.5683, 55.6761),  # Copenhagen
        (4.9041, 52.3676),  # Amsterdam
        (-8.6131, 41.1579),  # Porto
        (-9.1393, 38.7223),  # Lisbon
        (-6.2603, 53.3498),  # Dublin
    ]
    extra_offsets = [
        (1.25, 0.75),
        (-1.10, -0.80),
    ]
    half = 0.5
    bboxes: list[tuple[float, float, float, float]] = []
    for lon, lat in centers:
        clamped_lon = float(np.clip(lon, -179.0, 179.0))
        clamped_lat = float(np.clip(lat, -89.0, 89.0))
        bboxes.append(
            (
                clamped_lon - half,
                clamped_lat - half,
                clamped_lon + half,
                clamped_lat + half,
            )
        )
    for lon, lat in centers:
        for dlon, dlat in extra_offsets:
            clamped_lon = float(np.clip(lon + dlon, -179.0, 179.0))
            clamped_lat = float(np.clip(lat + dlat, -89.0, 89.0))
            bboxes.append(
                (
                    clamped_lon - half,
                    clamped_lat - half,
                    clamped_lon + half,
                    clamped_lat + half,
                )
            )
    return bboxes


def _iter_shards(bboxes: Iterable[tuple[float, float, float, float]]):
    for idx, bbox in enumerate(bboxes):
        yield idx, bbox


def _build_shard_table(
    *,
    dataset_id: str,
    shard_name: str,
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    requested_bands: list[str],
    max_items: int,
    max_concurrent: int,
    force: bool,
) -> pa.Table:
    from rasteret.catalog import DatasetRegistry

    descriptor = DatasetRegistry.get(str(dataset_id))
    if not descriptor.stac_api or not descriptor.stac_collection:
        raise ValueError(f"Dataset '{descriptor.id}' has no STAC API/collection.")

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
        name=shard_name,
        stac_api=str(descriptor.stac_api),
        collection=str(descriptor.stac_collection),
        data_source=str(descriptor.id),
        band_map=filtered_band_map,
        band_index_map=descriptor.band_index_map,
        bbox=bbox,
        date_range=date_range,
        max_concurrent=int(max_concurrent),
        force=bool(force),
        query={"max_items": int(max_items)},
    )

    schema_names = collection.dataset.schema.names
    selected: list[str] = []
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
    ]:
        if name in schema_names:
            selected.append(name)
    selected.extend(name for name in schema_names if name.endswith("_metadata"))
    return collection.dataset.to_table(columns=selected)


def main() -> None:
    args = parse_args()

    target_rows = int(args.target_rows)
    shard_max_items = int(args.shard_max_items)
    requested_bands = [str(b) for b in args.bands]
    date_range = (str(args.date_range[0]), str(args.date_range[1]))

    out_path = Path(args.workspace) / f"{args.name}_records"
    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Output collection already exists: {out_path}. " "Pass --force to rebuild."
        )

    seen_ids: set[str] = set()
    tables: list[pa.Table] = []

    bboxes = _default_shard_bboxes()
    for shard_idx, bbox in _iter_shards(bboxes):
        if len(seen_ids) >= target_rows:
            break

        shard_name = f"{args.name}-shard-{shard_idx:03d}"

        last_exc: Exception | None = None
        table: pa.Table | None = None
        for attempt in range(int(args.max_retries) + 1):
            try:
                table = _build_shard_table(
                    dataset_id=str(args.dataset),
                    shard_name=shard_name,
                    bbox=bbox,
                    date_range=date_range,
                    requested_bands=requested_bands,
                    max_items=shard_max_items,
                    max_concurrent=int(args.max_concurrent),
                    force=bool(args.force),
                )
                break
            except Exception as exc:  # noqa: BLE001 (benchmark tooling)
                last_exc = exc
                if attempt >= int(args.max_retries):
                    table = None
                    break
                sleep_s = float(args.retry_sleep_s) * (2**attempt)
                time.sleep(sleep_s)
        if table is None:
            msg = str(last_exc) if last_exc is not None else "unknown error"
            print(f"shard_failed name={shard_name} err={msg}")
            continue

        ids = pc.cast(table.column("id").combine_chunks(), pa.string()).to_pylist()
        keep = np.zeros(len(ids), dtype=bool)
        new_count = 0
        for i, value in enumerate(ids):
            if value is None:
                continue
            s = str(value)
            if s in seen_ids:
                continue
            seen_ids.add(s)
            keep[i] = True
            new_count += 1
            if len(seen_ids) >= target_rows:
                break

        if new_count == 0:
            continue

        filtered = table.filter(pa.array(keep.tolist()))
        tables.append(filtered)
        print(
            f"shard_ok name={shard_name} new={new_count} unique_total={len(seen_ids)}"
        )

    if len(seen_ids) < target_rows:
        raise RuntimeError(
            f"Could not collect enough unique scenes: {len(seen_ids)} < {target_rows}. "
            "Try increasing shard count/coverage or widen --date-range."
        )

    combined = pa.concat_tables(tables, promote_options="permissive")
    combined = combined.slice(0, target_rows)
    combined = combined.append_column(
        "major_tom_product_id", _major_tom_product_ids(combined)
    )

    enriched = rasteret.as_collection(
        combined,
        name=str(args.name),
        data_source=str(args.dataset),
        description="Major TOM benchmark scene cache (sharded STAC build)",
    )
    enriched.export(out_path)

    print(f"collection_path={out_path}")
    print(f"rows={enriched.dataset.count_rows()}")


if __name__ == "__main__":
    main()
