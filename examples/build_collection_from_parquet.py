"""Build a Rasteret Collection from any Parquet file with COG URLs.

Works with Source Cooperative exports, STAC GeoParquet, or custom Parquet.
Uses pyarrow.dataset scan APIs for:
- projection pushdown (`columns=...`)
- predicate pushdown (`filter_expr=...`)

Example usage:

    # Source Cooperative (public, no credentials)
    AWS_NO_SIGN_REQUEST=YES python build_collection_from_parquet.py \\
        --manifest-url s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet \\
        --name maxar-opendata

    # Any S3/GCS/local Parquet
    python build_collection_from_parquet.py \\
        --manifest-url s3://my-bucket/items.parquet \\
        --name my-collection --cloud-cover-lt 20
"""

from __future__ import annotations

import argparse
from typing import Any

import pyarrow.dataset as ds

import rasteret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-url",
        required=True,
        help="Remote Parquet URL/path (e.g. s3://.../items.parquet)",
    )
    parser.add_argument("--name", default="parquet_collection")
    parser.add_argument("--data-source", default="")
    parser.add_argument(
        "--workspace-dir",
        default="",
        help="Optional output directory to materialize the Rasteret collection parquet dataset",
    )
    parser.add_argument(
        "--cloud-cover-lt",
        type=float,
        default=None,
        help="Optional eo:cloud_cover predicate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    remote_dataset = ds.dataset(args.manifest_url, format="parquet")
    available = set(remote_dataset.schema.names)
    required = {"id", "datetime", "geometry", "assets"}
    missing = required - available
    if missing:
        raise ValueError(
            f"Record table missing required columns for Rasteret Collection: {sorted(missing)}"
        )

    projected_columns = [
        column
        for column in [
            "id",
            "datetime",
            "geometry",
            "assets",
            "collection",
            "proj:epsg",
            "eo:cloud_cover",
        ]
        if column in available
    ]

    filter_expr: Any = None
    if args.cloud_cover_lt is not None and "eo:cloud_cover" in available:
        filter_expr = ds.field("eo:cloud_cover") < args.cloud_cover_lt

    collection = rasteret.build_from_table(
        args.manifest_url,
        name=args.name,
        data_source=args.data_source,
        columns=projected_columns,
        filter_expr=filter_expr,
        workspace_dir=args.workspace_dir or None,
    )

    count = collection.dataset.count_rows() if collection.dataset is not None else 0
    print(f"Collection: {collection.name}")
    print(f"Rows: {count}")
    print(f"Columns: {collection.dataset.schema.names if collection.dataset else []}")


if __name__ == "__main__":
    main()
