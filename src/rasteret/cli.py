# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rasteret import __version__, build_from_stac, register_local
from rasteret.core.collection import Collection


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be minx,miny,maxx,maxy")
    try:
        bbox = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numbers") from exc
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError("bbox must satisfy minx<maxx and miny<maxy")
    return bbox


def _parse_date_range(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("date-range must be start,end")
    return parts[0], parts[1]


def _workspace_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    return Path.home() / "rasteret_workspace"


def _resolve_collection_path(name: str, workspace_dir: Path) -> Path:
    direct = workspace_dir / name
    if direct.exists():
        return direct

    suffixes = ("_stac", "_records")
    for suffix in suffixes:
        candidate = workspace_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate

    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(sorted(workspace_dir.glob(f"{name}*{suffix}")))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        joined = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            f"Ambiguous collection name '{name}'. Matches: {joined}"
        )

    raise FileNotFoundError(f"Collection '{name}' not found in {workspace_dir}")


def _collection_summary(
    collection: Collection, collection_path: Path
) -> dict[str, Any]:
    scene_count = 0
    date_start: str | None = None
    date_end: str | None = None
    columns: list[str] = []
    has_split_column = False

    if collection.dataset is not None:
        scene_count = collection.dataset.count_rows()
        columns = list(collection.dataset.schema.names)
        has_split_column = "split" in columns
        if "datetime" in columns:
            table = collection.dataset.to_table(columns=["datetime"])
            values = table.column("datetime").to_pylist()
            values = [value for value in values if value is not None]
            if values:
                date_start = min(values).isoformat()
                date_end = max(values).isoformat()

    return {
        "name": collection.name,
        "path": str(collection_path),
        "scene_count": scene_count,
        "date_start": date_start,
        "date_end": date_end,
        "has_split_column": has_split_column,
        "columns": columns,
    }


def _print_list_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No cached collections found.")
        return

    headers = ("name", "kind", "data_source", "size", "date_start", "date_end")
    matrix: list[tuple[str, str, str, str, str, str]] = []
    for row in rows:
        date_range = row.get("date_range")
        start = date_range[0] if date_range else "-"
        end = date_range[1] if date_range else "-"
        matrix.append(
            (
                str(row.get("name", "")),
                str(row.get("kind", "")),
                str(row.get("data_source", "")),
                str(row.get("size", 0)),
                start,
                end,
            )
        )

    widths = [
        max(len(headers[index]), *(len(item[index]) for item in matrix))
        for index in range(len(headers))
    ]
    print(
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    )
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for item in matrix:
        print(
            "  ".join(item[index].ljust(widths[index]) for index in range(len(headers)))
        )


def _handle_collections_build(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    query = json.loads(args.query) if args.query else None
    collection = build_from_stac(
        name=args.name,
        stac_api=args.stac_api,
        collection=args.collection,
        bbox=args.bbox,
        date_range=args.date_range,
        workspace_dir=workspace_dir,
        force=args.force,
        max_concurrent=args.max_concurrent,
        query=query,
    )
    collection_path = workspace_dir / f"{collection.name}_stac"
    summary = _collection_summary(collection, collection_path)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Collection built: {summary['name']}")
    print(f"Path: {summary['path']}")
    print(f"Scenes: {summary['scene_count']}")
    if summary["date_start"] and summary["date_end"]:
        print(f"Date range: {summary['date_start']} -> {summary['date_end']}")
    print(f"Split column: {'yes' if summary['has_split_column'] else 'no'}")
    return 0


def _handle_collections_list(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    rows = Collection.list_collections(workspace_dir=workspace_dir)
    rows = sorted(rows, key=lambda row: row.get("name", ""))

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    _print_list_rows(rows)
    return 0


def _handle_collections_info(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    try:
        collection_path = _resolve_collection_path(args.name, workspace_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", flush=True)
        return 1
    collection = Collection._load_cached(collection_path)
    summary = _collection_summary(collection, collection_path)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Name: {summary['name']}")
    print(f"Path: {summary['path']}")
    print(f"Scenes: {summary['scene_count']}")
    if summary["date_start"] and summary["date_end"]:
        print(f"Date range: {summary['date_start']} -> {summary['date_end']}")
    print(f"Split column: {'yes' if summary['has_split_column'] else 'no'}")
    print(f"Columns ({len(summary['columns'])}): {', '.join(summary['columns'])}")
    return 0


def _handle_collections_delete(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    try:
        collection_path = _resolve_collection_path(args.name, workspace_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", flush=True)
        return 1

    if not args.yes:
        response = input(f"Delete collection at {collection_path}? [y/N]: ")
        if response.strip().lower() not in {"y", "yes"}:
            print("Delete cancelled.")
            return 0

    shutil.rmtree(collection_path)
    print(f"Deleted: {collection_path}")
    return 0


def _handle_collections_import(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    collection_path = workspace_dir / f"{args.name}_records"
    if collection_path.exists():
        if not args.force:
            collection = Collection._load_cached(collection_path)
            summary = _collection_summary(collection, collection_path)
            if args.json:
                print(json.dumps(summary, indent=2))
                return 0
            print(f"Collection exists: {summary['name']}")
            print(f"Path: {summary['path']}")
            print(f"Scenes: {summary['scene_count']}")
            return 0
        shutil.rmtree(collection_path)

    column_map = json.loads(args.column_map) if args.column_map else None
    columns = (
        [value.strip() for value in args.columns.split(",") if value.strip()]
        if args.columns
        else None
    )

    from rasteret.ingest.parquet_record_table import RecordTableBuilder

    builder = RecordTableBuilder(
        args.record_table,
        data_source=args.data_source or "",
        column_map=column_map,
        columns=columns,
    )
    collection = builder.build(name=args.name, workspace_dir=collection_path)
    summary = _collection_summary(collection, collection_path)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Collection imported: {summary['name']}")
    print(f"Path: {summary['path']}")
    print(f"Scenes: {summary['scene_count']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the ``rasteret`` CLI."""
    parser = argparse.ArgumentParser(prog="rasteret")
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Show the rasteret version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    collections_parser = subparsers.add_parser(
        "collections", help="Manage local collections"
    )
    collections_subparsers = collections_parser.add_subparsers(
        dest="collections_command", required=True
    )

    build_parser = collections_subparsers.add_parser(
        "build", help="Build or refresh a STAC-backed collection cache"
    )
    build_parser.add_argument("name", help="Logical collection name")
    build_parser.add_argument("--stac-api", required=True, help="STAC API URL")
    build_parser.add_argument("--collection", required=True, help="STAC collection id")
    build_parser.add_argument(
        "--bbox",
        required=True,
        type=_parse_bbox,
        help="Bounding box as minx,miny,maxx,maxy",
    )
    build_parser.add_argument(
        "--date-range",
        required=True,
        type=_parse_date_range,
        help="Date range as YYYY-MM-DD,YYYY-MM-DD",
    )
    build_parser.add_argument(
        "--workspace-dir", help="Workspace directory for cached collections"
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if it already exists",
    )
    build_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent metadata fetch operations",
    )
    build_parser.add_argument(
        "--query",
        help="Optional STAC search query as JSON string",
    )
    build_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    list_parser = collections_subparsers.add_parser(
        "list", help="List local collections"
    )
    list_parser.add_argument(
        "--workspace-dir", help="Workspace directory for cached collections"
    )
    list_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    info_parser = collections_subparsers.add_parser(
        "info", help="Show details for a local collection"
    )
    info_parser.add_argument("name", help="Collection name or folder name")
    info_parser.add_argument(
        "--workspace-dir", help="Workspace directory for cached collections"
    )
    info_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    delete_parser = collections_subparsers.add_parser(
        "delete", help="Delete a local collection"
    )
    delete_parser.add_argument("name", help="Collection name or folder name")
    delete_parser.add_argument(
        "--workspace-dir", help="Workspace directory for cached collections"
    )
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    import_parser = collections_subparsers.add_parser(
        "import", help="Materialize a local collection from a Parquet record table"
    )
    import_parser.add_argument("name", help="Logical collection name")
    import_parser.add_argument(
        "--record-table",
        required=True,
        help="Parquet/GeoParquet record table path/URI",
    )
    import_parser.add_argument(
        "--data-source",
        default="",
        help="Optional data source identifier for band mapping and URL policy",
    )
    import_parser.add_argument(
        "--column-map",
        help='Optional column rename mapping as JSON (e.g. \'{"scene_id":"id"}\')',
    )
    import_parser.add_argument(
        "--columns",
        help="Optional projected columns (comma-separated) for scan-time pushdown",
    )
    import_parser.add_argument(
        "--workspace-dir", help="Workspace directory for cached collections"
    )
    import_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing imported collection",
    )
    import_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    # --- top-level build shortcut (mirrors rasteret.build()) ---
    top_build_parser = subparsers.add_parser(
        "build", help="Build a collection from a registered dataset"
    )
    top_build_parser.add_argument("dataset_id", help="Dataset ID")
    top_build_parser.add_argument("name", help="Logical collection name")
    top_build_parser.add_argument(
        "--bbox",
        type=_parse_bbox,
        help="Bounding box as minx,miny,maxx,maxy",
    )
    top_build_parser.add_argument(
        "--date-range",
        type=_parse_date_range,
        help="Date range as YYYY-MM-DD,YYYY-MM-DD",
    )
    top_build_parser.add_argument("--workspace-dir", help="Workspace directory")
    top_build_parser.add_argument("--force", action="store_true", help="Rebuild cache")
    top_build_parser.add_argument(
        "--max-concurrent", type=int, default=50, help="Max concurrent fetches"
    )
    top_build_parser.add_argument("--query", help="Additional STAC query as JSON")
    top_build_parser.add_argument(
        "--json", action="store_true", help="Emit JSON output"
    )

    # --- datasets subcommand ---
    ds_parser = subparsers.add_parser("datasets", help="Browse registered datasets")
    ds_subparsers = ds_parser.add_subparsers(dest="ds_command", required=True)

    ds_list_parser = ds_subparsers.add_parser("list", help="List registered datasets")
    ds_list_parser.add_argument("--search", default="", help="Filter by keyword")
    ds_list_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    ds_info_parser = ds_subparsers.add_parser(
        "info", help="Show details for a registered dataset"
    )
    ds_info_parser.add_argument(
        "dataset_id", help="Dataset ID (e.g. earthsearch/sentinel-2-l2a)"
    )
    ds_info_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    ds_build_parser = ds_subparsers.add_parser(
        "build", help="Build a collection from a registered dataset"
    )
    ds_build_parser.add_argument("dataset_id", help="Dataset ID")
    ds_build_parser.add_argument("name", help="Logical collection name")
    ds_build_parser.add_argument(
        "--bbox",
        type=_parse_bbox,
        help="Bounding box as minx,miny,maxx,maxy",
    )
    ds_build_parser.add_argument(
        "--date-range",
        type=_parse_date_range,
        help="Date range as YYYY-MM-DD,YYYY-MM-DD",
    )
    ds_build_parser.add_argument("--workspace-dir", help="Workspace directory")
    ds_build_parser.add_argument("--force", action="store_true", help="Rebuild cache")
    ds_build_parser.add_argument(
        "--max-concurrent", type=int, default=50, help="Max concurrent fetches"
    )
    ds_build_parser.add_argument("--query", help="Additional STAC query as JSON")
    ds_build_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    ds_register_local_parser = ds_subparsers.add_parser(
        "register-local",
        help="Register a local collection/Parquet as a reusable dataset descriptor",
    )
    ds_register_local_parser.add_argument(
        "dataset_id",
        help="Dataset ID (e.g. local/my-dataset)",
    )
    ds_register_local_parser.add_argument(
        "path",
        help="Collection path, Parquet path, or cached collection name",
    )
    ds_register_local_parser.add_argument(
        "--name",
        default="",
        help="Human-readable dataset name",
    )
    ds_register_local_parser.add_argument(
        "--description",
        default="",
        help="Optional one-line description",
    )
    ds_register_local_parser.add_argument(
        "--data-source",
        default="",
        help="Optional source id override (defaults to inferred metadata)",
    )
    ds_register_local_parser.add_argument(
        "--workspace-dir",
        help="Workspace directory (used when path is a cache name)",
    )
    ds_register_local_parser.add_argument(
        "--registry-path",
        help="Override persisted local registry path",
    )
    ds_register_local_parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Register only for this process; do not persist to disk",
    )
    ds_register_local_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )

    ds_unregister_local_parser = ds_subparsers.add_parser(
        "unregister-local",
        help="Remove a local dataset descriptor from registry and runtime",
    )
    ds_unregister_local_parser.add_argument(
        "dataset_id",
        help="Dataset ID (e.g. local/my-dataset)",
    )
    ds_unregister_local_parser.add_argument(
        "--registry-path",
        help="Override persisted local registry path",
    )
    ds_unregister_local_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )

    ds_export_local_parser = ds_subparsers.add_parser(
        "export-local",
        help="Export a local dataset descriptor as JSON for sharing",
    )
    ds_export_local_parser.add_argument(
        "dataset_id",
        help="Dataset ID (e.g. local/my-dataset)",
    )
    ds_export_local_parser.add_argument(
        "output_path",
        help="Destination JSON path",
    )
    ds_export_local_parser.add_argument(
        "--registry-path",
        help="Override persisted local registry path",
    )
    ds_export_local_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )

    return parser


def _handle_datasets_list(args: argparse.Namespace) -> int:
    from rasteret.catalog import DatasetRegistry

    if args.search:
        descriptors = DatasetRegistry.search(args.search)
    else:
        descriptors = DatasetRegistry.list()

    if args.json:
        import dataclasses

        print(json.dumps([dataclasses.asdict(d) for d in descriptors], indent=2))
        return 0

    if not descriptors:
        print("No datasets found.")
        return 0

    # Column widths
    id_w = max(len("ID"), *(len(d.id) for d in descriptors))
    name_w = max(len("Name"), *(len(d.name) for d in descriptors))
    cov_w = max(len("Coverage"), *(len(d.spatial_coverage) for d in descriptors))
    lic_w = max(
        len("License"),
        *(len(d.license[:20]) for d in descriptors),
    )

    header = f"{'ID':<{id_w}}  {'Name':<{name_w}}  {'Coverage':<{cov_w}}  {'License':<{lic_w}}  Auth"
    print(header)
    print(
        f"{'--':<{id_w}}  {'----':<{name_w}}  {'--------':<{cov_w}}  {'-------':<{lic_w}}  ----"
    )
    for d in descriptors:
        auth = "none" if not d.requires_auth else "required"
        lic = d.license[:20] if d.license else "-"
        print(
            f"{d.id:<{id_w}}  {d.name:<{name_w}}  {d.spatial_coverage:<{cov_w}}  {lic:<{lic_w}}  {auth}"
        )

    return 0


def _handle_datasets_info(args: argparse.Namespace) -> int:
    from rasteret.catalog import DatasetRegistry

    descriptor = DatasetRegistry.get(args.dataset_id)
    if descriptor is None:
        available = [d.id for d in DatasetRegistry.list()]
        print(f"Dataset '{args.dataset_id}' not found.")
        print(f"Available: {', '.join(available)}")
        return 1

    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(descriptor), indent=2))
        return 0

    print(f"{descriptor.name} ({descriptor.id})")
    if descriptor.description:
        print(f"  Description:  {descriptor.description}")
    if descriptor.stac_api:
        print(f"  STAC API:     {descriptor.stac_api}")
    if descriptor.stac_collection:
        print(f"  Collection:   {descriptor.stac_collection}")
    if descriptor.geoparquet_uri:
        print(f"  GeoParquet:   {descriptor.geoparquet_uri}")
    if descriptor.spatial_coverage:
        print(f"  Coverage:     {descriptor.spatial_coverage}")
    if descriptor.license:
        print(f"  License:      {descriptor.license}")
    if descriptor.license_url:
        print(f"  License URL:  {descriptor.license_url}")
    commercial = "yes" if descriptor.commercial_use else "no"
    print(f"  Commercial:   {commercial}")
    if descriptor.temporal_range:
        start, end = descriptor.temporal_range
        print(f"  Temporal:     {start} to {end}")
    if descriptor.example_bbox:
        bbox = ",".join(str(v) for v in descriptor.example_bbox)
        print(f"  Example bbox: {bbox}")
    if descriptor.example_date_range:
        start, end = descriptor.example_date_range
        print(f"  Example time: {start} to {end}")
    if descriptor.band_map:
        bands = ", ".join(descriptor.band_map.keys())
        print(f"  Bands:        {bands}")
    print(
        f"  Files:        {'separate (one per band)' if descriptor.separate_files else 'single (multi-band)'}"
    )
    print(f"  Auth:         {'required' if descriptor.requires_auth else 'none'}")
    if descriptor.torchgeo_class:
        verified = "yes" if descriptor.torchgeo_verified else "pending"
        print(
            f"  TorchGeo:     torchgeo.datasets.{descriptor.torchgeo_class} (verified: {verified})"
        )

    return 0


def _handle_datasets_build(args: argparse.Namespace) -> int:
    from rasteret import build
    from rasteret.catalog import DatasetRegistry

    workspace_dir = _workspace_dir(args.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    query = json.loads(args.query) if args.query else None

    descriptor = DatasetRegistry.get(args.dataset_id)
    if descriptor is None:
        available = [d.id for d in DatasetRegistry.list()]
        print(f"Dataset '{args.dataset_id}' not found.")
        print(f"Available: {', '.join(available)}")
        return 1

    if descriptor.stac_api and descriptor.stac_collection:
        if args.bbox is None or args.date_range is None:
            print("STAC datasets require --bbox and --date-range.")
            return 1

    try:
        collection = build(
            args.dataset_id,
            name=args.name,
            bbox=args.bbox,
            date_range=args.date_range,
            workspace_dir=workspace_dir,
            force=args.force,
            max_concurrent=args.max_concurrent,
            query=query,
        )
    except KeyError as exc:
        print(str(exc))
        return 1

    collection_path = workspace_dir / f"{collection.name}_stac"
    summary = _collection_summary(collection, collection_path)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Collection built: {summary['name']}")
    print(f"Path: {summary['path']}")
    print(f"Scenes: {summary['scene_count']}")
    if summary["date_start"] and summary["date_end"]:
        print(f"Date range: {summary['date_start']} -> {summary['date_end']}")
    return 0


def _handle_datasets_register_local(args: argparse.Namespace) -> int:
    workspace_dir = _workspace_dir(args.workspace_dir)
    input_path = Path(args.path).expanduser()

    if not input_path.exists():
        try:
            input_path = _resolve_collection_path(args.path, workspace_dir)
        except FileNotFoundError as exc:
            print(str(exc))
            return 1

    descriptor = register_local(
        args.dataset_id,
        input_path,
        name=args.name or None,
        description=args.description,
        data_source=args.data_source,
        persist=not args.no_persist,
        registry_path=args.registry_path,
    )

    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(descriptor), indent=2))
        return 0

    print(f"Registered local dataset: {descriptor.id}")
    print(f"Name: {descriptor.name}")
    print(f"GeoParquet: {descriptor.geoparquet_uri}")
    print(f"Persisted: {'no' if args.no_persist else 'yes'}")
    return 0


def _handle_datasets_unregister_local(args: argparse.Namespace) -> int:
    from rasteret.catalog import unregister_local_descriptor

    descriptor = unregister_local_descriptor(
        args.dataset_id,
        path=args.registry_path,
    )
    if descriptor is None:
        print(f"Local dataset '{args.dataset_id}' not found.")
        return 1

    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(descriptor), indent=2))
        return 0

    print(f"Unregistered local dataset: {descriptor.id}")
    return 0


def _handle_datasets_export_local(args: argparse.Namespace) -> int:
    from rasteret.catalog import export_local_descriptor

    try:
        destination = export_local_descriptor(
            args.dataset_id,
            args.output_path,
            path=args.registry_path,
        )
    except KeyError as exc:
        print(str(exc))
        return 1

    payload = {"id": args.dataset_id, "path": str(destination)}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Exported local descriptor: {payload['id']}")
    print(f"Path: {payload['path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rasteret`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "build":
        return _handle_datasets_build(args)

    if args.command == "collections":
        if args.collections_command == "build":
            return _handle_collections_build(args)
        if args.collections_command == "list":
            return _handle_collections_list(args)
        if args.collections_command == "info":
            return _handle_collections_info(args)
        if args.collections_command == "delete":
            return _handle_collections_delete(args)
        if args.collections_command == "import":
            return _handle_collections_import(args)

    if args.command == "datasets":
        if args.ds_command == "list":
            return _handle_datasets_list(args)
        if args.ds_command == "info":
            return _handle_datasets_info(args)
        if args.ds_command == "build":
            return _handle_datasets_build(args)
        if args.ds_command == "register-local":
            return _handle_datasets_register_local(args)
        if args.ds_command == "unregister-local":
            return _handle_datasets_unregister_local(args)
        if args.ds_command == "export-local":
            return _handle_datasets_export_local(args)

    parser.error("Unsupported command")
    return 2
