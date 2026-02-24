"""Production ML pipeline: build collection, assign splits, train.

Demonstrates the full workflow from STAC search to TorchGeo DataLoader:
1. Build a collection from STAC (cached after first run)
2. Assign train/val/test splits using PyArrow
3. Save the split-annotated collection as a shareable Parquet artifact
4. Create TorchGeo datasets for each split
5. Run a standard training loop

Requires: rasteret[torchgeo]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

import rasteret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-dir",
        default=str(Path.home() / "rasteret_workspace"),
        help="Directory for cached collections",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Train split ratio"
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--chip-size", type=int, default=256)
    return parser.parse_args()


def assign_splits(
    collection: rasteret.Collection,
    output_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> rasteret.Collection:
    """Add a 'split' column to a collection and save it.

    Uses deterministic random assignment so the same collection always
    gets the same splits (reproducible across runs and machines).
    """
    table = collection.dataset.to_table()
    n = len(table)

    rng = np.random.default_rng(seed)
    assignments = rng.random(n)
    splits = np.where(
        assignments < train_ratio,
        "train",
        np.where(assignments < train_ratio + val_ratio, "val", "test"),
    )

    table = table.append_column("split", pa.array(splits))

    # Persist to a new Parquet dataset
    output_path.mkdir(parents=True, exist_ok=True)
    partition_cols = [c for c in ("year", "month") if c in table.schema.names]
    ds.write_dataset(
        table,
        output_path,
        format="parquet",
        partitioning=partition_cols or None,
        existing_data_behavior="overwrite_or_ignore",
    )

    return rasteret.load(output_path, name=collection.name)


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_dir)

    bbox = (77.55, 13.01, 77.58, 13.08)

    # Step 1: Build collection (cached after first run)
    collection = rasteret.build_from_stac(
        name="bangalore",
        stac_api="https://earth-search.aws.element84.com/v1",
        collection="sentinel-2-l2a",
        bbox=bbox,
        date_range=("2024-01-01", "2024-06-30"),
        workspace_dir=workspace,
    )
    print(f"Collection: {collection.name}, rows={collection.dataset.count_rows()}")

    # Step 2: Assign splits and save as a shareable artifact
    split_path = workspace / "bangalore_with_splits"
    if (
        split_path.exists()
        and "split" in rasteret.load(split_path).dataset.schema.names
    ):
        print("Loading existing split-annotated collection...")
        collection = rasteret.load(split_path, name=collection.name)
    else:
        print("Assigning train/val/test splits...")
        collection = assign_splits(
            collection,
            output_path=split_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    # Show split distribution
    table = collection.dataset.to_table(columns=["split"])
    for split_name in ["train", "val", "test"]:
        count = pc.sum(pc.equal(table.column("split"), split_name)).as_py()
        print(f"  {split_name}: {count} rows")

    # Step 3: Create TorchGeo datasets per split
    from torch.utils.data import DataLoader
    from torchgeo.datasets.utils import stack_samples
    from torchgeo.samplers import RandomGeoSampler

    train_ds = collection.to_torchgeo_dataset(
        bands=["B04", "B03", "B02", "B08"],
        geometries=bbox,
        split="train",
        chip_size=args.chip_size,
    )

    sampler = RandomGeoSampler(train_ds, size=args.chip_size, length=args.num_samples)
    loader = DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=stack_samples,
    )

    # Step 4: Training loop (placeholder, replace with your model)
    for i, batch in enumerate(loader):
        img = batch["image"]
        print(f"  batch {i}: image={img.shape}, dtype={img.dtype}")
        if i >= 2:
            break

    # Step 5: Validation set
    val_collection = collection.subset(split="val")
    print(f"\nVal split: {val_collection.dataset.count_rows()} rows")
    print(f"\nShareable artifact saved at: {split_path}")


if __name__ == "__main__":
    main()
