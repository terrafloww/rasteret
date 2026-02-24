# ML Training with Splits and Labels

!!! note
    Requires `rasteret[torchgeo]`.

In TorchGeo, datasets ship with pre-defined train/val/test splits baked
into the dataset class. In Rasteret, splits and labels are **columns in
the Parquet index**. You add them yourself, which gives you full control
over partitioning strategy and makes the assignments reproducible and
shareable.

## 1. Build a collection

```python
from pathlib import Path
import rasteret

bbox = (77.55, 13.01, 77.58, 13.08)

collection = rasteret.build_from_stac(
    name="bangalore",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection="sentinel-2-l2a",
    bbox=bbox,
    date_range=("2024-01-01", "2024-03-31"),
    workspace_dir=Path.home() / "rasteret_workspace",
)
```

## 2. Assign splits

Before filtering by split, the collection needs a `split` column:

```python
import pyarrow as pa
import numpy as np

table = collection.dataset.to_table()
n = table.num_rows

rng = np.random.default_rng(42)
splits = rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15])
table = table.append_column("split", pa.array(splits))

# Optional: add a label column (e.g. land-cover class per scene)
labels = rng.integers(0, 5, size=n)
table = table.append_column("label", pa.array(labels, type=pa.int32()))

# Save the enriched table and reload as a Collection
import pyarrow.parquet as pq
pq.write_table(table, "./with_splits.parquet")
collection = rasteret.load("./with_splits.parquet")
```

The split column travels with the Parquet file. Reload later with
`rasteret.load("./with_splits.parquet")` and the splits are preserved.

## 3. Create TorchGeo datasets per split

`split="train"` filters the Parquet index before creating the dataset.
`label_field="label"` includes the label column (added in step 2) in each
sample as `sample["label"]`.

```python
train_dataset = collection.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    geometries=bbox,
    split="train",
    label_field="label",
    chip_size=256,
)

val_dataset = collection.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    geometries=bbox,
    split="val",
    chip_size=256,
)
```

See [`to_torchgeo_dataset()`](../reference/integrations/torchgeo.md) API reference.

## 4. Train

Everything below is standard TorchGeo:

```python
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

sampler = RandomGeoSampler(train_dataset, size=256, length=32)
loader = DataLoader(
    train_dataset,
    sampler=sampler,
    batch_size=4,
    num_workers=0,
    collate_fn=stack_samples,
)

for batch in loader:
    print(f"image: {batch['image'].shape}")
    if "label" in batch:
        print(f"label: {batch['label']}")
    break
```

The full runnable script is at `examples/ml_training_with_splits.py`.
