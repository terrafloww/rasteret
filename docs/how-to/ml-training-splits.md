# ML Training With Splits

Use this page when your Rasteret collection already represents the rasters you
want to train on, and you need split or label columns for a TorchGeo workflow.

In Rasteret, splits and labels are table columns. Add them with an Arrow-native
tool, wrap the table back with `as_collection()`, then pass the column names to
TorchGeo.

## Add Split And Label Columns

```python
import numpy as np
import polars as pl
import rasteret

frame = pl.from_arrow(collection)
n = frame.height
rng = np.random.default_rng(42)

frame = frame.with_columns(
    pl.Series(
        "split",
        rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15]),
    ),
    pl.Series("label", rng.integers(0, 5, size=n), dtype=pl.Int32),
)

experiment = rasteret.as_collection(
    frame,
    name="training-experiment-v1",
    data_source=collection.data_source,
)
```

Passing `data_source` preserves source-specific band mapping and avoids relying
on schema metadata that table engines may not round-trip exactly.

For joins with external GIS labels or AOI tables, see
[Enriched Parquet Workflows](enriched-parquet-workflows.md).

## Create Datasets Per Split

`split="train"` filters the collection before creating the TorchGeo dataset.
`label_field="label"` includes the label column as `sample["label"]`.

```python
train_dataset = experiment.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    split="train",
    label_field="label",
    chip_size=256,
)

val_dataset = experiment.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    split="val",
    label_field="label",
    chip_size=256,
)
```

## Train With TorchGeo

Everything after dataset creation is standard TorchGeo and PyTorch:

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
    print(batch["image"].shape)
    print(batch["label"])
    break
```

To persist the enriched collection for later runs:

```python
experiment.export("./training_experiment_v1")
reloaded = rasteret.load("./training_experiment_v1")
```

See [`to_torchgeo_dataset()`](../reference/integrations/torchgeo.md) for the
full API.
