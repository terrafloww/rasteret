# TorchGeo Integration

Use this page when you want a Rasteret Collection to behave like a standard TorchGeo `GeoDataset`.

Rasteret does not replace TorchGeo samplers, transforms, collation, or training loops. It provides the dataset reader: Rasteret uses its collection metadata and COG byte-range reader, then returns samples through TorchGeo's expected dataset contract.

## Create A GeoDataset

```python
import rasteret

collection = rasteret.load("my_experiment")

dataset = collection.to_torchgeo_dataset(
    bands=["B08", "B04", "B03"],
    chip_size=256,
)
```

The returned object is a TorchGeo `GeoDataset`, so standard TorchGeo samplers and PyTorch dataloaders can use it:

```python
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

sampler = RandomGeoSampler(dataset, size=256, length=128)
loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=8,
    num_workers=4,
    collate_fn=stack_samples,
)
```

## Filter Before Creating The Dataset

You can filter the collection first:

```python
train = collection.subset(split="train", cloud_cover_lt=20)

dataset = train.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)
```

Or pass common filters directly:

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
    split="train",
    cloud_cover_lt=20,
    date_range=("2024-01-01", "2024-06-30"),
)
```

## Include Labels

If your collection table has a label column, pass it with `label_field`:

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04"],
    label_field="biomass_value",
    chip_size=256,
)
```

Samples include the label under `sample["label"]`.

For adding split and label columns before this step, see [Bring Your Own AOIs, Points, And Metadata](https://terrafloww.github.io/rasteret/how-to/enriched-collection-workflows/index.md). For benchmark methodology and current numbers, see [Benchmarks](https://terrafloww.github.io/rasteret/explanation/benchmark/index.md) and the [TorchGeo Benchmark: Rasteret vs Native Rasterio notebook](https://terrafloww.github.io/rasteret/tutorials/05_torchgeo_benchmark_rasteret_vs_rasterio/index.md).
