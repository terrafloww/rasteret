# Conceptual Roadmap

Whether you are a deep learning researcher, a data scientist, or a geospatial engineer, your workflow usually starts with a question: *"I need these 10 bands from these 5,000 scenes to train this model."*

Rasteret was built to shorten the distance between that question and the final pixel buffer. This guide explains the core concepts that make that possible.

## 1. From Verbs to Nouns

Traditional geospatial libraries are built around **Verbs**. You are responsible for orchestrating the steps of the data lifecycle:
- `search()` the catalog...
- `list()` the items...
- `open()` the files...
- `read()` the windows...
- `align()` the CRS...
- `stack()` the arrays...

**Rasteret turns this lifecycle into a Noun: the `Collection`.**

When you have a `Collection`, you don't manage the steps; you manage the state. Filtering, indexing, and I/O orchestration are internal details of the object. You only ask for the output (`get_numpy`, `to_torchgeo_dataset`).

## 2. Relational Imagery (MetaData + Pixels)

The biggest friction in AI workflows is the disconnect between **tabular metadata** (your labels, your training splits) and **raster payloads** (the actual pixels).

Traditionally, you keep these in separate places—a CSV for your labels and a bucket of TIFFs for your imagery. You then write complex code to join them together at runtime.

**In Rasteret, pixels are just another column in your table.**
Because a Collection is backed by a GeoParquet index, you can treat satellite scenes like rows in a table.
- Want to join your soil sensor readings to the specific satellite scene they overlap? **It's a join.**
- Want to add a `is_training` flag to 40% of your scenes? **It's a column update.**

## 3. The "Data Engineer" in a Box

High-performance cloud I/O is hard. To get good throughput from S3 or GCS, you need to:
- Coalesce small requests to avoid request-latency overhead.
- Handle HTTP/2 concurrency without overwhelming the client.
- Parse TIFF headers efficiently without downloading the whole file.

Rasteret includes a specialized **I/O Engine** (built in Rust via `obstore` and `asyncio`) that acts as your dedicated data engineer. It knows how to "plan" the most efficient way to fetch your pixels, so you can focus on your model architecture instead of your TCP connection pooling.

## 4. Reproducibility as a File

Sharing a dataset today usually means sharing a massive folder of image "chips" or a complex script that might fail when the STAC API changes.

**A Rasteret Collection is a single folder (or S3 prefix) that you can share.**
When you `collection.export("my_experiment")`, you are saving the exact scene IDs, the exact splits, the exact labels, and the exact tile-layout metadata used for your study. Anyone can `rasteret.load("my_experiment")` and get the same identical pixels with zero "warm-up" time.

---

### Core Concept Map

| Concept | Traditional View | Rasteret View |
|---|---|---|
| **Identity** | A folder of TIFF files. | A queryable Parquet Table. |
| **Search** | An API call to a STAC endpoint. | A local operation on your Index. |
| **Labels** | A separate `.csv` or `.json` file. | A column in the Collection table. |
| **I/O** | `rasterio.open()` in a loop. | A batched `get_numpy()` call. |
| **Speed** | Determined by your multithreading code. | Determined by the I/O Coalescing engine. |

## Next Step: Managing the Shift

If you are currently using Rasterio or a standard STAC-client workflow, the next step is to see exactly how your code will change and the boilerplate you can delete.

👉 [**Transitioning from Rasterio & STAC**](../how-to/transitioning-from-rasterio.md)
