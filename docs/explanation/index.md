# Explanation

Background material that helps you understand *why* Rasteret works the
way it does.

Practical workflow lens:
`build/load/as_collection -> subset/where -> get_xarray/get_numpy/get_gdf/sample_points`.

- [Architecture](architecture.md): component overview and data flow
- [Design Decisions](design-decisions.md): why Parquet, why a custom reader, why COGs stay COGs, and why these choices enable an open community
- [Correctness Contract](correctness.md): the user-visible guarantees Rasteret makes (and the contract contributors must preserve)
- [Schema Contract](schema-contract.md): column guarantees for contributors building ingest drivers
- [Ecosystem Comparison](interop.md): how Rasteret relates to TorchGeo, xarray, rasterio, and similar tools
- [Benchmarks](benchmark.md): apples-to-apples performance comparison
- [TorchGeo Benchmark](../tutorials/05_torchgeo_comparison.ipynb): interactive notebook - Rasteret vs TorchGeo native, side-by-side

Every design choice here (Parquet indexes, open catalog descriptors, decoupled
storage backends) is made to keep Rasteret interoperable and community-driven.
The catalog format is intentionally **spec-aligned** and evolving toward a
portable standard shareable across tools, not just Rasteret.

These are deliberate trade-offs, not accidental ones. If you have ideas,
see something that could work differently, or want to discuss the reasoning,
open a [Discussion](https://github.com/terrafloww/rasteret/discussions).
The design benefits from more perspectives.
