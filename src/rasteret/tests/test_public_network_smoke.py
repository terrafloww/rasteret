# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Public network smoke tests (no auth).

These tests run by default and exercise Rasteret against real public endpoints:

- Anonymous S3 range reads (Sentinel-2 COGs)
- Anonymous GCS range reads (public Landsat index)
- End-to-end GeoParquet -> COG enrichment -> pixel read (AEF on Source Cooperative)

They are intentionally small and deterministic to keep CI coverage aligned with
the user-facing docs without requiring notebooks/examples to run.
"""

from __future__ import annotations

from pathlib import Path

import rasteret


class _AlarmTimeout(Exception):
    pass


def _timeout(seconds: int):
    """Best-effort wall-clock timeout for live network tests (POSIX only)."""
    import contextlib
    import signal

    @contextlib.contextmanager
    def _ctx():
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(_signum, _frame):  # type: ignore[no-untyped-def]
            raise _AlarmTimeout(f"Timed out after {seconds}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    return _ctx()


# A known-stable Sentinel-2 COG in the public sentinel-cogs bucket (us-west-2).
_S2_KEY = "sentinel-s2-l2a-cogs/1/C/CV/2024/1/S2B_1CCV_20240106_0_L2A/B02.tif"
_S2_HTTPS = f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{_S2_KEY}"


def test_s3store_reads_sentinel2_cog_header() -> None:
    """S3Store with skip_signature reads first 8 bytes of a Sentinel-2 COG."""
    import obstore as obs
    from obstore.store import S3Store

    store = S3Store(
        bucket="sentinel-cogs",
        config={"skip_signature": "true", "region": "us-west-2"},
    )
    buf = obs.get_range(store, _S2_KEY, start=0, length=8)
    data = bytes(buf)
    assert len(data) == 8
    # TIFF files start with II (little-endian) or MM (big-endian)
    assert data[:2] in (b"II", b"MM")


def test_auto_backend_routes_s3_url() -> None:
    """_AutoObstoreBackend routes a real S3 URL to S3Store and reads bytes."""
    import asyncio

    from rasteret.fetch.cog import _create_obstore_backend

    backend = _create_obstore_backend()

    async def _read():
        return await backend.get_range(_S2_HTTPS, start=0, length=8)

    data = asyncio.run(_read())
    assert len(data) == 8
    assert data[:2] in (b"II", b"MM")
    # Verify it was routed to S3Store, not HTTPStore
    assert "s3://sentinel-cogs" in backend._stores


# A known-stable file in the GCS public Landsat bucket (index.csv.gz).
# Using the index file rather than individual scenes avoids breakage
# when Google reorganises Landsat paths.
_GCS_BUCKET = "gcp-public-data-landsat"
_GCS_KEY = "index.csv.gz"
_GCS_HTTPS = f"https://storage.googleapis.com/{_GCS_BUCKET}/{_GCS_KEY}"


def test_gcs_store_reads_public_file() -> None:
    """GCSStore with skip_signature reads bytes from a public GCS bucket."""
    import obstore as obs
    from obstore.store import GCSStore

    store = GCSStore(
        bucket=_GCS_BUCKET,
        config={"skip_signature": "true"},
    )
    buf = obs.get_range(store, _GCS_KEY, start=0, length=4)
    data = bytes(buf)
    assert len(data) == 4
    # gzip magic number: 0x1f 0x8b
    assert data[:2] == b"\x1f\x8b"


def test_auto_backend_routes_gcs_https_url() -> None:
    """_AutoObstoreBackend routes a GCS HTTPS URL to GCSStore and reads bytes."""
    import asyncio

    from rasteret.fetch.cog import _create_obstore_backend

    backend = _create_obstore_backend()

    async def _read():
        return await backend.get_range(_GCS_HTTPS, start=0, length=4)

    data = asyncio.run(_read())
    assert len(data) == 4
    assert data[:2] == b"\x1f\x8b"
    assert f"gs://{_GCS_BUCKET}" in backend._stores


# AlphaEarth Embeddings (AEF): public multi-sample tiled GeoTIFF hosted on Source Cooperative.
_AEF_URL = (
    "https://data.source.coop/tge-labs/aef/v1/annual/2023/32N/"
    "xfj3s6mouxk5zgq0y-0000008192-0000008192.tiff"
)


def test_aef_build_enrich_and_read_matches_rasterio(tmp_path: Path) -> None:
    import asyncio

    import numpy as np
    import rasterio
    from rasterio.windows import Window

    from rasteret.fetch.header_parser import AsyncCOGHeaderParser

    bands = ["A00", "A01", "A31", "A63"]
    band_indices = [0, 1, 31, 63]
    rasterio_band_numbers = [i + 1 for i in band_indices]  # 1-based

    # Use a bbox that falls entirely within a single AEF tile to ensure
    # the rasterio pixel comparison hits the same tile as the xarray read.
    query_bbox = (11.35, -0.5, 11.45, -0.4)  # WGS84, single tile

    with _timeout(180):
        collection = rasteret.build(
            "aef/v1-annual",
            name="public-network-aef",
            bbox=query_bbox,
            date_range=("2023-01-01", "2023-12-31"),
            workspace_dir=tmp_path / "public-network-aef",
            force=True,
            max_concurrent=8,
        )

    assert collection.dataset is not None
    assert collection.dataset.count_rows() >= 1

    # Extract the real COG URL we selected (all bands share the same href).
    row = collection.dataset.to_table(columns=["assets"]).to_pylist()[0]
    url = row["assets"]["A00"]["href"]

    from rasteret.cloud import CloudConfig, rewrite_url

    # Prefer stable, anonymous S3 range reads over HTTPS for Source Cooperative.
    # Some HTTP frontends intermittently return short Content-Range responses.
    cloud_config = CloudConfig(
        provider="aws",
        region="us-west-2",
        requester_pays=False,
        url_patterns={
            "https://data.source.coop/": "s3://us-west-2.opendata.source.coop/",
        },
    )
    resolved_url = rewrite_url(url, cloud_config)

    # Confirm the raw header contains the full multi-sample tile tables.
    async def _parse_raw():
        async with AsyncCOGHeaderParser(max_concurrent=8, batch_size=1) as p:
            return (await p.process_cog_headers_batch([resolved_url]))[0]

    raw_md = asyncio.run(_parse_raw())
    assert raw_md is not None
    assert raw_md.compression == 50000  # TIFF ZSTD tag

    # Read a small region and compare per-pixel values against rasterio.
    with _timeout(180):
        ds = collection.get_xarray(
            geometries=query_bbox, bands=bands, cloud_config=cloud_config
        )

    arr0 = ds[bands[0]].values.squeeze()
    nodata = raw_md.nodata
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        nodata = 0
    valid_mask = arr0 != nodata
    assert bool(valid_mask.any()), "No non-nodata pixels found in AEF window"

    rows, cols = np.where(valid_mask)
    idx = len(rows) // 2
    r, c = int(rows[idx]), int(cols[idx])

    x_center = float(ds["x"].values[c])
    y_center = float(ds["y"].values[r])

    rasteret_vals = [float(ds[band].values.squeeze()[r, c]) for band in bands]
    assert len(set(int(v) for v in rasteret_vals)) > 1

    with rasterio.open(resolved_url) as src:
        assert src.count >= 64
        rr, cc = src.index(x_center, y_center)
        for band_name, band_num, expected in zip(
            bands, rasterio_band_numbers, rasteret_vals
        ):
            got = float(src.read(band_num, window=Window(cc, rr, 1, 1))[0, 0])
            assert got == expected, (
                f"RasterIO mismatch for {band_name} at row={rr}, col={cc}: "
                f"rasterio={got}, rasteret={expected}"
            )

    # Sanity: the public URL constant stays reachable (helps catch provider moves).
    with rasterio.open(_AEF_URL) as src:
        assert src.count >= 64
