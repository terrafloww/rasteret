# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Coroutine, Sequence
from datetime import datetime
from typing import Any, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import shapely
from affine import Affine
from pyproj import CRS

try:  # optional dependency
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:  # optional dependency
    from torchgeo.datasets import GeoDataset  # type: ignore
    from torchgeo.datasets.utils import GeoSlice  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GeoDataset = None
    GeoSlice = None

from rasteret.cloud import CloudConfig, backend_config_from_cloud_config
from rasteret.constants import BandRegistry
from rasteret.core.collection import Collection
from rasteret.core.geometry import bbox_array, coerce_to_geoarrow
from rasteret.core.utils import (
    compute_dst_grid,
    normalize_transform,
    reproject_array,
    transform_bbox,
    transform_polygon,
)
from rasteret.fetch.cog import COGReader, read_cog
from rasteret.types import CogMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T")

Sample = dict[str, Any]


class _AsyncCOGReaderPool:
    """Runs an asyncio loop + a persistent COGReader in a background thread."""

    def __init__(self, *, max_concurrent: int, backend: object | None = None) -> None:
        self.max_concurrent = max_concurrent
        self._backend = backend
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: COGReader | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def init() -> None:
            self._reader = COGReader(
                max_concurrent=self.max_concurrent,
                backend=self._backend,
            )
            await self._reader.__aenter__()

        try:
            loop.run_until_complete(init())
            self._loop = loop
        except BaseException as exc:
            self._error = exc
            self._loop = loop
        finally:
            self._ready.set()

        if self._error is not None:
            try:
                loop.close()
            finally:
                return

        try:
            loop.run_forever()
        finally:

            async def shutdown() -> None:
                if self._reader is not None:
                    await self._reader.__aexit__(None, None, None)

            try:
                loop.run_until_complete(shutdown())
            finally:
                loop.close()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the background loop and block for its result."""
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                "Failed to initialize async COG reader pool"
            ) from self._error
        if self._loop is None:
            raise RuntimeError("Event loop not initialized in COG reader pool")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def reader(self) -> COGReader:
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                "Failed to initialize async COG reader pool"
            ) from self._error
        if self._reader is None:
            raise RuntimeError("COG reader not initialized in pool")
        return self._reader

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready.wait()
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def __del__(self) -> None:
        self.close()


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        converted = value.as_py()
        if isinstance(converted, dict):
            return converted
    return None


def _coerce_label_value(value: Any) -> Any:
    """Convert common tabular label values to Torch-friendly tensors."""
    if torch is None:
        raise ImportError(
            "Torch is required for TorchGeo integration. Install rasteret[torchgeo]."
        )
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            if value.dtype.kind in {"i", "u", "b"}:
                return torch.as_tensor(value, dtype=torch.long)
            if value.dtype.kind == "f":
                return torch.as_tensor(value, dtype=torch.float32)
            return value.tolist()

    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind in {"i", "u", "b"}:
            return torch.as_tensor(array, dtype=torch.long)
        if array.dtype.kind == "f":
            return torch.as_tensor(array, dtype=torch.float32)
        return list(value)

    if isinstance(value, (bool, np.bool_)):
        return torch.tensor(int(value), dtype=torch.long)
    if isinstance(value, (int, np.integer)):
        return torch.tensor(int(value), dtype=torch.long)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return torch.tensor(float(value), dtype=torch.float32)

    if pd.isna(value):
        return None
    return value


if GeoDataset is not None and GeoSlice is not None and torch is not None:

    class RasteretGeoDataset(GeoDataset):
        """A TorchGeo ``GeoDataset`` backed by a Rasteret ``Collection``.

        Fetches COG tiles on-the-fly via async HTTP range reads and returns
        samples in the standard TorchGeo ``{"image": Tensor, ...}`` format.
        Compatible with all TorchGeo samplers, collation helpers, and transforms.
        """

        def __init__(
            self,
            collection: Collection,
            bands: Sequence[str],
            chip_size: int | None = None,
            is_image: bool = True,
            allow_resample: bool = False,
            label_field: str | None = None,
            geometries: Any = None,
            geometries_crs: int = 4326,
            transforms: Callable[[Sample], Sample] | None = None,
            cloud_config: CloudConfig | None = None,
            max_concurrent: int = 50,
            backend: object | None = None,
            time_series: bool = False,
            target_crs: int | None = None,
        ) -> None:
            """Initialize a Rasteret-backed TorchGeo dataset.

            Parameters
            ----------
            collection : Collection
                A Rasteret ``Collection`` (from ``build_from_stac``, ``load``, etc.).
            bands : Sequence[str]
                Band codes to load (e.g. ``["B04", "B03", "B02"]``).
            chip_size : int, optional
                Spatial chip size in pixels. Used for sampler hints.
            label_field : str, optional
                Column name in the collection table to use as a label.
            geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict, optional
                Spatial filter: only scenes intersecting these geometries are included.
                Accepts ``(minx, miny, maxx, maxy)`` bbox tuples, Arrow arrays
                (e.g. from GeoParquet), Shapely objects, raw WKB bytes, or GeoJSON dicts.
            geometries_crs : int
                EPSG code for *geometries*. Default ``4326``.
            transforms : callable, optional
                A TorchGeo-style transform ``fn(sample) -> sample``.
            cloud_config : CloudConfig, optional
                Cloud configuration (requester-pays, region, etc.).
            max_concurrent : int
                Maximum concurrent HTTP range requests. Default ``50``.
            backend : object, optional
                A ``StorageBackend`` instance (e.g. ``ObstoreBackend``).
                When ``None``, obstore is auto-detected if installed.
            time_series : bool
                If ``True``, stack all temporal scenes into a ``[T, C, H, W]``
                tensor per sample. Default ``False``.
            is_image : bool
                If ``True`` (default), return chips as ``sample[\"image\"]``.
                If ``False``, return chips as ``sample[\"mask\"]`` and squeeze
                the channel dimension when ``C == 1`` (TorchGeo-compatible).
            target_crs : int, optional
                EPSG code to reproject all scenes to.  When set, scenes from
                different CRS zones are reprojected on the fly.

            Raises
            ------
            ValueError
                If *bands* is empty, required columns are missing, or the
                collection contains no valid scenes.
            """
            if not bands:
                raise ValueError("At least one band is required")

            self.collection = collection
            self.bands = tuple(bands)
            self.chip_size = chip_size
            self.is_image = is_image
            self.allow_resample = bool(allow_resample)
            self.label_field = label_field
            self.time_series = time_series
            self.transforms = transforms
            self.max_concurrent = max_concurrent
            self._backend = backend
            self._pool: _AsyncCOGReaderPool | None = None
            self._pool_pid: int | None = None

            columns = [
                "id",
                "datetime",
                "assets",
                "proj:epsg",
                *[f"{b}_metadata" for b in self.bands],
            ]
            if label_field and label_field not in columns:
                columns.append(label_field)
            schema_names = self.collection.dataset.schema.names
            # collection column is optional - we prefer Collection.data_source
            if "collection" in schema_names:
                columns.append("collection")
            # STAC allows datetime=null when start/end_datetime are provided
            for _dt_col in ("start_datetime", "end_datetime"):
                if _dt_col in schema_names and _dt_col not in columns:
                    columns.append(_dt_col)
            missing_columns = [c for c in columns if c not in schema_names]
            if missing_columns:
                raise ValueError(
                    f"Collection is missing required columns: {missing_columns}"
                )

            table = self.collection.dataset.to_table(columns=columns)
            df = table.to_pandas()

            if df.empty:
                raise ValueError("Collection is empty")

            self.data_source = self.collection.data_source
            if not self.data_source and "collection" in df.columns:
                self.data_source = df["collection"].dropna().astype(str).iloc[0]

            if cloud_config is None:
                cloud_config = CloudConfig.get_config(self.data_source)

            if backend is None and cloud_config is not None:
                from rasteret.fetch.cog import _create_obstore_backend

                cfg = backend_config_from_cloud_config(cloud_config)
                if cfg:
                    backend = _create_obstore_backend(**cfg)
            self._backend = backend

            if "proj:epsg" not in df.columns:
                raise ValueError("Collection is missing required column 'proj:epsg'")

            epsg_series = df["proj:epsg"].dropna()
            if epsg_series.empty:
                raise ValueError("Collection records are missing 'proj:epsg'")

            if target_crs is not None:
                self.epsg = target_crs
                self._multi_crs = True
            else:
                self.epsg = int(epsg_series.iloc[0])
                self._multi_crs = False
                # Filter records to a single CRS.
                original_len = len(df)
                df = df[df["proj:epsg"] == self.epsg].reset_index(drop=True)
                if len(df) < original_len:
                    n_dropped = original_len - len(df)
                    n_zones = int(epsg_series.nunique())
                    logger.warning(
                        "%d of %d records dropped (CRS != EPSG:%d). "
                        "Collection spans %d CRS zones. Pass target_crs= "
                        "to reproject instead of dropping.",
                        n_dropped,
                        original_len,
                        self.epsg,
                        n_zones,
                    )
                if df.empty:
                    raise ValueError(f"No records with EPSG:{self.epsg}")
            crs = CRS.from_epsg(self.epsg)

            # Compute dataset resolution and record footprints using the first band metadata.
            first_band_meta = _as_dict(df.iloc[0][f"{self.bands[0]}_metadata"])
            if not first_band_meta:
                raise ValueError(f"Missing metadata for band '{self.bands[0]}'")

            try:
                scale_x, tx, scale_y, ty = normalize_transform(
                    first_band_meta.get("transform")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid transform for band '{self.bands[0]}': {exc}"
                ) from exc
            self._res = (abs(float(scale_x)), abs(float(scale_y)))

            # When multi-CRS, the source resolution may be in different
            # units than the target CRS (e.g. metres vs degrees).  Derive
            # the correct target-CRS resolution via GDAL's
            # calculate_default_transform so that compute_dst_grid receives
            # a resolution in the right units.
            if self._multi_crs:
                first_crs = int(df.iloc[0]["proj:epsg"])
                if first_crs != self.epsg:
                    from rasteret.core.utils import compute_dst_grid_from_src

                    w = int(first_band_meta["image_width"])
                    h = int(first_band_meta["image_height"])
                    xmin = float(tx)
                    xmax = float(tx) + w * float(scale_x)
                    ymax = float(ty)
                    ymin = float(ty) + h * float(scale_y)
                    src_bounds = (
                        min(xmin, xmax),
                        min(ymin, ymax),
                        max(xmin, xmax),
                        max(ymin, ymax),
                    )
                    dst_tf, _ = compute_dst_grid_from_src(
                        first_crs,
                        self.epsg,
                        w,
                        h,
                        src_bounds,
                    )
                    self._res = (abs(dst_tf.a), abs(dst_tf.e))

            # Ensure all requested bands have consistent resolution.
            self._resample_bands = False
            for band in self.bands[1:]:
                meta = _as_dict(df.iloc[0][f"{band}_metadata"])
                if not meta:
                    raise ValueError(f"Missing metadata for band '{band}'")
                try:
                    bx, _, by, _ = normalize_transform(meta.get("transform"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid transform for band '{band}': {exc}"
                    ) from exc
                same_x = np.isclose(
                    abs(float(bx)),
                    self._res[0],
                    rtol=1e-6,
                    atol=1e-9,
                )
                same_y = np.isclose(
                    abs(float(by)),
                    self._res[1],
                    rtol=1e-6,
                    atol=1e-9,
                )
                if not (same_x and same_y):
                    if self.allow_resample:
                        self._resample_bands = True
                        continue
                    raise ValueError(
                        "All requested bands must share the same resolution for TorchGeo sampling "
                        f"(expected {self._res}, got {(abs(float(bx)), abs(float(by)))} for band '{band}'). "
                        "To opt into resampling, pass allow_resample=True to Collection.to_torchgeo_dataset(...)."
                    )

            valid_rows: list[int] = []
            footprints: list[shapely.Geometry] = []
            intervals: list[tuple[datetime, datetime]] = []

            for i, row in df.iterrows():
                dt = row.get("datetime")
                if pd.isna(dt):
                    dt = row.get("start_datetime")
                if pd.isna(dt):
                    dt = row.get("end_datetime")
                if pd.isna(dt):
                    continue

                band_meta = _as_dict(row[f"{self.bands[0]}_metadata"])
                if not band_meta:
                    continue
                try:
                    sx, tx, sy, ty = normalize_transform(band_meta.get("transform"))
                except (TypeError, ValueError):
                    continue

                width = band_meta.get("image_width")
                height = band_meta.get("image_height")
                if width is None or height is None:
                    continue

                xmin = float(tx)
                xmax = float(tx) + float(width) * float(sx)
                ymax = float(ty)
                ymin = float(ty) + float(height) * float(sy)
                geom = shapely.box(
                    min(xmin, xmax),
                    min(ymin, ymax),
                    max(xmin, xmax),
                    max(ymin, ymax),
                )
                if self._multi_crs:
                    source_epsg = int(row.get("proj:epsg", self.epsg))
                    if source_epsg != self.epsg:
                        geom = transform_polygon(geom, source_epsg, self.epsg)
                footprints.append(geom)

                timestamp = pd.Timestamp(dt).to_pydatetime()
                intervals.append((timestamp, timestamp))
                valid_rows.append(i)

            if not valid_rows:
                raise ValueError("No valid records found for TorchGeo dataset creation")

            df = df.loc[valid_rows].reset_index(drop=True)

            # Keep a minimal TorchGeo-style index for compatibility with dataset
            # composition (IntersectionDataset/UnionDataset). TorchGeo calls
            # `dataset.index.reset_index()` during overlay; if the index name is
            # "datetime" then reset_index inserts a "datetime" column. Therefore,
            # Rasteret must not also include a "datetime" column in the index.
            #
            # We keep a separate in-memory payload table keyed by `rid` that
            # __getitem__ uses to fetch per-record metadata.
            payload_cols = [
                "assets",
                "proj:epsg",
                *[f"{b}_metadata" for b in self.bands],
            ]
            if self.label_field and self.label_field in df.columns:
                payload_cols.append(self.label_field)
            self._payload = df[payload_cols].copy()

            index = pd.IntervalIndex.from_tuples(
                intervals, closed="both", name="datetime"
            )
            index_df = pd.DataFrame({"rid": np.arange(len(self._payload), dtype=int)})
            index_df.index = index
            self.index = gpd.GeoDataFrame(index_df, geometry=footprints, crs=crs)

            if geometries is not None:
                geo_arr = coerce_to_geoarrow(geometries)
                xmin_arr, ymin_arr, xmax_arr, ymax_arr = bbox_array(geo_arr)
                # Compute union bbox of all input geometries
                union_bbox = (
                    pc.min(xmin_arr).as_py(),
                    pc.min(ymin_arr).as_py(),
                    pc.max(xmax_arr).as_py(),
                    pc.max(ymax_arr).as_py(),
                )
                if geometries_crs != self.epsg:
                    union_bbox = transform_bbox(union_bbox, geometries_crs, self.epsg)
                roi = shapely.box(*union_bbox)
                self.index = self.index[self.index.intersects(roi)].copy()

        def __getstate__(self) -> dict[str, Any]:
            """Drop non-pickleable state for multiprocessing DataLoader workers."""
            state = self.__dict__.copy()
            state["_pool"] = None
            state["_pool_pid"] = None
            return state

        def close(self) -> None:
            """Shut down the background async reader pool.

            Safe to call multiple times.  After ``close()``, the next
            ``__getitem__`` call will lazily spin up a fresh pool.
            """
            pool = self._pool
            self._pool = None
            self._pool_pid = None
            if pool is not None:
                pool.close()

        def _ensure_pool(self) -> _AsyncCOGReaderPool:
            pid = os.getpid()
            if self._pool is None or self._pool_pid != pid:
                # If we were forked, discard the old pool (threads don't survive fork safely).
                self._pool = _AsyncCOGReaderPool(
                    max_concurrent=self.max_concurrent,
                    backend=self._backend,
                )
                self._pool_pid = pid
            return self._pool

        def _build_band_requests(
            self,
            row: Any,
        ) -> list[tuple[str, CogMetadata, int | None]]:
            """Build (url, CogMetadata, band_index) tuples for a record row."""
            assets = _as_dict(row["assets"])
            if not assets:
                raise ValueError("Invalid assets in collection row")

            band_requests: list[tuple[str, CogMetadata, int | None]] = []
            for band in self.bands:
                # Support both legacy and normalized asset-key conventions.
                candidates: list[str] = [band]
                band_map = BandRegistry.get(self.data_source)
                forward = band_map.get(band)
                if forward:
                    candidates.append(forward)
                if band_map and band in band_map.values():
                    reverse = {v: k for k, v in band_map.items()}
                    back = reverse.get(band)
                    if back:
                        candidates.append(back)

                asset_key = next((c for c in candidates if c in assets), None)
                asset = assets.get(asset_key) if asset_key is not None else None
                if not isinstance(asset, dict) or "href" not in asset:
                    raise KeyError(
                        f"Missing asset href for band '{band}' (tried {candidates})"
                    )

                url: str = asset["href"]
                band_index = asset.get("band_index")
                idx: int | None = None
                if band_index is not None:
                    try:
                        idx = int(band_index)
                    except (TypeError, ValueError):
                        idx = None

                meta = _as_dict(row.get(f"{band}_metadata"))
                if not meta and asset_key is not None and asset_key != band:
                    meta = _as_dict(row.get(f"{asset_key}_metadata"))
                if not meta:
                    raise KeyError(f"Missing metadata for band '{band}'")

                try:
                    transform = list(normalize_transform(meta.get("transform")))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid transform for band '{band}': {exc}"
                    ) from exc

                source_crs = int(row["proj:epsg"]) if self._multi_crs else self.epsg
                band_requests.append(
                    (
                        url,
                        CogMetadata.from_dict(
                            meta,
                            crs=source_crs,
                            transform_override=transform,
                        ),
                        idx,
                    )
                )
            return band_requests

        def _fetch_arrays(
            self,
            requests: list[tuple[str, CogMetadata, int | None]],
            patch_array: pa.Array,
            pool: _AsyncCOGReaderPool,
            *,
            out_shape: tuple[int, int] | None,
        ) -> list[tuple[np.ndarray, Any, np.ndarray]]:
            """Fetch pixel arrays for all (url, metadata) pairs concurrently.

            Reuses the pool's persistent COGReader (and its obstore backend)
            so every request shares the same connection pool.
            Returns one (2-D array, Affine transform) per request.
            """
            reader = pool.reader

            async def _gather() -> list[tuple[np.ndarray, Any, np.ndarray]]:
                tasks = [
                    read_cog(
                        url,
                        meta,
                        band_index=band_index,
                        geom_array=patch_array,
                        geom_idx=0,
                        geometry_crs=self.epsg,
                        max_concurrent=self.max_concurrent,
                        reader=reader,
                        mode="window",
                        out_shape=out_shape,
                    )
                    for url, meta, band_index in requests
                ]
                results = list(await asyncio.gather(*tasks))
                return [(r.data, r.transform, r.valid_mask) for r in results]

            return pool.run(_gather())

        def _payload_row(self, rid: int) -> Any:
            try:
                return self._payload.iloc[int(rid)]
            except Exception as exc:  # pragma: no cover
                raise IndexError(f"Invalid rid={rid}") from exc

        def __getitem__(self, index: GeoSlice) -> Sample:
            """Return a sample dict for the given spatio-temporal slice.

            Parameters
            ----------
            index : GeoSlice
                A TorchGeo ``GeoSlice`` (x, y, t ranges) produced by a sampler.

            Returns
            -------
            dict[str, Any]
                ``{"image": Tensor, "bounds": Tensor, "transform": Tensor}``
                plus optional ``"label"`` if *label_field* was set.
                Image shape is ``[C, H, W]`` (single timestep) or
                ``[T, C, H, W]`` (time series).
            """
            x, y, t = self._disambiguate_slice(index)

            t_step = 1 if t.step is None else int(t.step)
            if self.time_series:
                # TorchGeo semantics: time_series stacks all spatially overlapping
                # records, regardless of the sampler-provided time slice.
                df = self.index.cx[x.start : x.stop, y.start : y.stop]
                df = df.iloc[::t_step]
            else:
                interval = pd.Interval(t.start, t.stop, closed="both")
                df = self.index.iloc[self.index.index.overlaps(interval)]
                df = df.iloc[::t_step]
                df = df.cx[x.start : x.stop, y.start : y.stop]

            if df.empty:
                raise IndexError(
                    f"index: {index} not found in dataset with bounds: {self.bounds}"
                )

            pool = self._ensure_pool()
            # Create GeoArrow patch from sampler bbox (no Shapely on fetch path)
            patch_array = coerce_to_geoarrow((x.start, y.start, x.stop, y.stop))
            n_bands = len(self.bands)
            # TorchGeo samplers produce bounds aligned to integer multiples of
            # `dataset.res`, but floating point representation can still yield
            # off-by-one in ceil-based shape computations. For ML chips we
            # prefer fixed-size outputs when `chip_size` is set.
            if self.chip_size is not None:
                dst_shape = (int(self.chip_size), int(self.chip_size))
                dst_transform = Affine(
                    float(x.step),
                    0.0,
                    float(x.start),
                    0.0,
                    -float(y.step),
                    float(y.stop),
                )
            else:
                dst_transform, dst_shape = compute_dst_grid(
                    (x.start, y.start, x.stop, y.stop),
                    self._res,
                )

            def _auto_resampling_for_dtype(dtype: np.dtype) -> str:
                # Match TorchGeo RasterDataset default: bilinear for float, nearest for int.
                if np.issubdtype(dtype, np.floating):
                    return "bilinear"
                return "nearest"

            def _warp_to_grid(
                arr: np.ndarray,
                src_transform: object,
                *,
                resampling: str,
            ) -> np.ndarray:
                from rasterio.crs import CRS as RioCRS
                from rasterio.warp import Resampling, reproject

                dst = np.empty(dst_shape, dtype=arr.dtype)
                dst.fill(0)
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=src_transform,
                    src_crs=RioCRS.from_epsg(self.epsg),
                    dst_transform=dst_transform,
                    dst_crs=RioCRS.from_epsg(self.epsg),
                    resampling=getattr(Resampling, resampling),
                )
                return dst

            fetch_out_shape = (
                None if (self._multi_crs or self._resample_bands) else dst_shape
            )

            if self.time_series:
                # Sort chronologically so T dimension is time-ordered.
                df = df.sort_index()

                # Build all T×C requests, fire concurrently via asyncio.gather.
                all_requests: list[tuple[str, CogMetadata, int | None]] = []
                n_timesteps = len(df)
                source_crs_per_request: list[int] = []
                for rid in df["rid"].to_list():
                    row = self._payload_row(int(rid))
                    band_requests = self._build_band_requests(row)
                    all_requests.extend(band_requests)
                    if self._multi_crs:
                        src_crs = int(row["proj:epsg"])
                        source_crs_per_request.extend([src_crs] * len(band_requests))

                fetch_results = self._fetch_arrays(
                    all_requests,
                    patch_array,
                    pool,
                    out_shape=fetch_out_shape,
                )

                # When multi-CRS, reproject every array to the target CRS grid.
                # This guarantees consistent (H, W) across all records/bands.
                if self._multi_crs:
                    arrays = [
                        reproject_array(
                            arr,
                            aff,
                            src_crs,
                            self.epsg,
                            dst_transform,
                            dst_shape,
                        )
                        for (arr, aff, _mask), src_crs in zip(
                            fetch_results,
                            source_crs_per_request,
                        )
                    ]
                elif self._resample_bands:
                    arrays = [
                        _warp_to_grid(
                            arr,
                            aff,
                            resampling=_auto_resampling_for_dtype(arr.dtype),
                        )
                        for (arr, aff, _mask) in fetch_results
                    ]
                else:
                    arrays = [r[0] for r in fetch_results]

                # arrays is flat: [t0_band0, t0_band1, ..., t1_band0, ...]
                # Reshape into [T, C, H, W] with minimal copies:
                #   np.stack at band level  → [C, H, W] (1 contiguous alloc per timestep)
                #   np.stack at time level  → [T, C, H, W] (1 contiguous alloc)
                #   torch.from_numpy        → zero-copy view
                timesteps = [
                    np.stack(arrays[t_idx * n_bands : (t_idx + 1) * n_bands], axis=0)
                    for t_idx in range(n_timesteps)
                ]
                image = torch.from_numpy(np.stack(timesteps, axis=0))  # [T, C, H, W]

            else:
                # Minimal TorchGeo adapter behavior: select a single record for this
                # spatiotemporal slice. (TorchGeo's RasterDataset mosaics multiple
                # overlapping files, but Rasteret avoids reimplementing merge
                # semantics here.)
                if len(df) > 1:
                    logger.warning(
                        "TorchGeo slice overlaps %d records; selecting the first record "
                        "(set time_series=True to stack multiple timesteps).",
                        len(df),
                    )

                rid = int(df.iloc[0]["rid"])
                row = self._payload_row(rid)
                band_requests = self._build_band_requests(row)
                fetch_results = self._fetch_arrays(
                    band_requests,
                    patch_array,
                    pool,
                    out_shape=fetch_out_shape,
                )
                if not fetch_results:
                    raise ValueError("No bands were fetched for the requested chip")

                if self._multi_crs:
                    src_crs = int(row["proj:epsg"])
                    arrays = [
                        reproject_array(
                            arr,
                            aff,
                            src_crs,
                            self.epsg,
                            dst_transform,
                            dst_shape,
                        )
                        for arr, aff, _mask in fetch_results
                    ]
                elif self._resample_bands:
                    arrays = [
                        _warp_to_grid(
                            arr,
                            aff,
                            resampling=_auto_resampling_for_dtype(arr.dtype),
                        )
                        for arr, aff, _mask in fetch_results
                    ]
                else:
                    arrays = [r[0] for r in fetch_results]

                # np.stack → [C, H, W] (1 alloc), torch.from_numpy → zero-copy.
                image = torch.from_numpy(np.stack(arrays, axis=0))  # [C, H, W]

            transform = torch.tensor(
                [x.step, 0.0, x.start, 0.0, -y.step, y.stop], dtype=torch.float32
            )

            sample: Sample = {
                "bounds": self._slice_to_tensor(index),
                "transform": transform,
            }
            if self.is_image:
                sample["image"] = image
            else:
                # TorchGeo RasterDataset squeezes channel dim for mask-style datasets.
                if self.time_series:
                    # [T, C, H, W] -> [T, H, W] when C==1
                    sample["mask"] = image.squeeze(1)
                else:
                    # [C, H, W] -> [H, W] when C==1
                    sample["mask"] = image.squeeze(0)
            if self.label_field is not None:
                label_value = None
                if not df.empty:
                    rid0 = int(df.iloc[0]["rid"])
                    label_row = self._payload_row(rid0)
                    label_value = _coerce_label_value(label_row.get(self.label_field))
                if label_value is not None:
                    sample["label"] = label_value
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample

else:

    class RasteretGeoDataset:  # pragma: no cover
        """Stub when Torch/TorchGeo are not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError(
                "TorchGeo integration requires torch + torchgeo. "
                "Install rasteret[torchgeo]."
            )
