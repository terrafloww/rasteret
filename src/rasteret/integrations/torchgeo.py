# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import warnings
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
    from torchgeo.datasets.utils import (  # type: ignore
        GeoSlice,
    )
    from torchgeo.datasets.utils import (
        array_to_tensor as _torchgeo_array_to_tensor,
    )
except ModuleNotFoundError:  # pragma: no cover
    GeoDataset = None
    GeoSlice = None
    _torchgeo_array_to_tensor = None

from rasteret.cloud import CloudConfig, backend_config_from_cloud_config
from rasteret.constants import BandRegistry
from rasteret.core.collection import Collection
from rasteret.core.geometry import bbox_array, coerce_to_geoarrow
from rasteret.core.rio_semantics import MergeGrid, merge_semantic_resample_single_source
from rasteret.core.utils import (
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

_TORCHGEO_DTYPE_CAST_MAP: dict[np.dtype[Any], np.dtype[Any]] = {
    np.dtype(np.uint16): np.dtype(np.int32),
    np.dtype(np.uint32): np.dtype(np.int64),
}


def _array_to_image_tensor_torchgeo_compatible(
    array: np.ndarray,
) -> tuple["torch.Tensor", tuple[np.dtype[Any], np.dtype[Any]] | None]:
    """Convert image array to tensor with TorchGeo-compatible dtype mapping.

    Mapping mirrors ``torchgeo.datasets.utils.array_to_tensor``:
    uint16 -> int32, uint32 -> int64.

    For all other dtypes, this path uses ``torch.from_numpy`` for zero-copy when
    the input is C-contiguous.
    """
    if torch is None:
        raise ImportError(
            "Torch is required for TorchGeo integration. Install rasteret[torchgeo]."
        )

    source_dtype = np.dtype(array.dtype)
    target_dtype = _TORCHGEO_DTYPE_CAST_MAP.get(source_dtype, source_dtype)
    cast_info: tuple[np.dtype[Any], np.dtype[Any]] | None = None

    if target_dtype != source_dtype:
        cast_info = (source_dtype, target_dtype)
        array = array.astype(target_dtype, copy=False)

    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    try:
        return torch.from_numpy(array), cast_info
    except (TypeError, ValueError):
        # Extremely defensive fallback. On modern torch this should not happen,
        # but keeps behavior compatible with environments where from_numpy has
        # narrower dtype support.
        if _torchgeo_array_to_tensor is not None:
            return _torchgeo_array_to_tensor(array), cast_info
        return torch.tensor(array), cast_info


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
            if self._thread.is_alive():
                logger.warning(
                    "COG reader pool thread did not join within 5 s; "
                    "resources may leak"
                )

    def __enter__(self) -> _AsyncCOGReaderPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
                In ``time_series=True`` mode the label is taken from the
                **first** (earliest) timestep only.
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
            self._warned_ts_temporal_skip = False
            self._warned_image_dtype_casts: set[tuple[str, str]] = set()
            self._warned_cog_read_failures = False

            scan_dataset = self.collection._filtered_data_dataset()
            if scan_dataset is None:
                raise ValueError(
                    "TorchGeo integration requires a dataset-backed collection scan. "
                    "Streaming-only collection backends are not supported here."
                )

            columns = [
                "id",
                "datetime",
                "assets",
                "proj:epsg",
                *[f"{b}_metadata" for b in self.bands],
            ]
            if label_field and label_field not in columns:
                columns.append(label_field)
            schema_names = scan_dataset.schema.names
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

            table = scan_dataset.to_table(columns=columns)
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
                    msg = (
                        f"{n_dropped} of {original_len} records dropped "
                        f"(CRS != EPSG:{self.epsg}). Collection spans "
                        f"{n_zones} CRS zones. Pass target_crs= to "
                        f"reproject instead of dropping."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
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
            _skip_no_datetime = 0
            _skip_no_metadata = 0
            _skip_no_dimensions = 0
            _res_mismatch_count = 0
            total_rows = len(df)

            for i, row in df.iterrows():
                dt = row.get("datetime")
                if pd.isna(dt):
                    dt = row.get("start_datetime")
                if pd.isna(dt):
                    dt = row.get("end_datetime")
                if pd.isna(dt):
                    _skip_no_datetime += 1
                    continue

                band_meta = _as_dict(row[f"{self.bands[0]}_metadata"])
                if not band_meta:
                    _skip_no_metadata += 1
                    continue
                try:
                    sx, tx, sy, ty = normalize_transform(band_meta.get("transform"))
                except (TypeError, ValueError):
                    _skip_no_metadata += 1
                    continue

                row_res = (abs(float(sx)), abs(float(sy)))
                if not (
                    np.isclose(row_res[0], self._res[0], rtol=1e-6, atol=1e-9)
                    and np.isclose(row_res[1], self._res[1], rtol=1e-6, atol=1e-9)
                ):
                    _res_mismatch_count += 1

                width = band_meta.get("image_width")
                height = band_meta.get("image_height")
                if width is None or height is None:
                    _skip_no_dimensions += 1
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

            skipped = total_rows - len(valid_rows)
            if skipped:
                parts = []
                if _skip_no_datetime:
                    parts.append(f"no datetime: {_skip_no_datetime}")
                if _skip_no_metadata:
                    parts.append(f"missing/invalid metadata: {_skip_no_metadata}")
                if _skip_no_dimensions:
                    parts.append(f"missing dimensions: {_skip_no_dimensions}")
                logger.warning(
                    "%d of %d records skipped (%s). "
                    "Ensure the collection was built with enrich_cog=True.",
                    skipped,
                    total_rows,
                    "; ".join(parts),
                )

            if _res_mismatch_count:
                warnings.warn(
                    f"{_res_mismatch_count} of {len(valid_rows)} valid records "
                    f"have a different native resolution than the dataset "
                    f"resolution {self._res} (derived from the first record). "
                    f"Chips for those records will be resampled.",
                    UserWarning,
                    stacklevel=2,
                )

            if not valid_rows:
                raise ValueError(
                    f"No valid records found for TorchGeo dataset creation. "
                    f"All {total_rows} records were skipped "
                    f"(no datetime: {_skip_no_datetime}, "
                    f"missing/invalid metadata: {_skip_no_metadata}, "
                    f"missing dimensions: {_skip_no_dimensions}). "
                    f"Build the collection with enrich_cog=True to populate "
                    f"per-band metadata."
                )

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
        ) -> list[tuple[Any, CogMetadata]]:
            """Fetch pixel arrays for all (url, metadata) pairs concurrently.

            Reuses the pool's persistent COGReader (and its obstore backend)
            so every request shares the same connection pool.
            Returns one ``(CogReadResult, CogMetadata)`` per request.

            Individual read failures are caught and logged; the failed
            request is excluded from the result list.  Callers must
            handle a shorter-than-expected result when reads fail.
            """
            reader = pool.reader

            async def _gather() -> list[tuple[Any, CogMetadata]]:
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
                raw = await asyncio.gather(*tasks, return_exceptions=True)
                out: list[tuple[Any, CogMetadata]] = []
                first_error: BaseException | None = None
                n_failed = 0
                for result, (url, meta, _band_index) in zip(raw, requests):
                    if isinstance(result, BaseException):
                        n_failed += 1
                        if first_error is None:
                            first_error = result
                        logger.warning(
                            "COG read failed for %s (band_index=%s): %s",
                            url,
                            _band_index,
                            result,
                        )
                        continue
                    data = getattr(result, "data", None)
                    if (
                        not isinstance(data, np.ndarray)
                        or data.ndim != 2
                        or data.size == 0
                    ):
                        n_failed += 1
                        empty_error = ValueError(
                            "COG read returned empty/non-2D data "
                            f"(shape={getattr(data, 'shape', None)})"
                        )
                        if first_error is None:
                            first_error = empty_error
                        logger.warning(
                            "COG read returned empty data for %s (band_index=%s): %s",
                            url,
                            _band_index,
                            empty_error,
                        )
                        continue
                    out.append((result, meta))
                if (
                    n_failed
                    and out
                    and first_error is not None
                    and not self._warned_cog_read_failures
                ):
                    self._warned_cog_read_failures = True
                    warnings.warn(
                        "RasteretGeoDataset skipped failed COG reads "
                        f"({n_failed}/{len(requests)} failure(s)) while building a chip; "
                        f"first failure: {first_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if not out and first_error is not None:
                    raise ValueError(
                        f"All {len(requests)} COG read request(s) failed for the requested chip."
                    ) from first_error
                return out

            return pool.run(_gather())

        def _filter_positive_overlap(
            self, df: gpd.GeoDataFrame, x: slice, y: slice
        ) -> gpd.GeoDataFrame:
            """Keep only records with positive-area overlap with query bounds."""
            if df.empty:
                return df
            xmin = float(x.start)
            ymin = float(y.start)
            xmax = float(x.stop)
            ymax = float(y.stop)
            bounds = df.geometry.bounds
            overlap_w = np.minimum(bounds["maxx"].to_numpy(), xmax) - np.maximum(
                bounds["minx"].to_numpy(), xmin
            )
            overlap_h = np.minimum(bounds["maxy"].to_numpy(), ymax) - np.maximum(
                bounds["miny"].to_numpy(), ymin
            )
            overlap_area = np.clip(overlap_w, 0.0, None) * np.clip(overlap_h, 0.0, None)
            mask = overlap_area > 0.0
            if not bool(np.any(mask)):
                return df.iloc[0:0].copy()
            filtered = df.iloc[np.flatnonzero(mask)].copy()
            filtered["__rasteret_overlap_area__"] = overlap_area[mask]
            return filtered

        def _payload_row(self, rid: int) -> Any:
            try:
                return self._payload.iloc[int(rid)]
            except Exception as exc:  # pragma: no cover
                raise IndexError(f"Invalid rid={rid}") from exc

        def _warn_image_dtype_cast_once(
            self,
            source_dtype: np.dtype[Any],
            target_dtype: np.dtype[Any],
        ) -> None:
            key = (str(source_dtype), str(target_dtype))
            if key in self._warned_image_dtype_casts:
                return
            self._warned_image_dtype_casts.add(key)
            warnings.warn(
                "RasteretGeoDataset casting image dtype "
                f"{source_dtype} -> {target_dtype} to match TorchGeo "
                "array_to_tensor semantics (uint16->int32, uint32->int64).",
                UserWarning,
                stacklevel=2,
            )

        def _merge_resample_to_query_grid(
            self,
            data: np.ndarray,
            data_transform: Affine,
            meta: CogMetadata,
            query_grid: MergeGrid,
            *,
            resampling: str,
        ) -> np.ndarray:
            return merge_semantic_resample_single_source(
                data,
                src_crop_transform=data_transform,
                src_full_transform=Affine(
                    *(
                        lambda sx, tx, sy, ty: (
                            float(sx),
                            0.0,
                            float(tx),
                            0.0,
                            float(sy),
                            float(ty),
                        )
                    )(*normalize_transform(meta.transform))
                ),
                src_full_width=int(meta.width),
                src_full_height=int(meta.height),
                src_crs=int(meta.crs or self.epsg),
                grid=query_grid,
                resampling=resampling,  # TorchGeo-like
                src_nodata=meta.nodata,
            )

        def _image_tensor_from_numpy(self, array: np.ndarray) -> torch.Tensor:
            tensor, cast_info = _array_to_image_tensor_torchgeo_compatible(array)
            if cast_info is not None:
                src_dtype, dst_dtype = cast_info
                self._warn_image_dtype_cast_once(src_dtype, dst_dtype)
            return tensor

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
                # time_series=True stacks ALL spatially overlapping records
                # regardless of the sampler's time slice.  Rasteret stores
                # precise per-scene dates from STAC, so applying the sampler's
                # narrow time window would miss most scenes.  Users control
                # date range upstream via build(date_range=...) or
                # collection.subset().
                if not self._warned_ts_temporal_skip:
                    self._warned_ts_temporal_skip = True
                    logger.info(
                        "time_series=True: sampler time slices are ignored; "
                        "all spatially overlapping records are stacked. "
                        "Use collection.subset(date_range=...) to limit the "
                        "temporal range before creating the dataset."
                    )
                df = self.index.cx[x.start : x.stop, y.start : y.stop]
                df = df.iloc[::t_step]
            else:
                interval = pd.Interval(t.start, t.stop, closed="both")
                df = self.index.iloc[self.index.index.overlaps(interval)]
                df = df.iloc[::t_step]
                df = df.cx[x.start : x.stop, y.start : y.stop]

            df = self._filter_positive_overlap(df, x, y)

            if df.empty:
                raise IndexError(
                    f"index: {index} not found in dataset with bounds: {self.bounds}"
                )

            pool = self._ensure_pool()
            # Fetch on the raster's native grid; sampling to the query grid is
            # handled below using TorchGeo-aligned merge semantics.
            patch_array = coerce_to_geoarrow((x.start, y.start, x.stop, y.stop))
            n_bands = len(self.bands)
            query_grid = MergeGrid(
                bounds=(float(x.start), float(y.start), float(x.stop), float(y.stop)),
                res=(abs(float(x.step)), abs(float(y.step))),
            )

            def _auto_resampling_for_dtype(dtype: np.dtype) -> str:
                # Match TorchGeo RasterDataset default: bilinear for float, nearest for int.
                if np.issubdtype(dtype, np.floating):
                    return "bilinear"
                return "nearest"

            # Always let the reader return its natural grid-aligned crop with a
            # self-consistent (data, transform) pair.
            fetch_out_shape = None
            label_rid: int | None = None

            if self.time_series:
                # Sort chronologically so T dimension is time-ordered.
                if "__rasteret_overlap_area__" in df.columns:
                    df = df.drop(columns=["__rasteret_overlap_area__"])
                df = df.sort_index()
                if not df.empty:
                    label_rid = int(df.iloc[0]["rid"])

                # Build all TxC requests, fire concurrently via asyncio.gather.
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

                expected = n_timesteps * n_bands
                got = len(fetch_results)
                if got == 0:
                    raise ValueError("All COG reads failed for time series sample")
                if got != expected:
                    raise ValueError(
                        f"{expected - got} of {expected} COG reads failed "
                        f"for time series sample (check warnings above)"
                    )

                # When multi-CRS, reproject every array to the target CRS grid.
                # This guarantees consistent (H, W) across all records/bands.
                if self._multi_crs:
                    arrays = [
                        reproject_array(
                            r.data,
                            r.transform,
                            src_crs,
                            self.epsg,
                            query_grid.transform,
                            query_grid.shape,
                        )
                        for (r, _meta), src_crs in zip(
                            fetch_results,
                            source_crs_per_request,
                        )
                    ]
                else:
                    arrays = []
                    for r, meta in fetch_results:
                        arrays.append(
                            self._merge_resample_to_query_grid(
                                r.data,
                                r.transform,
                                meta,
                                query_grid,
                                resampling=_auto_resampling_for_dtype(r.data.dtype),
                            )
                        )

                # arrays is flat: [t0_band0, t0_band1, ..., t1_band0, ...]
                # Reshape into [T, C, H, W] with minimal copies:
                #   np.stack at band level  -> [C, H, W] (1 contiguous alloc per timestep)
                #   np.stack at time level  -> [T, C, H, W] (1 contiguous alloc)
                #   torch.from_numpy        -> zero-copy view when dtype is unchanged
                timesteps = [
                    np.stack(arrays[t_idx * n_bands : (t_idx + 1) * n_bands], axis=0)
                    for t_idx in range(n_timesteps)
                ]
                image_np = np.stack(timesteps, axis=0)
                image = self._image_tensor_from_numpy(image_np)  # [T, C, H, W]

            else:
                # Minimal TorchGeo adapter behavior: select a single record for this
                # spatiotemporal slice. (TorchGeo's RasterDataset mosaics multiple
                # overlapping files, but Rasteret avoids reimplementing merge
                # semantics here.)
                if "__rasteret_overlap_area__" in df.columns:
                    df = df.sort_values(
                        "__rasteret_overlap_area__", ascending=False, kind="mergesort"
                    )
                if len(df) > 1:
                    logger.warning(
                        "TorchGeo slice overlaps %d records; selecting the record with "
                        "the largest spatial overlap (set time_series=True to stack "
                        "multiple timesteps).",
                        len(df),
                    )
                last_error: BaseException | None = None
                image: torch.Tensor | None = None
                for _, candidate in df.iterrows():
                    rid = int(candidate["rid"])
                    row = self._payload_row(rid)
                    band_requests = self._build_band_requests(row)
                    try:
                        fetch_results = self._fetch_arrays(
                            band_requests,
                            patch_array,
                            pool,
                            out_shape=fetch_out_shape,
                        )
                    except ValueError as exc:
                        last_error = exc
                        continue
                    if not fetch_results:
                        continue

                    try:
                        if self._multi_crs:
                            src_crs = int(row["proj:epsg"])
                            arrays = [
                                reproject_array(
                                    r.data,
                                    r.transform,
                                    src_crs,
                                    self.epsg,
                                    query_grid.transform,
                                    query_grid.shape,
                                )
                                for r, _meta in fetch_results
                            ]
                        else:
                            arrays = []
                            for r, meta in fetch_results:
                                arrays.append(
                                    self._merge_resample_to_query_grid(
                                        r.data,
                                        r.transform,
                                        meta,
                                        query_grid,
                                        resampling=_auto_resampling_for_dtype(
                                            r.data.dtype
                                        ),
                                    )
                                )

                        # np.stack -> [C, H, W] (1 alloc), then torch view/cast.
                        image_np = np.stack(arrays, axis=0)
                        image = self._image_tensor_from_numpy(image_np)  # [C, H, W]
                        label_rid = rid
                        break
                    except ValueError as exc:
                        last_error = exc
                        continue
                if image is None:
                    raise ValueError(
                        "No readable records were available for the requested chip."
                    ) from last_error

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
                # Use the first (earliest) record's label.  For time_series
                # this means the label is NOT per-timestep - it represents
                # the scene-level label of the earliest observation.
                label_value = None
                if label_rid is not None:
                    label_row = self._payload_row(label_rid)
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
