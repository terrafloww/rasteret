# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Display helpers for Collection.describe().

Returns a DescribeResult object that renders as:
- A plain-text table in terminals (``__repr__``)
- An HTML table in Jupyter/marimo notebooks (``_repr_html_``)

No external dependencies (no rich, no IPython imports).
"""

from __future__ import annotations

from typing import Any

# Terrafloww brand blue
_BRAND_BLUE = "#009DD1"
_BRAND_BLUE_LIGHT = "#38bdf8"
_BRAND_DARK = "#0c4a6e"


def _format_bands(bands: list[str], max_show: int = 5) -> str:
    """Format a band list, truncating if needed."""
    if not bands:
        return "-"
    if len(bands) <= max_show:
        return ", ".join(bands)
    shown = ", ".join(bands[:max_show])
    return f"{shown} (+{len(bands) - max_show} more)"


def _format_bounds(bounds: tuple[float, ...] | None) -> str:
    if bounds is None:
        return "-"
    return f"({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f})"


def _format_crs(epsg: list[int]) -> str:
    if not epsg:
        return "-"
    if len(epsg) == 1:
        return f"EPSG:{epsg[0]}"
    if len(epsg) <= 3:
        return ", ".join(f"EPSG:{e}" for e in epsg)
    shown = ", ".join(f"EPSG:{e}" for e in epsg[:3])
    return f"{shown} (+{len(epsg) - 3} more)"


class DescribeResult:
    """Render-friendly result from ``Collection.describe()``.

    Adapts to the display environment automatically:

    - **Terminal / REPL**: plain-text table via ``__repr__``
    - **Jupyter / marimo**: styled HTML table via ``_repr_html_``

    The underlying data is accessible as a plain dict via ``.data``.
    """

    __slots__ = ("_rows", "_title", "_data")

    def __init__(
        self,
        rows: list[tuple[str, str]],
        title: str,
        data: dict[str, Any],
    ) -> None:
        self._rows = rows
        self._title = title
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        """Raw data as a plain dict."""
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # ── Terminal rendering ──────────────────────────────────────────

    def __repr__(self) -> str:
        if not self._rows:
            return f"{self._title}: (empty)"

        key_width = max(len(k) for k, _ in self._rows)
        val_width = max(len(v) for _, v in self._rows)
        key_width = max(key_width, 8)  # minimum width

        lines: list[str] = [self._title, ""]
        lines.append(f"  {'Property':<{key_width}}  Value")
        lines.append(f"  {'─' * key_width}  {'─' * val_width}")
        for key, val in self._rows:
            lines.append(f"  {key:<{key_width}}  {val}")
        lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # ── Notebook rendering (Jupyter, marimo, Colab, etc.) ───────────

    def _repr_html_(self) -> str:
        rows_html = ""
        for key, val in self._rows:
            rows_html += (
                f"<tr>"
                f"<td style='padding:4px 12px 4px 0;color:{_BRAND_DARK};"
                f"font-weight:600;white-space:nowrap'>{_esc(key)}</td>"
                f"<td style='padding:4px 0;font-family:monospace'>"
                f"{_esc(val)}</td>"
                f"</tr>"
            )

        return (
            f"<div style='font-family:system-ui,-apple-system,sans-serif;"
            f"font-size:13px;max-width:560px'>"
            f"<div style='font-weight:700;font-size:14px;"
            f"color:{_BRAND_BLUE};margin-bottom:6px'>"
            f"{_esc(self._title)}</div>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"<thead><tr>"
            f"<th style='text-align:left;padding:4px 12px 4px 0;"
            f"border-bottom:2px solid {_BRAND_BLUE};font-size:12px;"
            f"color:#666'>Property</th>"
            f"<th style='text-align:left;padding:4px 0;"
            f"border-bottom:2px solid {_BRAND_BLUE};font-size:12px;"
            f"color:#666'>Value</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )

    def _repr_mimebundle_(
        self, *, include: Any = None, exclude: Any = None, **kwargs: Any
    ) -> dict[str, str]:
        """Support IPython/marimo display protocol."""
        return {
            "text/plain": self.__repr__(),
            "text/html": self._repr_html_(),
        }


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_describe_result(
    *,
    name: str,
    records: int,
    bands: list[str],
    bounds: tuple[float, float, float, float] | None,
    crs: list[int],
    dates: tuple[str, str] | None,
    source: str,
) -> DescribeResult:
    """Build a DescribeResult from collection properties."""
    rows: list[tuple[str, str]] = [
        ("Records", str(records)),
        ("Bands", _format_bands(bands)),
        ("CRS", _format_crs(crs)),
        ("Bounds", _format_bounds(bounds)),
    ]
    if dates:
        rows.append(("Dates", f"{dates[0]} .. {dates[1]}"))
    if source:
        rows.append(("Source", source))

    data: dict[str, Any] = {
        "name": name,
        "records": records,
        "bands": bands,
        "bounds": bounds,
        "crs": crs,
    }
    if dates:
        data["dates"] = dates
    if source:
        data["source"] = source

    return DescribeResult(rows=rows, title=f"Collection: {name}", data=data)


def build_catalog_comparison(
    *,
    name: str,
    records: int,
    bands: list[str],
    bounds: tuple[float, float, float, float] | None,
    crs: list[int],
    dates: tuple[str, str] | None,
    source: str,
    catalog_name: str,
    catalog_bands: list[str],
    catalog_temporal: tuple[str, str] | None,
    catalog_coverage: str,
    catalog_auth: bool,
    catalog_license: str,
) -> DescribeResult:
    """Build a comparison DescribeResult (collection vs catalog)."""
    # Band comparison
    my_set = set(bands)
    cat_set = set(catalog_bands)
    missing = sorted(cat_set - my_set)
    if missing:
        bands_val = f"{_format_bands(bands)} (missing: {', '.join(missing)})"
    else:
        bands_val = f"{_format_bands(bands)} ({len(bands)}/{len(catalog_bands)})"

    # Date comparison
    dates_val = "-"
    if dates and catalog_temporal:
        dates_val = (
            f"{dates[0]} .. {dates[1]}"
            f"  (source: {catalog_temporal[0]} .. {catalog_temporal[1]})"
        )
    elif dates:
        dates_val = f"{dates[0]} .. {dates[1]}"

    rows: list[tuple[str, str]] = [
        ("Records", str(records)),
        ("Bands", bands_val),
        ("CRS", _format_crs(crs)),
        ("Bounds", _format_bounds(bounds)),
        ("Dates", dates_val),
        ("Source", f"{source} ({catalog_name})"),
        ("Coverage", catalog_coverage or "-"),
        ("Auth", "required" if catalog_auth else "none"),
    ]
    if catalog_license:
        rows.append(("License", catalog_license))

    data: dict[str, Any] = {
        "name": name,
        "records": records,
        "bands": bands,
        "bounds": bounds,
        "crs": crs,
        "source": source,
        "catalog": {
            "name": catalog_name,
            "all_bands": catalog_bands,
            "temporal_range": catalog_temporal,
            "spatial_coverage": catalog_coverage,
            "requires_auth": catalog_auth,
            "license": catalog_license,
        },
    }
    if dates:
        data["dates"] = dates

    return DescribeResult(
        rows=rows,
        title=f"Collection: {name}",
        data=data,
    )
