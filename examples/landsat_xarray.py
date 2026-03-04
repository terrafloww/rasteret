"""Landsat xarray workflow (requester-pays).

Same build -> get_xarray -> NDVI pattern as Sentinel-2,
but targeting Landsat on Earth Search.

Requires AWS credentials (``aws configure`` or environment variables)
because the Landsat bucket is requester-pays.
"""

from __future__ import annotations

import rasteret

BBOX = (77.55, 13.01, 77.58, 13.08)


def _has_aws_credentials() -> bool:
    try:
        import boto3
    except Exception:
        return False

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        return False
    frozen = creds.get_frozen_credentials()
    return bool(frozen.access_key and frozen.secret_key)


def main() -> None:
    if not _has_aws_credentials():
        print(
            "This example requires AWS credentials (Landsat on Earth Search is "
            "requester-pays).\n\n"
            "Set credentials via one of:\n"
            "- `aws configure`\n"
            "- env vars: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "(and optionally AWS_SESSION_TOKEN)\n"
            "- an AWS profile: AWS_PROFILE\n"
        )
        return

    collection = rasteret.build(
        "earthsearch/landsat-c2-l2",
        name="bangalorelandsat",
        bbox=BBOX,
        date_range=("2024-01-01", "2024-01-31"),
        force=True,
    )
    print(f"Collection: {collection.name}, rows={collection.dataset.count_rows()}")

    ds = collection.get_xarray(
        geometries=BBOX,
        bands=["B4", "B5"],
    )

    ndvi = (ds["B5"] - ds["B4"]) / (ds["B5"] + ds["B4"])
    print(f"NDVI shape: {ndvi.shape}")
    print(f"NDVI range: [{float(ndvi.min()):.3f}, {float(ndvi.max()):.3f}]")


if __name__ == "__main__":
    main()
