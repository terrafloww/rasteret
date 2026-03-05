# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Run the Major TOM benchmark from an existing Rasteret cache (HF streaming only)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HF-streaming benchmark from a prebuilt Rasteret collection cache"
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        required=True,
        help="Path to scene-level Rasteret collection cache",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path(__file__).with_name("03_hf_vs_rasteret_benchmark.py"),
        help="Path to benchmark script",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Sample counts to run",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B02", "B08"],
        help="Bands to benchmark",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed when sample strategy=random",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["head", "random"],
        default="head",
        help="Sample strategy for metadata key selection",
    )
    parser.add_argument(
        "--min-unique-epsg",
        type=int,
        default=1,
        help="Minimum CRS diversity",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=150,
        help="Rasteret max_concurrent in benchmark",
    )
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Path to HF token file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.collection_path.exists():
        raise SystemExit(f"collection path not found: {args.collection_path}")
    if not args.benchmark_script.exists():
        raise SystemExit(f"benchmark script not found: {args.benchmark_script}")

    env = os.environ.copy()
    if args.hf_token_file is not None and args.hf_token_file.exists():
        token = args.hf_token_file.read_text(encoding="utf-8").strip()
        if token and "HF_TOKEN" not in env:
            env["HF_TOKEN"] = token

    for sample_count in args.samples:
        cmd = [
            sys.executable,
            str(args.benchmark_script),
            "--samples",
            str(int(sample_count)),
            "--bands",
            *[str(b) for b in args.bands],
            "--sample-strategy",
            str(args.sample_strategy),
            "--random-seed",
            str(int(args.random_seed)),
            "--min-unique-epsg",
            str(int(args.min_unique_epsg)),
            "--max-concurrent",
            str(int(args.max_concurrent)),
            "--collection-path",
            str(args.collection_path),
        ]
        print(f"\n=== running samples={sample_count} (HF streaming) ===")
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
