#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rope_control.viz import plot_results, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.csv.parent
    summarize_results(args.csv, out)
    plot_results(args.csv, out)


if __name__ == "__main__":
    main()

