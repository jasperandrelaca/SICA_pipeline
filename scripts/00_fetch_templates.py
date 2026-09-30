#!/usr/bin/env python3
"""
00_fetch_templates.py -- copy the MNIPediatricAsym (cohort-1, 2 mm) template
files needed by MELODIC into $CATD_TEMPLATES_DIR.

The functional data were preprocessed into MNIPediatricAsym:cohort-1 space at
2 mm resolution, so MELODIC needs the matching brain mask and T1w background
image. Both are distributed through TemplateFlow.

Usage:
    export CATD_ROOT=/path/to/project
    python scripts/00_fetch_templates.py            # or: --out /some/dir
"""

import argparse
import shutil
from pathlib import Path

from templateflow import api as tflow

import catd_config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="Destination directory (default: $CATD_TEMPLATES_DIR)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else catd_config.templates_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        "brain mask": dict(desc="brain", suffix="mask"),
        "T1w": dict(suffix="T1w"),
    }
    for label, query in wanted.items():
        src = tflow.get("MNIPediatricAsym", cohort=1, resolution=2,
                        extension="nii.gz", **query)
        src = Path(str(src))
        dst = out_dir / src.name
        shutil.copy(src, dst)
        print(f"{label}: {dst}")


if __name__ == "__main__":
    main()
