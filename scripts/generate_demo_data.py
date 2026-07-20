from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmd.config import DATA_DIR
from wmd.synthetic import generate_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo MRI data")
    parser.add_argument("--n-per-class", type=int, default=40)
    parser.add_argument("--out", default=str(DATA_DIR / "synthetic"))
    parser.add_argument("--with-clinical", action="store_true")
    parser.add_argument(
        "--multiclass", action="store_true",
        help="Generate per-etiology classes (cause) instead of binary.",
    )
    args = parser.parse_args()

    manifest = generate_dataset(
        args.out,
        n_per_class=args.n_per_class,
        with_clinical=args.with_clinical,
        multiclass=args.multiclass,
    )
    print(f"Generated synthetic dataset. Manifest: {manifest}")

if __name__ == "__main__":
    main()
