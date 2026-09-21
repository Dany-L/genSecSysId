"""Prepare the Cascaded Tanks benchmark for the loader, using its OFFICIAL split.

Two water tanks in series, the upper draining into the lower, driven by a pump.
The identification interest is the *overflow*: when the upper tank runs over,
part of the water bypasses the lower tank, which makes the dynamics switch
rather than merely saturate. That hard nonlinearity is what the benchmark is
for, and it is also why a Lur'e model with one static nonlinearity in feedback
is a genuine modelling choice here rather than a structural match.

    M. Schoukens and J.P. Noel. Three benchmarks addressing open challenges in
    nonlinear system identification. IFAC World Congress, 2017.

Split. ``nonlinear_benchmarks.Cascaded_Tanks()`` returns the benchmark's own
two records, and there is nothing to choose:

    uEst / yEst  ->  1024 samples, the estimation record
    uVal / yVal  ->  1024 samples, the test record

The estimation record is cut into ``train/`` and a contiguous ``validation/``
tail (``--val-fraction``); the test record goes to ``test/`` whole. The
benchmark scores its test error after a 5-sample state-initialization window.

Two properties of this data drive the shipped config, and both are unusual for
this package:

* **1024 samples per record.** That is two orders of magnitude less than
  Silverbox. Long windows plus a large stride would leave a handful of
  sequences, so the config windows short and strides hard.
* **The signals are not zero-mean.** Levels live in ``u ~ [0.4, 6.5]``,
  ``y ~ [2.9, 10.0]`` — strictly positive, around an operating point, and the
  10.0 ceiling is the overflow. A Lur'e model rolled out from ``x0 = 0`` has its
  equilibrium at the origin, so the config uses
  ``normalization_method: standard`` (mean removed) rather than the
  ``scale_only`` the Duffing and Silverbox configs use. With ``scale_only`` the
  model would have to spend its transient climbing to the operating point on
  every window, and the certified set would be centred in the wrong place.

Usage::

    python scripts/prepare_cascaded_tanks_dataset.py
    python scripts/prepare_cascaded_tanks_dataset.py --subsequence-length 256
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sysid.data.benchmark_prep import (  # noqa: E402
    clean_folders,
    record_stats,
    split_train_val,
    write_metadata,
    write_split,
)

# Column names written into every CSV: ``u``/``y`` so the configs read the same
# as the Duffing, OneD and Silverbox ones.
INPUT_COLS = ["u"]
OUTPUT_COLS = ["y"]

# Tail of the estimation record held out for validation. 20% of 1024 leaves 205
# validation samples — small, but a contiguous tail is still the right shape of
# split (see split_train_val).
DEFAULT_VAL_FRACTION = 0.2

# The benchmark scores its test error after this many samples of state
# initialization. Recorded in metadata.json; training.warmup_steps is the
# analogue on our side and is set higher, because the trainer rolls out from
# x0 = 0 rather than from a fitted initial state.
BENCHMARK_INIT_WINDOW = 5

SPLIT_FOLDERS = ["train", "validation", "test"]


def prepare_dataset(
    estimation: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    out_dir: Path,
    sampling_time: float,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    subsequence_length: Optional[int] = None,
    clean: bool = True,
) -> Dict:
    """Write the folder layout and return the metadata describing it.

    Split from ``main`` so the tests can drive it with synthetic records instead
    of fetching the benchmark archive.
    """
    out_dir = Path(out_dir).expanduser()
    if clean:
        clean_folders(out_dir, SPLIT_FOLDERS)
    out_dir.mkdir(parents=True, exist_ok=True)

    u_est, y_est = estimation
    u_tr, y_tr, u_va, y_va = split_train_val(u_est, y_est, val_fraction)

    files = {
        "train": len(
            write_split(out_dir / "train", u_tr, y_tr, INPUT_COLS, OUTPUT_COLS,
                        subsequence_length, stem="tanks_train")
        ),
        "validation": len(
            write_split(out_dir / "validation", u_va, y_va, INPUT_COLS, OUTPUT_COLS,
                        subsequence_length, stem="tanks_validation")
        ),
        # The test record is written full-length regardless of
        # --subsequence-length: the benchmark RMSE is defined over the whole
        # record, and load_split_data evaluates test with whole sequences.
        "test": len(
            write_split(out_dir / "test", *test, INPUT_COLS, OUTPUT_COLS,
                        None, stem="tanks_test")
        ),
    }

    metadata = {
        "dataset": "Cascaded Tanks",
        "source": "nonlinear_benchmarks.Cascaded_Tanks() (official estimation/test split)",
        "reference": (
            "M. Schoukens and J.P. Noel, 'Three benchmarks addressing open challenges "
            "in nonlinear system identification', IFAC World Congress, 2017."
        ),
        "sampling_time": float(sampling_time),
        "input_col": list(INPUT_COLS),
        "output_col": list(OUTPUT_COLS),
        "val_fraction": float(val_fraction),
        "subsequence_length": subsequence_length,
        "benchmark_state_initialization_window_length": BENCHMARK_INIT_WINDOW,
        "n_files": files,
        "notes": (
            "Signals are not zero-mean (levels around an operating point, 10.0 is "
            "the overflow ceiling); the shipped config uses "
            "normalization_method 'standard' so the model's equilibrium at x=0 "
            "matches the data."
        ),
        "records": {
            "train": record_stats(u_tr, y_tr),
            "validation": record_stats(u_va, y_va),
            "test": record_stats(*test),
        },
    }
    write_metadata(out_dir, metadata)
    return metadata


def load_official_cascaded_tanks(force_download: bool = False):
    """Download (or reuse the cache) and unpack the official split as numpy."""
    try:
        import nonlinear_benchmarks
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "The 'nonlinear_benchmarks' package is required to fetch Cascaded Tanks.\n"
            "    pip install nonlinear-benchmarks"
        ) from e

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks(
        atleast_2d=True, force_download=force_download
    )
    return (
        (np.asarray(train_val.u), np.asarray(train_val.y)),
        (np.asarray(test.u), np.asarray(test.y)),
        float(train_val.sampling_time),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the Cascaded Tanks benchmark (official split) for sysid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=str, default="~/genSecSysId-Data/data/CascadedTanks/id",
        help="Destination for train/, validation/ and test/.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help="Tail of the estimation record held out for validation.",
    )
    parser.add_argument(
        "--subsequence-length", type=int, default=None,
        help="Rows per train/validation CSV. Omit to write one full-length file "
             "per split and window from the config instead.",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download instead of reusing the nonlinear_benchmarks cache.",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Keep pre-existing CSVs in the split folders (risks mixing lengths).",
    )
    args = parser.parse_args(argv)

    estimation, test, sampling_time = load_official_cascaded_tanks(args.force_download)
    out_dir = Path(args.out_dir).expanduser()
    metadata = prepare_dataset(
        estimation=estimation,
        test=test,
        out_dir=out_dir,
        sampling_time=sampling_time,
        val_fraction=args.val_fraction,
        subsequence_length=args.subsequence_length,
        clean=not args.no_clean,
    )

    print(f"Cascaded Tanks prepared in {out_dir}")
    print(f"  sampling_time : {metadata['sampling_time']:.8g} s "
          f"({1.0 / metadata['sampling_time']:.4f} Hz)")
    print(f"  columns       : {', '.join(INPUT_COLS)} (input), "
          f"{', '.join(OUTPUT_COLS)} (output)")
    for split, stats in metadata["records"].items():
        print(
            f"  {split:12s} N={stats['n_samples']:5d}  "
            f"u in [{stats['u_mean'][0] - 3 * stats['u_std'][0]:.2f}, "
            f"{stats['u_abs_max'][0]:.2f}]  |y|max={stats['y_abs_max'][0]:.3f}  "
            f"y_mean={stats['y_mean'][0]:.3f}"
        )
    print("  note: signals are NOT zero-mean -- the shipped config uses "
          "normalization_method 'standard'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
