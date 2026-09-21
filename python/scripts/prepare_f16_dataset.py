"""Prepare the F-16 Ground Vibration Test benchmark for the loader.

A full-scale F-16 with two dummy payloads at the wing tips, excited by one
shaker under the right wing and measured by three accelerometers. The
nonlinearity sits in the payload-to-wing mounting interfaces (clearance and
friction), so this is a lightly damped, high-order, *multi-output* system —
the hardest of the three benchmarks this package prepares.

    J.P. Noel and M. Schoukens. F-16 aircraft benchmark based on ground
    vibration test data. Workshop on Nonlinear System Identification
    Benchmarks, pp. 19-23, Brussels, 2017.

**There is no official train/test split** — the benchmark ships records and
leaves the protocol to the user (``nonlinear_benchmarks`` prints a warning
saying exactly that). What this script does instead is spelled out below, and
it is a choice, not a standard.

Records. Three excitation families, all at 400 Hz, one input (``Force``) and
three outputs (``Acceleration``), with the level indexing the forcing
amplitude:

    FullMSine_Level{1,3,5,7}            73728 samples   estimation levels
    FullMSine_Level{2,4,6}_Validation   73728 samples   validation levels
    SineSw_Level{1..7}[_Validation]    ~108k samples    swept sine
    SpecialOddMSine_Level{1,2,3}[_Validation]  49152    odd-multisine

Amplitude grows monotonically with the level (|F|max runs 43 -> 337 N over the
FullMSine levels), so the level is the extrapolation axis: fit at one level and
test at a higher one and the model is asked about states it never saw. That is
the same structure the Silverbox script exploits with its ``arrow`` record, and
it is why the default test record is one level ABOVE the training record.

Default protocol (override with ``--train-record`` / ``--test-record``):

    train/ + validation/   FullMSine_Level3          (contiguous tail split)
    test/                  FullMSine_Level4_Validation   (mild extrapolation)
    test_<name>/           every record named by --extra-test-records

Outputs. All three accelerometers are written to every CSV as ``y1,y2,y3``;
which of them a run fits is the config's ``data.output_col``. Switching between
one and three outputs is therefore a config edit, never a re-preparation.

    IMPORTANT -- three outputs do not currently train. ``identity`` parameter
    initialization divides C by the normalizer's ``output_std``, which is a
    per-channel vector when ne > 1, and the multiply is numpy-array-times-
    tensor: it raises TypeError before the first epoch. ne = 1 is unaffected.
    The shipped config therefore fits ``y1`` alone. See the module docstring of
    scripts/prepare_f16_dataset.py in the repo history, the README note, or run
    with --outputs to write a single-output CSV instead.

Usage::

    python scripts/prepare_f16_dataset.py
    python scripts/prepare_f16_dataset.py --train-record F16Data_FullMSine_Level5
    python scripts/prepare_f16_dataset.py --extra-test-records F16Data_SineSw_Level3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sysid.data.benchmark_prep import (  # noqa: E402
    clean_folders,
    record_stats,
    split_train_val,
    write_metadata,
    write_split,
)

INPUT_COLS = ["u"]
# All three accelerometers go into the CSV; data.output_col picks the subset a
# run actually fits.
ALL_OUTPUT_COLS = ["y1", "y2", "y3"]

DEFAULT_VAL_FRACTION = 0.2

# Mid-amplitude multisine to fit, the next level up to test on. Both are
# FullMSine so the excitation type is held fixed and only the amplitude moves.
DEFAULT_TRAIN_RECORD = "F16Data_FullMSine_Level3"
DEFAULT_TEST_RECORD = "F16Data_FullMSine_Level4_Validation"
# Two rungs further up the same ladder, for a harder extrapolation check.
DEFAULT_EXTRA_TEST_RECORDS = ("F16Data_FullMSine_Level6_Validation",)

SPLIT_FOLDERS = ["train", "validation", "test"]


def load_f16_records(
    names: Optional[Sequence[str]] = None, force_download: bool = False
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], float]:
    """Load the requested F-16 ``.mat`` records as ``{name: (u, y)}`` plus ``Ts``.

    The package's own ``F16()`` wrapper returns a single accelerometer chosen by
    ``output_index`` and drops the SpecialOddMSine files, so this reads the
    ``.mat`` files it downloads directly and keeps all three outputs.
    ``names=None`` loads every record.
    """
    try:
        import nonlinear_benchmarks.not_splitted_benchmarks as nsb
        from nonlinear_benchmarks.benchmarks import loadmat
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "The 'nonlinear_benchmarks' package is required to fetch the F-16 data.\n"
            "    pip install nonlinear-benchmarks"
        ) from e

    paths = {
        Path(p).stem: p
        for p in nsb.F16(data_file_locations=True, force_download=force_download)
    }
    wanted = list(paths) if names is None else list(names)
    missing = [n for n in wanted if n not in paths]
    if missing:
        raise SystemExit(
            f"unknown F-16 record(s) {missing}.\nAvailable:\n  "
            + "\n  ".join(sorted(paths))
        )

    records: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sampling_times = set()
    for name in wanted:
        mat = loadmat(paths[name])
        force = np.asarray(mat["Force"][0], dtype=np.float64).reshape(-1, 1)
        accel = np.asarray(mat["Acceleration"], dtype=np.float64)
        if accel.shape[0] != 3:
            raise RuntimeError(
                f"{name}: expected 3 accelerometer rows, got shape {accel.shape}"
            )
        y = accel.T  # (N, 3)
        if len(force) != len(y):
            raise RuntimeError(
                f"{name}: Force has {len(force)} samples, Acceleration has {len(y)}"
            )
        records[name] = (force, y)
        sampling_times.add(round(1.0 / float(mat["Fs"][0, 0]), 12))

    if len(sampling_times) != 1:
        raise RuntimeError(f"records disagree on the sampling time: {sampling_times}")
    return records, sampling_times.pop()


def select_outputs(y: np.ndarray, outputs: Sequence[int]) -> np.ndarray:
    """Keep the 1-based accelerometer indices in ``outputs``."""
    idx = [i - 1 for i in outputs]
    if any(i < 0 or i > 2 for i in idx):
        raise ValueError(f"accelerometer indices must be in 1..3, got {list(outputs)}")
    return np.asarray(y)[:, idx]


def prepare_dataset(
    records: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    sampling_time: float,
    train_record: str = DEFAULT_TRAIN_RECORD,
    test_record: str = DEFAULT_TEST_RECORD,
    extra_test_records: Sequence[str] = (),
    outputs: Sequence[int] = (1, 2, 3),
    val_fraction: float = DEFAULT_VAL_FRACTION,
    subsequence_length: Optional[int] = None,
    clean: bool = True,
) -> Dict:
    """Write the folder layout and return the metadata describing it.

    Split from ``main`` so the tests can drive it with synthetic records instead
    of a 148 MB download.
    """
    for name in (train_record, test_record, *extra_test_records):
        if name not in records:
            raise ValueError(f"record {name!r} was not loaded; have {sorted(records)}")

    output_cols = [ALL_OUTPUT_COLS[i - 1] for i in outputs]
    out_dir = Path(out_dir).expanduser()
    sibling = {name: f"test_{name}" for name in (test_record, *extra_test_records)}
    if clean:
        clean_folders(out_dir, [*SPLIT_FOLDERS, *sibling.values()])
    out_dir.mkdir(parents=True, exist_ok=True)

    u_tv, y_tv = records[train_record]
    y_tv = select_outputs(y_tv, outputs)
    u_tr, y_tr, u_va, y_va = split_train_val(u_tv, y_tv, val_fraction)

    files = {
        "train": len(
            write_split(out_dir / "train", u_tr, y_tr, INPUT_COLS, output_cols,
                        subsequence_length, stem="f16_train")
        ),
        "validation": len(
            write_split(out_dir / "validation", u_va, y_va, INPUT_COLS, output_cols,
                        subsequence_length, stem="f16_validation")
        ),
    }

    # Test records stay full-length regardless of --subsequence-length.
    stats = {"train": record_stats(u_tr, y_tr), "validation": record_stats(u_va, y_va)}
    for name, folder in sibling.items():
        u_te, y_te = records[name]
        y_te = select_outputs(y_te, outputs)
        files[folder] = len(
            write_split(out_dir / folder, u_te, y_te, INPUT_COLS, output_cols,
                        None, stem=name)
        )
        stats[folder] = record_stats(u_te, y_te)

    u_sel, y_sel = records[test_record]
    files["test"] = len(
        write_split(out_dir / "test", u_sel, select_outputs(y_sel, outputs),
                    INPUT_COLS, output_cols, None, stem=test_record)
    )

    metadata = {
        "dataset": "F-16 GVT",
        "source": "nonlinear_benchmarks F16 .mat files, read directly (all 3 accelerometers)",
        "reference": (
            "J.P. Noel and M. Schoukens, 'F-16 aircraft benchmark based on ground "
            "vibration test data', Workshop on Nonlinear System Identification "
            "Benchmarks, pp. 19-23, Brussels, 2017."
        ),
        "official_split": False,
        "split_note": (
            "The benchmark defines no train/test split. train/validation come from "
            f"{train_record} (contiguous tail split); test/ is {test_record}, which "
            "sits one amplitude level higher, so the test set is a mild "
            "extrapolation rather than a resample of the training distribution."
        ),
        "sampling_time": float(sampling_time),
        "input_col": list(INPUT_COLS),
        "output_col": list(output_cols),
        "accelerometers_written": [int(i) for i in outputs],
        "train_record": train_record,
        "test_record": test_record,
        "extra_test_records": list(extra_test_records),
        "val_fraction": float(val_fraction),
        "subsequence_length": subsequence_length,
        "n_files": files,
        "multi_output_note": (
            "All requested accelerometers are in the CSVs; data.output_col selects "
            "which a run fits. ne > 1 does NOT train on the current code: identity "
            "initialization multiplies a numpy per-channel output_std by a torch C "
            "and raises TypeError. Fit one output, or pass an explicit "
            "model.custom_params.identity_init.C.value."
        ),
        "records": stats,
    }
    write_metadata(out_dir, metadata)
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the F-16 GVT benchmark for sysid (no official split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=str, default="~/genSecSysId-Data/data/F16/id",
        help="Destination for train/, validation/, test/ and the test_* siblings.",
    )
    parser.add_argument(
        "--train-record", type=str, default=DEFAULT_TRAIN_RECORD,
        help="Record cut into train/ and validation/.",
    )
    parser.add_argument(
        "--test-record", type=str, default=DEFAULT_TEST_RECORD,
        help="Record copied into test/. The default is one amplitude level above "
             "--train-record.",
    )
    parser.add_argument(
        "--extra-test-records", type=str, nargs="*", default=list(DEFAULT_EXTRA_TEST_RECORDS),
        help="Further records written to test_<name>/ for extrapolation studies.",
    )
    parser.add_argument(
        "--outputs", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
        help="Which accelerometers to write. All three by default; the config's "
             "data.output_col then picks the subset to fit.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help="Tail of the training record held out for validation.",
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
    parser.add_argument(
        "--list-records", action="store_true",
        help="Print the available record names and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_records:
        records, ts = load_f16_records(None, args.force_download)
        print(f"F-16 records (sampling_time = {ts:g} s):")
        for name in sorted(records):
            u, y = records[name]
            print(f"  {name:45s} N={len(u):6d}  |F|max={np.abs(u).max():8.3f}")
        return 0

    wanted: List[str] = [args.train_record, args.test_record, *args.extra_test_records]
    records, sampling_time = load_f16_records(
        sorted(set(wanted)), force_download=args.force_download
    )
    out_dir = Path(args.out_dir).expanduser()
    metadata = prepare_dataset(
        records=records,
        out_dir=out_dir,
        sampling_time=sampling_time,
        train_record=args.train_record,
        test_record=args.test_record,
        extra_test_records=args.extra_test_records,
        outputs=args.outputs,
        val_fraction=args.val_fraction,
        subsequence_length=args.subsequence_length,
        clean=not args.no_clean,
    )

    print(f"F-16 GVT prepared in {out_dir}")
    print(f"  sampling_time : {metadata['sampling_time']:.8g} s "
          f"({1.0 / metadata['sampling_time']:.1f} Hz)")
    print(f"  columns       : {', '.join(INPUT_COLS)} (input), "
          f"{', '.join(metadata['output_col'])} (output)")
    print(f"  train record  : {metadata['train_record']}")
    print(f"  test record   : {metadata['test_record']}  (higher amplitude level)")
    for split, stats in metadata["records"].items():
        ymax = ", ".join(f"{v:.3f}" for v in stats["y_abs_max"])
        print(f"  {split:35s} N={stats['n_samples']:6d}  "
              f"|u|max={stats['u_abs_max'][0]:8.3f}  |y|max=[{ymax}]")
    print("  NOTE: this benchmark has NO official split -- see metadata.json.")
    if len(metadata["output_col"]) > 1:
        print("  NOTE: fitting >1 output does not train on the current code; the "
              "shipped config fits y1. See metadata.json 'multi_output_note'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
