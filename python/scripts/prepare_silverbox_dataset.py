"""Prepare the Silverbox benchmark for the loader, using its OFFICIAL split.

The Silverbox is an electronic Duffing oscillator: a 2nd-order LTI system with a
3rd-degree polynomial static nonlinearity in feedback. That is exactly the Lur'e
structure this package's ``crnn`` assumes, which is why ``nx=2`` in the shipped
configs is a modelling choice and not a guess.

    T. Wigren and J. Schoukens. Three free data sets for development and
    benchmarking in nonlinear system identification. ECC 2013, pp. 2933-2938.

Why go through ``nonlinear_benchmarks`` rather than the .mat files already in
``data/SilverboxFiles``. The existing ``prepared_*`` folders were cut with an
ad-hoc split, so their numbers are not comparable to anything published. The
package applies the split every Silverbox paper reports against::

    all      = SNLS80mV.mat  (V1 = input, V2 = output), 1/610.35 s sampling
    arrow    = all[100:40575]      filtered Gaussian noise, ramping amplitude
    multisine= all[40650:127400]   random-odd multisine, constant amplitude
      train_val  = multisine[:75%]   -> 65062 samples   (what you fit on)
      test       = multisine[75%:]   -> 21688 samples   (in-distribution test)
    arrow_no_extrapolation = arrow[:32000]

The arrow record is the interesting one here: its amplitude ramps *past* the
training range (u reaches 0.149 vs 0.101 on train, y reaches 0.300 vs 0.216), so
it leaves the region any regional certificate was fit over. ``arrow`` is the
extrapolation test, ``arrow_no_extrapolation`` its truncation that stays inside.
Both are written out alongside ``test/`` so a regional-vs-global comparison has
something to actually disagree about.

Layout. ``sysid.data.direct_loader.load_split_data`` hard-codes the folder names
``train/``, ``validation/``, ``test/`` and ``np.stack``s every CSV inside one
folder, so all files in a folder must share a row count. By default each split is
written as ONE full-length CSV and the windowing is left to the config
(``data.train_sequence_length`` / ``data.sequence_stride``), which keeps the
window length tunable without re-preparing. ``--subsequence-length`` instead cuts
fixed-length files, mirroring the older ``data/SilverboxFiles/prepared_131``.

The extra ``test_arrow/`` and ``test_arrow_no_extrapolation/`` folders are inert
as far as ``load_split_data`` is concerned — it never looks at them. Use
``--test-set`` to choose which record lands in ``test/``.

Usage::

    python scripts/prepare_silverbox_dataset.py
    python scripts/prepare_silverbox_dataset.py --test-set arrow
    python scripts/prepare_silverbox_dataset.py --subsequence-length 2048
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Column names written into every CSV. ``u``/``y`` rather than the raw ``V1``/
# ``V2`` so the configs read the same as the Duffing and OneD ones.
INPUT_COL = "u"
OUTPUT_COL = "y"

# Fraction of the official train_val block held out as validation, taken from the
# END of the record. A random split would be wrong: consecutive samples of one
# multisine record are not independent, so shuffled windows leak the validation
# dynamics into training. The tail is also what the benchmark's own train/test
# cut does one level up.
DEFAULT_VAL_FRACTION = 0.2

# The benchmark specifies 50 samples of state initialization before the test
# error is scored. Recorded in metadata.json; the configs' training.warmup_steps
# is the analogue on our side and is set higher, because the trainer rolls out
# from x0 = 0 rather than from a fitted initial state.
BENCHMARK_INIT_WINDOW = 50

# Which of the three official test records goes into ``test/``.
TEST_SET_CHOICES = ("multisine", "arrow", "arrow_no_extrapolation")

# Position of each record in the tuple nonlinear_benchmarks.Silverbox() returns.
# Keyed by position on purpose: v0.1.2 labels test[1] 'test SB multisine' even
# though it is the arrow record, so the ``.name`` attribute cannot be trusted.
_TEST_ORDER = ("multisine", "arrow", "arrow_no_extrapolation")


def _as_2d(a: np.ndarray) -> np.ndarray:
    """(N,) or (N, 1) -> (N, 1). Silverbox is SISO; anything else is a bug."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"expected a SISO signal, got shape {arr.shape}")
    return arr


def split_train_val(
    u: np.ndarray, y: np.ndarray, val_fraction: float = DEFAULT_VAL_FRACTION
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut the official train_val block into a train head and a validation tail.

    Contiguous, not shuffled — see DEFAULT_VAL_FRACTION.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    u, y = _as_2d(u), _as_2d(y)
    if len(u) != len(y):
        raise ValueError(f"u and y differ in length: {len(u)} vs {len(y)}")

    n_train = int(round(len(u) * (1.0 - val_fraction)))
    if n_train <= 0 or n_train >= len(u):
        raise ValueError(
            f"val_fraction={val_fraction} leaves an empty split for {len(u)} samples"
        )
    return u[:n_train], y[:n_train], u[n_train:], y[n_train:]


def write_split(
    folder: Path,
    u: np.ndarray,
    y: np.ndarray,
    subsequence_length: Optional[int] = None,
    stem: str = "record",
) -> List[Path]:
    """Write one split as CSV(s) with columns ``u,y``.

    ``subsequence_length=None`` writes a single full-length file. Otherwise the
    record is cut into non-overlapping files of exactly that many rows and the
    remainder is DROPPED — a short trailing file would break the ``np.stack`` in
    ``load_split_data``, which is the whole reason for the uniform-length rule.
    """
    u, y = _as_2d(u), _as_2d(y)
    if len(u) != len(y):
        raise ValueError(f"u and y differ in length: {len(u)} vs {len(y)}")

    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if subsequence_length is None:
        path = folder / f"{stem}.csv"
        pd.DataFrame({INPUT_COL: u[:, 0], OUTPUT_COL: y[:, 0]}).to_csv(path, index=False)
        return [path]

    if subsequence_length <= 0:
        raise ValueError(f"subsequence_length must be positive, got {subsequence_length}")
    n_chunks = len(u) // subsequence_length
    if n_chunks == 0:
        raise ValueError(
            f"subsequence_length={subsequence_length} exceeds the {len(u)}-sample "
            f"record for '{folder.name}'"
        )
    for i in range(n_chunks):
        sl = slice(i * subsequence_length, (i + 1) * subsequence_length)
        path = folder / f"{stem}_{i:04d}.csv"
        pd.DataFrame(
            {INPUT_COL: u[sl, 0], OUTPUT_COL: y[sl, 0]}
        ).to_csv(path, index=False)
        written.append(path)
    return written


def _record_stats(u: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Amplitude summary, so metadata.json shows which record extrapolates."""
    u, y = _as_2d(u), _as_2d(y)
    return {
        "n_samples": int(len(u)),
        "u_min": float(u.min()), "u_max": float(u.max()), "u_std": float(u.std()),
        "y_min": float(y.min()), "y_max": float(y.max()), "y_std": float(y.std()),
        "u_abs_max": float(np.abs(u).max()), "y_abs_max": float(np.abs(y).max()),
    }


def prepare_dataset(
    train_val: Tuple[np.ndarray, np.ndarray],
    test_records: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    sampling_time: float,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    subsequence_length: Optional[int] = None,
    test_set: str = "multisine",
    clean: bool = True,
) -> Dict:
    """Write the full folder layout and return the metadata that describes it.

    Split from ``main`` so the tests can drive it with synthetic records instead
    of a 6 MB download.

    Args:
        train_val: ``(u, y)`` of the official train_val block.
        test_records: the official test records keyed by the names in
            ``TEST_SET_CHOICES``.
        out_dir: destination; ``train/``, ``validation/``, ``test/`` and the
            ``test_*`` sibling folders are created inside it.
        sampling_time: seconds between samples, recorded in the metadata so the
            config's ``data.sampling_time`` can be checked against it.
        val_fraction: tail of train_val held out for validation.
        subsequence_length: rows per CSV, or None for one file per split.
        test_set: which record from ``test_records`` lands in ``test/``.
        clean: remove pre-existing split folders first, so a re-run with a
            different ``--subsequence-length`` cannot leave stale CSVs behind
            that then fail the uniform-length stack.
    """
    if test_set not in TEST_SET_CHOICES:
        raise ValueError(f"test_set must be one of {TEST_SET_CHOICES}, got {test_set!r}")
    missing = [k for k in TEST_SET_CHOICES if k not in test_records]
    if missing:
        raise ValueError(f"test_records is missing {missing}")

    out_dir = Path(out_dir).expanduser()
    # Sibling folder per official test record. The one selected by --test-set is
    # ALSO written to test/ (duplicated on disk, a few MB) so that load_split_data
    # finds it under the name it expects while the others stay addressable.
    sibling = {
        "multisine": "test_multisine",
        "arrow": "test_arrow",
        "arrow_no_extrapolation": "test_arrow_no_extrapolation",
    }
    if clean:
        for name in ["train", "validation", "test", *sibling.values()]:
            shutil.rmtree(out_dir / name, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    u_tv, y_tv = train_val
    u_tr, y_tr, u_va, y_va = split_train_val(u_tv, y_tv, val_fraction)

    files: Dict[str, int] = {}
    files["train"] = len(
        write_split(out_dir / "train", u_tr, y_tr, subsequence_length, stem="multisine_train")
    )
    files["validation"] = len(
        write_split(
            out_dir / "validation", u_va, y_va, subsequence_length, stem="multisine_validation"
        )
    )

    # Test records are written full-length regardless of --subsequence-length:
    # load_split_data always evaluates test with whole sequences, and the
    # benchmark RMSE is defined over the whole record.
    for name, (u_te, y_te) in test_records.items():
        files[sibling[name]] = len(
            write_split(out_dir / sibling[name], u_te, y_te, None, stem=sibling[name])
        )
    u_sel, y_sel = test_records[test_set]
    files["test"] = len(
        write_split(out_dir / "test", u_sel, y_sel, None, stem=sibling[test_set])
    )

    metadata = {
        "dataset": "Silverbox",
        "source": "nonlinear_benchmarks.Silverbox() (official train/test split)",
        "reference": (
            "T. Wigren and J. Schoukens, 'Three free data sets for development and "
            "benchmarking in nonlinear system identification', ECC 2013, pp. 2933-2938."
        ),
        "sampling_time": float(sampling_time),
        "input_col": [INPUT_COL],
        "output_col": [OUTPUT_COL],
        "val_fraction": float(val_fraction),
        "subsequence_length": subsequence_length,
        "test_set_in_test_folder": test_set,
        "benchmark_state_initialization_window_length": BENCHMARK_INIT_WINDOW,
        "n_files": files,
        "records": {
            "train": _record_stats(u_tr, y_tr),
            "validation": _record_stats(u_va, y_va),
            **{sibling[n]: _record_stats(*r) for n, r in test_records.items()},
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def load_official_silverbox(force_download: bool = False) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]], float
]:
    """Download (or reuse the cache) and unpack the official Silverbox split.

    Returns ``(train_val, test_records, sampling_time)`` as plain numpy, so the
    rest of this module never touches the package's data types.
    """
    try:
        import nonlinear_benchmarks
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "The 'nonlinear_benchmarks' package is required to fetch Silverbox.\n"
            "    pip install nonlinear-benchmarks"
        ) from e

    train_val, tests = nonlinear_benchmarks.Silverbox(
        atleast_2d=True, force_download=force_download
    )
    if len(tests) != len(_TEST_ORDER):
        raise RuntimeError(
            f"expected {len(_TEST_ORDER)} Silverbox test records "
            f"{_TEST_ORDER}, got {len(tests)} — the package layout changed."
        )
    records = {name: (t.u, t.y) for name, t in zip(_TEST_ORDER, tests)}
    return (train_val.u, train_val.y), records, float(train_val.sampling_time)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the Silverbox benchmark (official split) for sysid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="~/genSecSysId-Data/data/Silverbox/id",
        help="Destination for train/, validation/, test/ and the test_* siblings.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help="Tail of the official train_val block held out for validation.",
    )
    parser.add_argument(
        "--subsequence-length", type=int, default=None,
        help="Rows per train/validation CSV. Omit to write one full-length file "
             "per split and window from the config instead.",
    )
    parser.add_argument(
        "--test-set", type=str, default="multisine", choices=TEST_SET_CHOICES,
        help="Which official test record is copied into test/. 'arrow' is the "
             "extrapolation record (amplitudes beyond the training range).",
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

    train_val, test_records, sampling_time = load_official_silverbox(args.force_download)
    out_dir = Path(args.out_dir).expanduser()
    metadata = prepare_dataset(
        train_val=train_val,
        test_records=test_records,
        out_dir=out_dir,
        sampling_time=sampling_time,
        val_fraction=args.val_fraction,
        subsequence_length=args.subsequence_length,
        test_set=args.test_set,
        clean=not args.no_clean,
    )

    print(f"Silverbox prepared in {out_dir}")
    print(f"  sampling_time : {metadata['sampling_time']:.8g} s "
          f"({1.0 / metadata['sampling_time']:.2f} Hz)")
    print(f"  columns       : {INPUT_COL} (input), {OUTPUT_COL} (output)")
    print(f"  test/ holds   : {metadata['test_set_in_test_folder']}")
    for split, stats in metadata["records"].items():
        print(
            f"  {split:28s} N={stats['n_samples']:6d}  "
            f"|u|max={stats['u_abs_max']:.4f}  |y|max={stats['y_abs_max']:.4f}"
        )
    print(
        "  note: 'test_arrow' exceeds the training amplitude range "
        "(extrapolation); 'test_arrow_no_extrapolation' does not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
