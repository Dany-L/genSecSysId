"""Shared layout helpers for benchmark preparation.

``scripts/prepare_benchmark.py`` writes the folder layout that
:func:`sysid.data.direct_loader.load_split_data` expects::

    <out_dir>/train/        one or more CSVs, all with the SAME row count
    <out_dir>/validation/   idem
    <out_dir>/test/         idem
    <out_dir>/metadata.json

``load_split_data`` hard-codes those three folder names and ``np.stack``s every
CSV inside one folder, so the uniform-row-count rule is not optional — a short
trailing file fails far from its cause. Any other folder (``test_<name>``) is
inert as far as the loader is concerned and is there to keep the other official
records addressable.

These helpers are MIMO: the F-16 carries three accelerometers in one CSV, and
which of them a run actually fits is left to the config's ``data.output_col``.
That way switching between one and three outputs is a config edit and never a
re-preparation.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def as_2d(array: np.ndarray, name: str, n_cols: Optional[int] = None) -> np.ndarray:
    """``(N,)`` or ``(N, k)`` -> ``(N, k)`` float64, with an optional width check."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected a 1-D or 2-D signal, got shape {arr.shape}")
    if n_cols is not None and arr.shape[1] != n_cols:
        raise ValueError(
            f"{name}: expected {n_cols} column(s), got {arr.shape[1]} (shape {arr.shape})"
        )
    return arr


def split_train_val(u: np.ndarray, y: np.ndarray, val_fraction: float):
    """Cut a record into a train head and a validation tail.

    Contiguous, never shuffled: consecutive samples of one excitation record are
    not independent, so shuffled windows leak the validation dynamics into
    training. Returns ``(u_train, y_train, u_val, y_val)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    u, y = as_2d(u, "u"), as_2d(y, "y")
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
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    subsequence_length: Optional[int] = None,
    stem: str = "record",
) -> List[Path]:
    """Write one split as CSV(s) with columns ``input_cols + output_cols``.

    ``subsequence_length=None`` writes a single full-length file and leaves the
    windowing to the config (``data.train_sequence_length`` /
    ``data.sequence_stride``), which keeps the window tunable without
    re-preparing. Otherwise the record is cut into non-overlapping files of
    exactly that many rows and the remainder is DROPPED — a short trailing file
    would break the ``np.stack`` in ``load_split_data``.
    """
    u = as_2d(u, "u", len(input_cols))
    y = as_2d(y, "y", len(output_cols))
    if len(u) != len(y):
        raise ValueError(f"u and y differ in length: {len(u)} vs {len(y)}")

    folder.mkdir(parents=True, exist_ok=True)

    def frame(sl: slice) -> pd.DataFrame:
        data = {name: u[sl, i] for i, name in enumerate(input_cols)}
        data.update({name: y[sl, i] for i, name in enumerate(output_cols)})
        return pd.DataFrame(data)

    if subsequence_length is None:
        path = folder / f"{stem}.csv"
        frame(slice(None)).to_csv(path, index=False)
        return [path]

    if subsequence_length <= 0:
        raise ValueError(f"subsequence_length must be positive, got {subsequence_length}")
    n_chunks = len(u) // subsequence_length
    if n_chunks == 0:
        raise ValueError(
            f"subsequence_length={subsequence_length} exceeds the {len(u)}-sample "
            f"record for '{folder.name}'"
        )
    written: List[Path] = []
    for i in range(n_chunks):
        path = folder / f"{stem}_{i:04d}.csv"
        frame(slice(i * subsequence_length, (i + 1) * subsequence_length)).to_csv(
            path, index=False
        )
        written.append(path)
    return written


def record_stats(u: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Per-channel amplitude summary, so metadata.json shows what extrapolates."""
    u, y = as_2d(u, "u"), as_2d(y, "y")
    return {
        "n_samples": int(len(u)),
        "u_abs_max": [float(np.abs(u[:, i]).max()) for i in range(u.shape[1])],
        "u_mean": [float(u[:, i].mean()) for i in range(u.shape[1])],
        "u_std": [float(u[:, i].std()) for i in range(u.shape[1])],
        "y_abs_max": [float(np.abs(y[:, i]).max()) for i in range(y.shape[1])],
        "y_mean": [float(y[:, i].mean()) for i in range(y.shape[1])],
        "y_std": [float(y[:, i].std()) for i in range(y.shape[1])],
    }


def clean_folders(out_dir: Path, names: Sequence[str]) -> None:
    """Remove split folders so a re-run cannot leave stale CSVs of another length."""
    for name in names:
        shutil.rmtree(out_dir / name, ignore_errors=True)


def write_metadata(out_dir: Path, metadata: Dict) -> Path:
    path = Path(out_dir) / "metadata.json"
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    return path


# ── multi-record layout (used by the generic benchmark adapter) ───────────────
SPLIT_FOLDERS = ["train", "validation", "test"]


def split_records(records: Sequence, val_fraction: float) -> Tuple[List, List]:
    """Split a list of records into train and validation.

    Two regimes, because the right thing depends on what the records ARE:

    * **one record** -> a contiguous tail split WITHIN it, via
      :func:`split_train_val`. There is only one experiment, so the held-out
      part has to come from its end.
    * **many records** -> hold out the last ``ceil(n * val_fraction)``
      RECORDS whole. ParWHF's 200 realizations are independent experiments;
      cutting inside one would put the same experiment on both sides of the
      split while leaving the other 199 untouched.

    Records are ``(name, u, y)`` triples (``benchmark_registry.Record``).
    Returns ``(train_records, validation_records)``.
    """
    if not records:
        raise ValueError("no records to split")
    kind = type(records[0])

    if len(records) == 1:
        name, u, y = records[0]
        u_tr, y_tr, u_va, y_va = split_train_val(u, y, val_fraction)
        return [kind(f"{name}_train", u_tr, y_tr)], [kind(f"{name}_validation", u_va, y_va)]

    n_val = int(np.ceil(len(records) * val_fraction))
    n_val = max(1, min(n_val, len(records) - 1))
    return list(records[:-n_val]), list(records[-n_val:])


def uniform_groups(records: Sequence) -> Dict[int, List]:
    """Group records by row count -- the axis ``np.stack`` cares about."""
    groups: Dict[int, List] = {}
    for rec in records:
        groups.setdefault(len(rec.u), []).append(rec)
    return groups


def write_records(
    folder: Path,
    records: Sequence,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    subsequence_length: Optional[int] = None,
) -> int:
    """Write several records into one folder, one CSV each.

    Every CSV in a folder that ``load_split_data`` reads must have the same row
    count -- it ``np.stack``s them and otherwise fails with a bare "all input
    arrays must have the same shape" that names no folder. This raises with the
    folder and the offending lengths instead.
    """
    lengths = {len(r.u) for r in records}
    if subsequence_length is None and len(lengths) > 1:
        raise ValueError(
            f"{folder.name}/ would hold records of different lengths {sorted(lengths)}; "
            "np.stack in load_split_data needs them uniform. Write them to "
            "separate sibling folders, or pass subsequence_length."
        )
    written = 0
    for rec in records:
        written += len(
            write_split(folder, rec.u, rec.y, input_cols, output_cols,
                        subsequence_length, stem=rec.name)
        )
    return written


def write_test_records(
    out_dir: Path,
    records: Sequence,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    selected: Optional[str] = None,
) -> Dict[str, int]:
    """Write ``test/`` plus one inert ``test_<name>/`` sibling per record.

    The siblings mean no record is ever lost even when the set is ragged, and
    ``evaluate.py --test-data`` can be pointed at any of them. ``test/`` gets
    the record named by ``selected``; with ``selected=None`` it gets them all,
    which is only valid when their lengths agree.
    """
    files: Dict[str, int] = {}
    for rec in records:
        folder = f"test_{rec.name}"
        files[folder] = write_records(out_dir / folder, [rec], input_cols, output_cols)

    if selected is None:
        chosen = list(records)
    else:
        chosen = [r for r in records if r.name == selected]
        if not chosen:
            raise ValueError(
                f"test record {selected!r} not found; have "
                f"{[r.name for r in records]}"
            )
    files["test"] = write_records(out_dir / "test", chosen, input_cols, output_cols)
    return files
