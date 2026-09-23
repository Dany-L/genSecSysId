"""Prepare any `nonlinear_benchmarks` dataset for this repo's loader.

Replaces the per-benchmark prep scripts: every dataset is described by a
:class:`~sysid.data.benchmark_registry.BenchmarkSpec` and this script just walks
the registry.

Usage::

    python scripts/prepare_benchmark.py --list
    python scripts/prepare_benchmark.py --benchmark Silverbox
    python scripts/prepare_benchmark.py --benchmark Silverbox --test-record arrow
    python scripts/prepare_benchmark.py --all

Layout written (what ``sysid.data.direct_loader.load_split_data`` reads)::

    <out-dir>/train/         CSVs, all with the SAME row count
    <out-dir>/validation/    idem
    <out-dir>/test/          idem
    <out-dir>/test_<name>/   one per test record; inert to the loader
    <out-dir>/metadata.json

The default ``--out-dir`` ends in ``/id``, and that component is not cosmetic:
``evaluate.py`` and ``sweep.py`` key the ``id``/``ood`` MLflow metric prefix off
an exact ``id`` or ``ood`` path component.

NOTE none of these benchmarks ships a validation set -- they give a
``(train_val, test)`` pair -- so validation is carved out of the training side
here, by record where there are several and by a contiguous tail where there is
only one.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sysid.data import benchmark_prep as bp          # noqa: E402
from sysid.data import benchmark_config as bc        # noqa: E402
from sysid.data.benchmark_registry import (          # noqa: E402
    REGISTRY,
    SUPPORTED,
    BenchmarkSpec,
    apply_declared_split,
    get,
    has_official_split,
    package_version,
)

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_ROOT = "~/genSecSysId-Data/data"

logger = logging.getLogger("prepare_benchmark")


def default_out_dir(name: str, root: str = DEFAULT_ROOT) -> Path:
    return Path(root).expanduser() / name / "id"


def prepare(
    spec: BenchmarkSpec,
    out_dir: Path,
    train_records: Sequence,
    test_records: Sequence,
    sampling_time: Optional[float],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    subsequence_length: Optional[int] = None,
    test_record: Optional[str] = None,
    clean: bool = True,
) -> dict:
    """Write the layout and return the metadata describing it.

    Split from ``main`` so tests can drive it with synthetic records instead of
    downloading anything.
    """
    out_dir = Path(out_dir).expanduser()
    siblings = [f"test_{r.name}" for r in test_records]
    if clean:
        bp.clean_folders(out_dir, [*bp.SPLIT_FOLDERS, *siblings])
    out_dir.mkdir(parents=True, exist_ok=True)

    train, validation = bp.split_records(train_records, val_fraction)
    files = {
        "train": bp.write_records(out_dir / "train", train, spec.input_cols,
                                  spec.output_cols, subsequence_length),
        "validation": bp.write_records(out_dir / "validation", validation, spec.input_cols,
                                       spec.output_cols, subsequence_length),
    }

    # Test records stay full-length: the benchmark error is defined over the
    # whole record, and load_split_data evaluates test with whole sequences.
    selected = test_record or spec.default_test_record
    if selected is None and len({len(r.u) for r in test_records}) > 1:
        # Ragged and nothing declared -- take the longest, the most informative
        # single record, rather than failing.
        selected = max(test_records, key=lambda r: len(r.u)).name
        logger.warning(
            "  test records are ragged and no --test-record given; "
            "putting the longest (%s) in test/", selected
        )
    files.update(bp.write_test_records(out_dir, test_records, spec.input_cols,
                                       spec.output_cols, selected))

    stats = {"train": _merge_stats(train), "validation": _merge_stats(validation)}
    for rec in test_records:
        stats[f"test_{rec.name}"] = bp.record_stats(rec.u, rec.y)

    # Ask the INSTALLED package rather than trusting the spec: ParWH is absent
    # from all_splitted_benchmarks in 0.1.2 and present in 1.0.1.
    derived = has_official_split(spec.name, spec.loader_aliases)
    official = spec.official_split if derived is None else derived

    metadata = {
        "dataset": spec.name,
        "source": "nonlinear_benchmarks (see reference)",
        # The splits are NOT stable across versions -- ParWH changed shape
        # between 0.1.2 and 1.0.1 -- so a result is only comparable to another
        # produced by the same one.
        "nonlinear_benchmarks_version": package_version(),
        "reference": spec.reference,
        "official_split": official,
        "sampling_time": sampling_time,
        "input_col": list(spec.input_cols),
        "output_col": list(spec.output_cols),
        "val_fraction": float(val_fraction),
        "subsequence_length": subsequence_length,
        "benchmark_state_initialization_window_length": spec.init_window,
        "test_record_in_test_folder": selected,
        "validation_note": (
            "This benchmark ships no validation set; validation is carved out "
            "of the training side "
            + ("by record." if len(train_records) > 1 else "as a contiguous tail.")
        ),
        "notes": spec.notes,
        "n_files": files,
        "records": stats,
    }
    bp.write_metadata(out_dir, metadata)
    return metadata


def _merge_stats(records: Sequence) -> dict:
    """Stats over a split, with ``n_samples`` the total across its records.

    ``n_samples`` stays the total because that is what the config's
    ``train_sequence_length`` has to fit inside, and tests read it.
    """
    import numpy as np

    stats = bp.record_stats(
        np.concatenate([r.u for r in records], axis=0),
        np.concatenate([r.y for r in records], axis=0),
    )
    stats["n_records"] = len(records)
    stats["record_length"] = int(len(records[0].u))
    return stats


def write_starter_config(
    spec, out_dir: Path, config_dir: Path, sampling_time, meta: dict,
    b2c2: float = bc.DEFAULT_B2C2, learning_rate: float = bc.DEFAULT_LEARNING_RATE,
    overwrite: bool = False, target_root: Optional[str] = None,
    write_sweep: bool = False, n_seeds: int = 3,
) -> Path:
    """Fit the balanced linear init, size the loop gain, and emit a config.

    The loop-gain search is a bisection rather than a sweep because ``gain = 0``
    opens the Lur'e loop: with a Schur ``A`` under ``alpha`` the LMI is then
    feasible by construction, so the search always has a lower bracket.
    """
    import copy
    import io
    import contextlib
    import numpy as np
    from sysid.config import Config
    from sysid.data import DataNormalizer
    from sysid.data.direct_loader import load_csv_folder
    from sysid.models.factory import create_model
    from sysid.models.linear_init import fit_linear_model, largest_feasible_loop_gain
    from sysid.optimization import LureCertificateSynthesizer

    u, y, _, _ = load_csv_folder(
        folder_path=str(out_dir / "train"), input_col=list(spec.input_cols),
        output_col=list(spec.output_cols), state_col=None, pattern="*.csv",
    )
    u, y = np.stack(u), np.stack(y)
    norm = DataNormalizer(method="scale_only")
    norm.fit(u, y)
    un = np.asarray(norm.transform_inputs(u)).reshape(u.shape)
    yn = np.asarray(norm.transform_outputs(y)).reshape(y.shape)
    input_floor = float(np.sqrt((un ** 2).sum(-1).max()))

    logger.info("  fitting a balanced linear model at nx=%d ...", spec.default_nx)
    linear = fit_linear_model(un, yn, nx=spec.default_nx)
    init_dir = out_dir / "linear_init"
    linear.save(init_dir)
    # Just above rho(A): the Lyapunov condition needs rho(A) < alpha, and
    # nothing else pins alpha.
    alpha_0 = min(0.99999, linear.rho + 0.5 * (1.0 - linear.rho))

    template = Config.from_dict({
        "data": {"train_path": str(out_dir), "input_col": list(spec.input_cols),
                 "output_col": list(spec.output_cols), "normalization_method": "scale_only"},
        "model": {"model_type": "crnn", "nx": spec.default_nx, "nw": spec.default_nw,
                  "activation": "dzn",
                  "custom_params": {"learn_L": True, "freeze_alpha": False},
                  "initialization": {"method": "identity"}},
        "training": {"use_custom_regularization": True},
    })
    probed = {}

    def feasible(gain: float) -> bool:
        cfg = copy.deepcopy(template)
        cfg.model.custom_params["alpha_0"] = alpha_0
        cfg.model.custom_params["identity_init"] = {
            "A": {"load_from": str(init_dir / "A.npy")},
            "B": {"load_from": str(init_dir / "B.npy")},
            "C": {"load_from": str(init_dir / "C.npy")},
            "B2": {"std": b2c2 * gain}, "C2": {"std": b2c2 * gain},
            "D21": {"std": 0.5},
        }
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                logging.disable(logging.CRITICAL)
                model = create_model(cfg)
                model.initialize_parameters(train_inputs=u, train_outputs=y,
                                            train_states=None, normalizer=norm)
                if not model.check_constraints():
                    return False
                sol = LureCertificateSynthesizer.from_model(model).max_s()
            if sol is None:
                return False
            probed[gain] = float(sol.s)
            # Both conditions: the LMI must hold AND the certified set must admit
            # the training inputs, or MaxS goes infeasible during training.
            return float(sol.s) >= input_floor
        except Exception:
            return False
        finally:
            logging.disable(logging.NOTSET)

    try:
        gain, _ = largest_feasible_loop_gain(feasible, hi=1.0, tolerance=0.1)
    except RuntimeError as exc:
        # Even the open loop cannot admit the inputs -- report rather than guess.
        logger.warning("  loop-gain search: %s", exc)
        gain = 0.0
    achieved = b2c2 * gain
    logger.info("  loop gain %.3f -> B2=C2=%g (requested %g)", gain, achieved, b2c2)

    # The config is read on whichever machine trains. When that is not this
    # one (a GPU cluster with its own /data root), --target-root rewrites the
    # embedded paths without moving any files, so the tree can be rsynced.
    cfg_out_dir, cfg_init_dir = out_dir, init_dir
    if target_root:
        rel = out_dir.relative_to(Path(out_dir).parents[1])
        cfg_out_dir = Path(target_root) / rel
        cfg_init_dir = cfg_out_dir / init_dir.name

    train_stats = meta["records"]["train"]
    text = bc.render_config(
        spec, cfg_out_dir, cfg_init_dir, linear, alpha_0, achieved, b2c2, sampling_time,
        n_records=train_stats["n_records"], record_length=train_stats["record_length"],
        input_floor=input_floor, s_at_gain=probed.get(gain),
        learning_rate=learning_rate,
    )
    config_dir = Path(config_dir).expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"crnn_{spec.name.lower().replace('_', '-')}.yaml"
    if path.exists() and not overwrite:
        # These configs get hand-tuned, and a generated one would silently
        # discard that. Refuse rather than clobber.
        raise FileExistsError(
            f"{path} already exists; pass --overwrite-configs to replace it "
            "(any hand tuning in it will be lost)"
        )
    path.write_text(text)

    if write_sweep:
        sweep_path = config_dir / f"sweep_{spec.name.lower().replace('_', '-')}.yaml"
        if sweep_path.exists() and not overwrite:
            raise FileExistsError(
                f"{sweep_path} already exists; pass --overwrite-configs to replace it"
            )
        base_ref = str(path)
        sweep_ref = str(sweep_path)
        if target_root:
            # Both references are read on the TRAINING machine, so both follow
            # --target-root, not where these files happen to be written now.
            base_ref = str(Path(target_root) / "configs" / path.name)
            sweep_ref = str(Path(target_root) / "configs" / sweep_path.name)
        sweep_path.write_text(
            bc.render_sweep(spec, base_ref, achieved, learning_rate, n_seeds,
                            sweep_path=sweep_ref)
        )
        logger.info("  sweep         : %s", sweep_path)
    return path


def _print_registry() -> None:
    print(f"{'benchmark':32s} {'status':10s} notes")
    for name, spec in REGISTRY.items():
        status = "supported" if spec.supported else "UNSUPPORTED"
        detail = spec.unsupported or (spec.notes.split(".")[0] + "." if spec.notes else "")
        print(f"  {name:30s} {status:10s} {detail[:90]}")
    print(f"\n{len(SUPPORTED)} supported: {', '.join(SUPPORTED)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", help="name from --list")
    parser.add_argument("--all", action="store_true",
                        help="prepare every supported benchmark")
    parser.add_argument("--list", action="store_true",
                        help="show the registry and exit")
    parser.add_argument("--out-dir", help=f"default: {DEFAULT_ROOT}/<Name>/id")
    parser.add_argument("--data-root", default=DEFAULT_ROOT,
                        help="root under which <Name>/id is created")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="held out of the training side for validation")
    parser.add_argument("--subsequence-length", type=int, default=None,
                        help="rows per train/validation CSV; omit for full-length")
    parser.add_argument("--test-record", default=None,
                        help="which test record lands in test/ (see metadata.json)")
    parser.add_argument("--write-config", action="store_true",
                        help="also fit the balanced linear init and emit a starter config")
    parser.add_argument("--config-dir", default="~/genSecSysId-Data/configs",
                        help="where --write-config puts the YAML")
    parser.add_argument("--overwrite-configs", action="store_true",
                        help="replace an existing generated config (loses hand tuning)")
    parser.add_argument("--write-sweep", action="store_true",
                        help="also emit sweep_<name>.yaml with the three arms")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="seeds per arm in the generated sweep")
    parser.add_argument("--target-root", default=None,
                        help="data root as it will appear on the machine that "
                             "TRAINS (e.g. /data/work/<user>/benchmarks). Only "
                             "rewrites paths inside the generated files.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep pre-existing CSVs (risks mixing row counts)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.list:
        _print_registry()
        return 0
    if not args.benchmark and not args.all:
        parser.error("give --benchmark NAME, or --all, or --list")

    names = SUPPORTED if args.all else [args.benchmark]
    if args.all and args.out_dir:
        parser.error("--out-dir makes no sense with --all; use --data-root")

    failures = []
    for name in names:
        spec = get(name)
        out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir(
            name, args.data_root
        )
        logger.info("\n=== %s -> %s", name, out_dir)
        try:
            train, test, ts = spec.fetch(force_download=args.force_download)
            if not test:
                # The package gave a flat list with no split of its own, so the
                # spec's declared choice decides. Anything else is a registry bug.
                if not spec.declared_train_record:
                    raise ValueError(
                        f"{name} returned no test records and its spec declares "
                        "no split; set declared_train_record/declared_test_records"
                    )
                train, test = apply_declared_split(spec, train)
            meta = prepare(
                spec, out_dir, train, test, ts,
                val_fraction=args.val_fraction,
                subsequence_length=args.subsequence_length,
                test_record=args.test_record,
                clean=not args.no_clean,
            )
        except Exception as exc:  # keep --all going, report at the end
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            failures.append((name, exc))
            continue

        logger.info("  sampling_time : %s", meta["sampling_time"])
        logger.info("  columns       : %s (in), %s (out)",
                    ", ".join(meta["input_col"]), ", ".join(meta["output_col"]))
        logger.info("  test/ holds   : %s", meta["test_record_in_test_folder"])
        for split in ("train", "validation"):
            st = meta["records"][split]
            logger.info("  %-12s %d record(s) x %d samples", split + ":",
                        st["n_records"], st["record_length"])
        if not meta["official_split"]:
            logger.warning("  NOTE: no official train/test split -- see metadata.json")

        if args.write_config:
            try:
                path = write_starter_config(spec, out_dir, args.config_dir,
                                            meta["sampling_time"], meta,
                                            overwrite=args.overwrite_configs,
                                            target_root=args.target_root,
                                            write_sweep=args.write_sweep,
                                            n_seeds=args.n_seeds)
                logger.info("  config        : %s", path)
            except Exception as exc:
                logger.error("  config generation FAILED: %s: %s", type(exc).__name__, exc)
                failures.append((f"{name} (config)", exc))

    if failures:
        logger.error("\n%d benchmark(s) failed:", len(failures))
        for name, exc in failures:
            logger.error("  %s: %s", name, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
