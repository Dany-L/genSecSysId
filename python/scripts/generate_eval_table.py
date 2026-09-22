"""Generate the per-experiment evaluation table.

Usage:
    python scripts/generate_eval_table.py [--config results/results_confg.yaml]
                                          [--out results/tables/eval-table.tex]
                                          [--no-divergence] [--experiment NAME]

Reads the experiment/model layout from the config YAML, picks per model class
the run with the best `criterion` metric, and renders the LaTeX table of
results/eval_table_template.tex. Every row carries its run id and MLflow
experiment name as a trailing LaTeX comment.

ONE TABLE PER `stability` CLASS of the config: the synthetic systems
(`regionally-stable`) and the measured benchmarks (`unknown`) no longer share a
table, so `--out .../eval-table.tex` writes `eval-table-regionally-stable.tex`
and `eval-table-unknown.tex`. On the regionally-stable table the highest
`# div/total` per column is bold: those inputs make the true system diverge, so
the model reproducing most of them is the faithful one. The measured benchmarks
have no diverging trajectories at all, so that table drops the column outright
and the rollout is never run for them.

All columns but one come straight off the run. The "# div/total" column is not
logged, so for experiments whose runs have a diverging test set the best model
is downloaded and replayed: each diverging input, then the input held at zero
for `padding` steps, counted as diverged when ||x_N|| > eps. Experiments with
no diverging inputs logged render "--" there.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import yaml

from sysid.config import Config, allow_file_store
from sysid.data import DataNormalizer
from sysid.models import load_model
from sysid.reporting.eval_table import (
    N_PARS_PARAM,
    NW_PARAM,
    SIGMA_KEY,
    Y_BAR_KEYS,
    Y_MAX_KEY,
    EvalRow,
    build_eval_table,
    count_diverged,
    first_metric,
    is_regional,
    load_table_config,
    param_int,
    select_best_run,
    split_by_stability,
    stability_slug,
)

DEFAULT_CONFIG = "results/results_confg.yaml"
DEFAULT_OUT = "results/tables/eval-table.tex"

logger = logging.getLogger("generate_eval_table")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def set_tracking_uri(server_uri: Optional[str]) -> str:
    """Point MLflow at the experiment's store; ``None`` -> the local ./mlruns."""
    allow_file_store()
    uri = server_uri or f"file://{project_root() / 'mlruns'}"
    mlflow.set_tracking_uri(uri)
    return uri


class ConfigLoader:
    """Cached ``outputs/config.yaml`` lookup, for runs that carry no tags."""

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def __call__(self, run_id: str) -> Optional[Dict[str, Any]]:
        if run_id not in self._cache:
            try:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="outputs/config.yaml"
                )
                with open(path) as fh:
                    self._cache[run_id] = yaml.safe_load(fh)
            except Exception as exc:  # run predates the artifact, or is pruned
                logger.debug("no logged config for %s: %s", run_id, exc)
                self._cache[run_id] = None
        return self._cache[run_id]


def divergence_counts(run_id: str, divergence: Dict[str, Any]):
    """``((n_diverged, n_total), None)`` for a run, else ``(None, reason)``.

    The reason is carried onto the table row so a "--" that comes from a run
    with no diverging test set is not silently confused with one that comes
    from a model this code can no longer load.
    """
    artifact = divergence["diverging_inputs"]
    try:
        inputs_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact
        )
    except Exception:
        logger.info("      no %s logged -> '# div/total' is '--'", artifact)
        return None, "no diverging test set logged"

    try:
        outputs_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="outputs"
        )
        models_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="models"
        )
        config = Config.from_yaml(os.path.join(outputs_dir, "config.yaml"))
        model = load_model(os.path.join(models_dir, "best_model.pt"), config)
        normalizer = DataNormalizer.load(os.path.join(models_dir, "normalizer.json"))
    except Exception as exc:
        # Most often a config that predates a removed field -- the run is fine,
        # this code just cannot rebuild the model from it any more.
        logger.warning("      could not load model for %s: %s", run_id, exc)
        return None, f"div not computed: {type(exc).__name__} loading the model"

    counts = count_diverged(
        model,
        normalizer,
        np.load(inputs_path),
        padding=divergence["padding"],
        eps=divergence["eps"],
    )
    logger.info("      diverged %d/%d", *counts)
    return counts, None


def build_rows(config: Dict[str, Any], with_divergence: bool, only: Optional[str]):
    rows = []
    for experiment in config["experiments"]:
        if only and only not in (experiment["name"], experiment["mlflow_experiment_name"]):
            continue

        uri = set_tracking_uri(experiment["mlflow_server_uri"])
        name = experiment["mlflow_experiment_name"]
        criterion = experiment["criterion"]
        logger.info("\n%s  [%s @ %s]", experiment["name"], name, uri)

        try:
            runs_df = mlflow.search_runs(experiment_names=[name], max_results=50_000)
        except Exception as exc:
            logger.error("  could not search '%s': %s", name, exc)
            runs_df = None

        if runs_df is None or runs_df.empty:
            logger.warning("  no runs found")
            for run in experiment["runs"]:
                rows.append(
                    EvalRow(
                        experiment=experiment["name"],
                        model_name=run["model_name"],
                        mlflow_experiment_name=name,
                        stability=experiment["stability"],
                        note=f"no runs in experiment {name}",
                    )
                )
            continue

        loader = ConfigLoader()
        for run in experiment["runs"]:
            model_name = run["model_name"]
            best = select_best_run(runs_df, run["selector"], criterion, config_loader=loader)
            if best is None:
                logger.warning("  %-7s no run matches %s", model_name, run["selector"])
                rows.append(
                    EvalRow(
                        experiment=experiment["name"],
                        model_name=model_name,
                        mlflow_experiment_name=name,
                        stability=experiment["stability"],
                        note=f"no run matching {run['selector']}",
                    )
                )
                continue

            run_id = best["run_id"]
            nrmse = first_metric(best, [criterion])
            logger.info(
                "  %-7s %s  %s=%.4f  n_pars=%s nw=%s",
                model_name, run_id, criterion, nrmse,
                param_int(best, N_PARS_PARAM), param_int(best, NW_PARAM),
            )

            diverged, note = None, None
            if with_divergence and is_regional(experiment["stability"]):
                diverged, note = divergence_counts(run_id, experiment["divergence"])

            rows.append(
                EvalRow(
                    experiment=experiment["name"],
                    model_name=model_name,
                    run_id=run_id,
                    mlflow_experiment_name=name,
                    stability=experiment["stability"],
                    nrmse=nrmse,
                    y_bar=first_metric(best, Y_BAR_KEYS),
                    y_max=first_metric(best, [Y_MAX_KEY]),
                    diverged=diverged,
                    # Only the regional arm has an admissible input set to size;
                    # the table leaves the row blank for the others anyway.
                    sigma_u=(
                        first_metric(best, [SIGMA_KEY])
                        if model_name == "GenSec" else None
                    ),
                    n_pars=param_int(best, N_PARS_PARAM),
                    nw=param_int(best, NW_PARAM),
                    note=note,
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="experiment layout YAML")
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="LaTeX output path; the stability class is appended to the stem, "
             "so eval-table.tex -> eval-table-regionally-stable.tex, "
             "eval-table-unknown.tex",
    )
    parser.add_argument(
        "--nrmse-significant", type=int, default=2,
        help="significant digits the smallest NRMSE in a column must keep; the "
             "decimal count per column follows from it",
    )
    parser.add_argument(
        "--significant-digits", type=int, default=3,
        help="significant figures for y_max, ybar and sigma",
    )
    parser.add_argument(
        "--min-decimals", type=int, default=2, help="floor on the per-column decimals"
    )
    parser.add_argument(
        "--max-decimals", type=int, default=6, help="cap on the per-column decimals"
    )
    parser.add_argument("--experiment", help="only this experiment (name or MLflow name)")
    parser.add_argument(
        "--no-divergence",
        action="store_true",
        help="skip the model rollout; '# div/total' renders '--'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    config = load_table_config(config_path)

    rows = build_rows(config, not args.no_divergence, args.experiment)
    if not rows:
        logger.error("no rows produced -- nothing written")
        return 1

    divergence = config["experiments"][0]["divergence"]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One table per stability class: the synthetic systems are compared on a
    # divergence count the measured benchmarks cannot report at all.
    for stability, group in split_by_stability(rows):
        # The column exists only where a diverging test set does, and there the
        # highest count is the faithful model, so it is the one emphasised.
        shows_divergence = is_regional(stability)
        comment = (
            f"generated by scripts/generate_eval_table.py from {config_path.name}; "
            f"stability: {stability}; "
            f"NRMSE: lowest per column bold, second lowest bold-italic"
        )
        if shows_divergence:
            comment += (
                f"; '# div/total': ||x_N|| > {divergence['eps']} after "
                f"{divergence['padding']} zero-input steps, highest per column bold"
            )
        tex = build_eval_table(
            group,
            comment=comment,
            nrmse_significant=args.nrmse_significant,
            min_decimals=args.min_decimals,
            max_decimals=args.max_decimals,
            significant_digits=args.significant_digits,
            with_diverged=shows_divergence,
            emphasize_diverged=shows_divergence,
        )
        path = out_path.with_name(
            f"{out_path.stem}-{stability_slug(stability)}{out_path.suffix}"
        )
        path.write_text(tex + "\n")
        logger.info("\nSaved %s\n\n%s", path, tex)
    missing = [r for r in rows if r.run_id is None]
    incomplete = [r for r in rows if r.run_id is not None and r.note]
    if missing:
        logger.warning(
            "\n%d row(s) had no matching run:\n%s",
            len(missing),
            "\n".join(f"  {r.experiment} / {r.model_name}: {r.note}" for r in missing),
        )
    if incomplete:
        logger.warning(
            "\n%d row(s) have no divergence count:\n%s",
            len(incomplete),
            "\n".join(f"  {r.experiment} / {r.model_name}: {r.note}" for r in incomplete),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
