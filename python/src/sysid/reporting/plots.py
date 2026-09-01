"""Plotting helpers for the comparison/reporting notebook.

Val-loss training curves (best individual run + mean/std over the best HP
group) and the multi-run trajectory comparison. Pure aggregation helpers
(``smooth_ema``, ``mean_std_per_step``) are unit-tested; the plotting and
MLflow-artifact functions keep the notebook's original behaviour but take their
inputs as arguments instead of globals.
"""

import os
from typing import Dict, List

import numpy as np


# ── Pure aggregation helpers ──────────────────────────────────────────────────
def smooth_ema(values: List[float], alpha: float = 0.9) -> List[float]:
    """Exponential moving average; alpha -> 1 means heavier smoothing."""
    if not values:
        return values
    s = [values[0]]
    for v in values[1:]:
        s.append(alpha * s[-1] + (1 - alpha) * v)
    return s


def mean_std_per_step(histories: List[Dict[int, float]]):
    """Aggregate per-run step->value dicts into per-step mean and std."""
    all_steps = sorted(set(s for h in histories for s in h))
    means, stds = [], []
    for step in all_steps:
        vals = [h[step] for h in histories if step in h]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return all_steps, means, stds


def _default_client():
    import mlflow

    return mlflow.tracking.MlflowClient()


def fetch_histories(client, run_ids: List[str], metric_key: str) -> List[Dict[int, float]]:
    """Return a list of step->value dicts, one per run id."""
    out = []
    for run_id in run_ids:
        try:
            h = client.get_metric_history(run_id, metric_key)
            if h:
                out.append({pt.step: pt.value for pt in h})
        except Exception as e:  # pragma: no cover - network failures
            print(f"  Error fetching {metric_key} for {run_id}: {e}")
    return out


# ── Val-loss curves ───────────────────────────────────────────────────────────
def plot_val_loss_individual(
    ax, div_key, best_runs, stab_type_dict, model_name_map, client,
    val_loss_key="val_loss", smoothing=0.6,
):
    """Best individual run per model class (smoothed line + raw faint line)."""
    for stab_key in stab_type_dict:
        info = best_runs.get(f"stab={stab_key}_div={div_key}")
        if info is None:
            continue
        try:
            h = client.get_metric_history(info["indiv_run_id"], val_loss_key)
        except Exception as e:  # pragma: no cover
            print(f"  [{stab_key}/{div_key}] individual: {e}")
            h = []
        if h:
            steps = [pt.step for pt in h]
            values = [pt.value for pt in h]
            line, = ax.semilogy(
                steps, smooth_ema(values, smoothing),
                label=model_name_map[stab_key], linewidth=1.8,
            )
            ax.semilogy(steps, values, color=line.get_color(), alpha=0.15, linewidth=0.7)


def plot_val_loss_mean_std(
    ax, div_key, best_runs, stab_type_dict, model_name_map, client,
    val_loss_key="val_loss", smoothing=0.6,
):
    """Mean +/- std across all runs in the best HP group, per model class."""
    for stab_key in stab_type_dict:
        info = best_runs.get(f"stab={stab_key}_div={div_key}")
        if info is None:
            continue
        histories = fetch_histories(client, info["all_run_ids"], val_loss_key)
        if not histories:
            continue
        steps_m, means, stds = mean_std_per_step(histories)
        sm = smooth_ema(means, smoothing)
        ss = smooth_ema(stds, smoothing)
        line, = ax.semilogy(steps_m, sm, label=model_name_map[stab_key], linewidth=1.8)
        color = line.get_color()
        lower = np.clip(np.array(sm) - np.array(ss), 1e-10, None)
        upper = np.array(sm) + np.array(ss)
        ax.fill_between(steps_m, lower, upper, color=color, alpha=0.2)


def _style_val_loss_ax(ax, val_loss_key, xlabel=False):
    ax.set_ylabel(val_loss_key)
    if xlabel:
        ax.set_xlabel("Epoch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")


def plot_val_loss(
    best_runs, stab_type_dict, use_div_traj, model_name_map, fig_out_dir,
    val_loss_key="val_loss", smoothing=0.6, client=None, save_tikz=True,
):
    """Save one standalone .tex per (row x div) subplot and return a combined
    2-row x n_div grid figure (row 0: best individual, row 1: mean +/- std)."""
    import matplotlib.pyplot as plt

    if client is None:
        client = _default_client()
    fig_out_dir = _as_path(fig_out_dir)
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("individual", plot_val_loss_individual),
        ("mean_std", plot_val_loss_mean_std),
    ]
    if save_tikz:
        import tikzplotlib

        for row_name, plot_fn in plot_specs:
            for div_key in use_div_traj:
                fig_s, ax_s = plt.subplots(figsize=(5, 3.5))
                plot_fn(ax_s, div_key, best_runs, stab_type_dict, model_name_map,
                        client, val_loss_key, smoothing)
                _style_val_loss_ax(ax_s, val_loss_key, xlabel=True)
                fig_s.tight_layout()
                out_path = fig_out_dir / f"val_loss_{row_name}_div_{div_key}.tex"
                # tikzplotlib.save(str(out_path), figure=fig_s)
                plt.close(fig_s)
                print(f"Saved {out_path.name}")

    n_div = len(use_div_traj)
    fig, axes = plt.subplots(2, n_div, figsize=(5 * n_div, 8), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for col, div_key in enumerate(use_div_traj):
        plot_val_loss_individual(axes[0, col], div_key, best_runs, stab_type_dict,
                                 model_name_map, client, val_loss_key, smoothing)
        plot_val_loss_mean_std(axes[1, col], div_key, best_runs, stab_type_dict,
                               model_name_map, client, val_loss_key, smoothing)
        _style_val_loss_ax(axes[0, col], val_loss_key)
        _style_val_loss_ax(axes[1, col], val_loss_key, xlabel=True)
    fig.tight_layout()
    return fig


# ── Multi-run trajectory comparison ───────────────────────────────────────────
def load_evaluation_data(internal_id, run_id):
    """Download evaluation artifacts and load inputs/targets/predictions."""
    import mlflow

    try:
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="evaluation"
        )
        data = {
            "internal_id": internal_id,
            "run_id": run_id,
            "inputs": np.load(os.path.join(artifact_dir, "inputs.npy")),
            "targets": np.load(os.path.join(artifact_dir, "targets.npy")),
            "predictions": np.load(os.path.join(artifact_dir, "predictions.npy")),
        }
        print(f"✓ {internal_id} ({run_id}): targets {data['targets'].shape}")
        return data
    except Exception as e:  # pragma: no cover - network failures
        print(f"✗ Failed to load {internal_id}: {e}")
        return None


def select_random_trajectories(data_dict, n_trajectories=5, seed=42):
    """Randomly pick up to ``n_trajectories`` trajectories (reproducible)."""
    rng = np.random.default_rng(seed)
    n_total = data_dict["targets"].shape[0]
    n_select = min(n_trajectories, n_total)
    idx = np.sort(rng.choice(n_total, size=n_select, replace=False))
    return {
        "indices": idx,
        "inputs": data_dict["inputs"][idx],
        "targets": data_dict["targets"][idx],
        "predictions": data_dict["predictions"][idx],
        "internal_id": data_dict.get("internal_id"),
        "run_id": data_dict.get("run_id"),
    }


def load_run_comparison(run_id_map, n_trajectories=5, seed=42):
    """Load + sample trajectories for every run in ``run_id_map`` (id->run_id)."""
    selected = []
    for internal_id, run_id in run_id_map.items():
        data = load_evaluation_data(internal_id, run_id)
        if data is not None:
            selected.append(select_random_trajectories(data, n_trajectories, seed))
    print(f"\nLoaded {len(selected)}/{len(run_id_map)} runs")
    return selected


def plot_comparison_all_runs(selected_runs, n_trajectories=5):
    """Compare predictions (left) and errors (right) across runs for the same
    randomly selected trajectories. Returns ``(fig, axes)``."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_trajectories, 2, figsize=(14, 3 * n_trajectories))
    if n_trajectories == 1:
        axes = axes.reshape(1, -1)

    targets = selected_runs[0]["targets"]
    target_indices = selected_runs[0]["indices"]

    for traj_idx in range(n_trajectories):
        y = targets[traj_idx]

        ax = axes[traj_idx, 0]
        ax.plot(y, "--", linewidth=2, label="Target", alpha=0.7, color="black")
        for selected in selected_runs:
            ax.plot(selected["predictions"][traj_idx], "-", linewidth=1.5,
                    label=selected["internal_id"], alpha=0.7)
        ax.set_title(f"Trajectory {target_indices[traj_idx]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Output")
        ax.set_xlabel("Time step")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[traj_idx, 1]
        for selected in selected_runs:
            ax.plot(selected["predictions"][traj_idx] - y, "-", linewidth=1.5,
                    label=selected["internal_id"], alpha=0.7)
        ax.set_title("Prediction Errors", fontsize=11, fontweight="bold")
        ax.set_ylabel("Prediction - Target")
        ax.set_xlabel("Time step")
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _as_path(p):
    from pathlib import Path

    return p if isinstance(p, Path) else Path(p)
