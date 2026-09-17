"""Plot the accuracy vs admissible-input-size trade-off from a sigma sweep.

Reads the MLflow **file store directly** -- no server, no UI. That is deliberate:
the trade-off figure is the deliverable, and it should not depend on the tracking
UI being reachable or working.

For each run of the sweep it recovers three things:

* **requested** ``c``      -- the constraint, ``training.sigma_target``;
* **achieved** ``sigma(U)``-- the last logged ``sigma_u``, i.e. what the run
  actually delivered. sigma is an OUTCOME, never an assumption, so the plot is
  drawn against this, with ``c`` annotated;
* **certifiable** ``sigma``-- ``post_process/max_s/s * sqrt(1-alpha^2)``, the
  largest sigma the *final* theta admits by SDP. The gap between achieved and
  certifiable is how much the run left on the table.

...against the prediction error on the converging and the diverging test split,
kept in **separate panels**: they differ by three orders of magnitude, and a
second y-axis on one plot would be unreadable and misleading.

**Stalled runs are excluded from the curve, not plotted as large errors.** With
``sigma_protect_s`` on, an unreachable ``c`` makes every repair get refused and
every batch roll back, so the run ends where it started. Its error is large
because training never happened, which would fake exactly the trade-off this
figure is meant to test. They are drawn as open markers and labelled instead.

Usage::

    python scripts/plot_sigma_tradeoff.py --experiment crnn-oneD-sigma-sweep
    python scripts/plot_sigma_tradeoff.py --experiment ... --out results/figures/tradeoff.png
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated against the dataviz six checks on the light surface #fcfcfb:
# CVD dE 23.8 (protan), normal dE 31.6, both >= 3:1 contrast.
# Two arms are two conditions, so they are two series -- not one line through
# both, which would draw a path between points that never share a run.
SERIES = "#2a78d6"      # slot 1: sigma_protect_s on  (sigma tracks c)
SERIES_2 = "#4a3aa7"    # slot 2: sigma_protect_s off (repair may free s)
CRITICAL = "#d03b3b"    # status: did not train (always label + legend, never colour alone)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"


def _read_metric_last(metric_dir: Path, key: str) -> Optional[float]:
    """Last logged value of a metric, or None."""
    p = metric_dir / key
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            last = float(parts[1])
    return last


def _read_metric_series(metric_dir: Path, key: str) -> Dict[int, float]:
    p = metric_dir / key
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[int(parts[2])] = float(parts[1])
    return out


def _read_param(param_dir: Path, key: str) -> Optional[str]:
    p = param_dir / key
    return p.read_text().strip() if p.exists() else None


def collect(mlruns: Path, experiments: List[str]) -> List[Dict[str, Any]]:
    """One record per run across the named experiments.

    Several experiments can be pooled because the honest x-axis is the
    **achieved** sigma, not the requested ``c`` or which arm produced it. Pooling
    the ``protect_s`` on/off arms is what supplies near-replicates at the same
    sigma, and those measure the run-to-run noise floor directly -- without them
    a single-seed curve cannot be told apart from scatter.
    """
    exp_dirs = []
    for d in mlruns.iterdir():
        meta = d / "meta.yaml"
        if d.is_dir() and meta.exists():
            for line in meta.read_text().splitlines():
                if line.startswith("name:") and line.split(":", 1)[1].strip() in experiments:
                    exp_dirs.append(d)
    if not exp_dirs:
        raise SystemExit(f"No experiment named any of {experiments!r} under {mlruns}")

    rows: List[Dict[str, Any]] = []
    for exp in exp_dirs:
        for run in sorted(exp.iterdir()):
            if not (run / "meta.yaml").exists():
                continue
            md, pd_ = run / "metrics", run / "params"
            if not md.exists():
                continue

            sigma_hist = _read_metric_series(md, "sigma_u")
            val_hist = _read_metric_series(md, "val_loss")
            rb_hist = _read_metric_series(md, "rollback_count")
            if not sigma_hist:
                continue

            target_raw = _read_param(pd_, "sigma_target")
            try:
                c = float(target_raw)
            except (TypeError, ValueError):
                # "auto" resolves to max_k||u_k|| at setup and is logged as a
                # metric, so the numeric value is still recoverable -- an auto
                # run is a point on the same axis, not a missing one.
                c = _read_metric_last(md, "sigma_target")
            constrained = (_read_param(pd_, "sigma_constraint") or "").lower() == "true"
            protect = (_read_param(pd_, "sigma_protect_s") or "").lower() == "true"

            alpha = _read_metric_last(md, "alpha") or 0.99
            k = math.sqrt(max(1.0 - alpha ** 2, 1e-12))
            s_max = _read_metric_last(md, "post_process/max_s/s")

            first_val = val_hist[min(val_hist)] if val_hist else None
            best_val = min(val_hist.values()) if val_hist else None
            epochs = max(sigma_hist) + 1
            rollbacks = sum(rb_hist.values()) if rb_hist else 0.0
            try:
                budget = int(_read_param(pd_, "max_epochs"))
            except (TypeError, ValueError):
                budget = None

            # Two ways a run can fail to be a data point about the trade-off,
            # both from the protect_s deadlock against an unreachable c:
            #
            #  1. it never improved on its own starting point -- the pure stall;
            #  2. it rolled back most batches, which decays the barrier weight
            #     every epoch, so the reg-weight early stop fires within a
            #     handful of epochs. The loss then looks bad because training
            #     barely happened, not because the constraint cost accuracy --
            #     which would fake exactly the trade-off this figure tests.
            no_progress = bool(
                first_val is not None and best_val is not None
                and best_val >= 0.999 * first_val
            )
            died_early = bool(
                budget and epochs < 0.15 * budget and rollbacks > 0
            )
            stalled = no_progress or died_early

            rows.append({
                "run_id": run.name,
                "c": c,
                "constrained": constrained,
                "protect_s": protect,
                "sigma_achieved": sigma_hist[max(sigma_hist)],
                "sigma_certifiable": (s_max * k) if s_max is not None else None,
                "rmse_conv": _read_metric_last(md, "id/conv/eval_rmse"),
                "rmse_div": _read_metric_last(md, "id/div/eval_rmse"),
                "best_val": best_val,
                "epochs": epochs,
                "budget": budget,
                "rollbacks": rollbacks,
                "rb_per_epoch": (rollbacks / epochs) if epochs else 0.0,
                "lambda": _read_metric_last(md, "dual_lambda"),
                "h_norm_P": _read_metric_last(md, "h_norm_P"),
                "stalled": stalled,
            })
    return rows


def _axis_scale(ax, which, values):
    """Scale and format one axis from the data it has to carry.

    * span > 1 decade -> log. Beyond ~3 decades ``ScalarFormatter`` degenerates
      into ``0.00 / 0.00 / 1e6``, so the wide case keeps matplotlib's
      ``10^n`` mathtext; the narrow case gets plain numbers, which read better.
    * otherwise linear, with scientific notation allowed for small magnitudes --
      an RMSE of 4.6e-4 renders as ``0.00100`` under the default.

    Minor tick labels are suppressed on log axes either way: matplotlib draws
    ``2x10^0 3x10^0 4x10^0`` on top of each other into unreadable mush.
    """
    from matplotlib.ticker import (LogFormatterSciNotation, NullFormatter,
                                   ScalarFormatter)

    finite = [v for v in values if v is not None and v > 0]
    axis = ax.xaxis if which == "x" else ax.yaxis
    if not finite:
        return ax
    span = max(finite) / min(finite)
    if span > 10:
        (ax.set_xscale if which == "x" else ax.set_yscale)("log")
        # Plain numbers read better than 10^n, but only where the magnitudes
        # are human-sized. An RMSE axis running 2e-4..5e-3 under ScalarFormatter
        # collapses to a single "0.00100" tick and hides the scale entirely.
        human = 1e-2 <= min(finite) and max(finite) <= 1e4 and span <= 1e3
        axis.set_major_formatter(
            ScalarFormatter() if human else LogFormatterSciNotation()
        )
        axis.set_minor_formatter(NullFormatter())
    else:
        from matplotlib.ticker import MaxNLocator
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 4))
        axis.set_major_formatter(fmt)
        # A near-flat series can otherwise be left with a single labelled tick,
        # which hides the scale the reader needs to judge "flat" against.
        axis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    return ax


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def _series(ax, pts, color, annotate=True):
    """Plot one arm: line + markers, ordered by x. ``pts`` = (x, y, label)."""
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        return
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color,
            linewidth=2, zorder=2)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=color,
            markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    if annotate:
        # Alternate above/below: on a steep stretch consecutive labels sit at
        # nearly the same height and overprint each other.
        for n, (x, y, lab) in enumerate(pts):
            if lab:
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(0, 9 if n % 2 == 0 else -15),
                            ha="center", fontsize=7, color=INK_2)


def _excluded(ax, pts):
    """Open markers for runs that did not train, labelled with how few epochs."""
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        return False
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", markerfacecolor="none",
            markeredgecolor=CRITICAL, markeredgewidth=2, markersize=9,
            linestyle="none", zorder=3)
    pts.sort()
    for n, (x, y, lab) in enumerate(pts):
        if lab:
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(0, 11 if n % 2 == 0 else -17), ha="center",
                        fontsize=6.5, color=CRITICAL)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mlruns", default="mlruns", help="path to the MLflow file store")
    ap.add_argument("--experiment", default=["crnn-oneD-sigma-sweep"], nargs="+",
                    help="one or more experiment names; runs are pooled")
    ap.add_argument("--out", default="results/figures/sigma_tradeoff.png")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    args = ap.parse_args()

    rows = collect(Path(args.mlruns).expanduser(), list(args.experiment))
    rows.sort(key=lambda r: (r["sigma_achieved"] is None, r["sigma_achieved"] or 0.0))
    if not rows:
        raise SystemExit("no runs with metrics found")

    hdr = (f"{'c':>8}{'sigma_ach':>12}{'sigma_cert':>12}{'realised':>10}"
           f"{'rmse_conv':>12}{'rmse_div':>11}{'lambda':>11}{'|h|_P':>10}"
           f"{'rb/ep':>7}{'ep':>5}  note")
    print(hdr)
    print("-" * len(hdr))

    def f(v, w, p=4):
        return f"{v:>{w}.{p}g}" if v is not None else " " * (w - 1) + "-"

    for r in rows:
        real = ""
        if r["sigma_certifiable"]:
            real = f"{100 * r['sigma_achieved'] / r['sigma_certifiable']:.0f}%"
        note = "DID NOT TRAIN (excluded)" if r["stalled"] else ""
        print(f"{f(r['c'],8)}{f(r['sigma_achieved'],12)}{f(r['sigma_certifiable'],12)}"
              f"{real:>10}{f(r['rmse_conv'],12)}{f(r['rmse_div'],11)}"
              f"{f(r['lambda'],11)}{f(r['h_norm_P'],10)}{r['rb_per_epoch']:>7.1f}"
              f"{r['epochs']:>5}  {note}")

    if args.csv:
        import csv as _csv
        p = Path(args.csv).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ntable -> {p}")

    swept = [r for r in rows if r["constrained"] and r["c"] is not None]
    control = next((r for r in rows if not r["constrained"]), None)
    if not swept:
        raise SystemExit("no constrained runs with a numeric sigma_target")

    ok = [r for r in swept if not r["stalled"]]
    bad = [r for r in swept if r["stalled"]]
    arms = [
        ("sigma_protect_s: true", SERIES, [r for r in ok if r["protect_s"]]),
        ("sigma_protect_s: false", SERIES_2, [r for r in ok if not r["protect_s"]]),
    ]
    arms = [a for a in arms if a[2]]
    all_c = [r["c"] for r in swept if r["c"] is not None]
    all_ach = [r["sigma_achieved"] for r in swept if r["sigma_achieved"] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    fig.patch.set_facecolor(SURFACE)

    # (a) did the run deliver what was asked? -- the y=x line is the whole point
    ax = axes[0]
    _style(ax)
    lo, hi = min(all_c + all_ach) * 0.6, max(all_c + all_ach) * 1.5
    ax.plot([lo, hi], [lo, hi], "--", color=INK_2, linewidth=1, zorder=1)
    for _, colour, rs in arms:
        _series(ax, [(r["c"], r["sigma_achieved"], "") for r in rs], colour,
                annotate=False)
    has_bad = _excluded(ax, [(r["c"], r["sigma_achieved"], "") for r in bad])
    _axis_scale(ax, "x", all_c + [lo, hi])
    _axis_scale(ax, "y", all_ach + [lo, hi])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.annotate("constraint met\n(achieved = c)", (0.97, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlabel("requested $c$  (normalised)", fontsize=9, color=INK)
    ax.set_ylabel(r"achieved $\sigma(\mathcal{U})$", fontsize=9, color=INK)
    ax.set_title("(a) constraint satisfaction", fontsize=10, color=INK, loc="left")

    # (b), (c) the trade-off itself, one panel per split -- never a second y-axis
    for ax, key, name in ((axes[1], "rmse_conv", "converging test split"),
                          (axes[2], "rmse_div", "diverging test split")):
        _style(ax)
        ys = [r[key] for r in swept if r[key] is not None]
        if not ys:
            ax.text(0.5, 0.5, "no evaluation metrics logged", ha="center",
                    va="center", fontsize=9, color=INK_2, transform=ax.transAxes)
        else:
            for _, colour, rs in arms:
                # Only the protected arm spans the sigma range, so only it gets
                # direct labels; the other four runs sit in one tight cluster
                # where per-point labels would overprint each other.
                lab = colour == SERIES
                _series(ax, [(r["sigma_achieved"], r[key],
                              f"c={r['c']:.3g}" if lab else "") for r in rs],
                        colour, annotate=lab)
            has_bad |= _excluded(
                ax, [(r["sigma_achieved"], r[key], f"{r['epochs']} ep") for r in bad])
            if control and control.get(key) is not None:
                ax.axhline(control[key], linestyle="--", color=INK_2, linewidth=1, zorder=1)
                ax.annotate("unconstrained", (ax.get_xlim()[0], control[key]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7.5, color=INK_2)
            _axis_scale(ax, "x", all_ach)
            _axis_scale(ax, "y", ys)
            # direct labels sit above the marks; without headroom the outermost
            # ones clip against the panel edge
            ax.margins(x=0.14, y=0.16)
        ax.set_xlabel(r"achieved $\sigma(\mathcal{U})$", fontsize=9, color=INK)
        ax.set_ylabel("test RMSE", fontsize=9, color=INK)
        ax.set_title(f"({'b' if key.endswith('conv') else 'c'}) {name}",
                     fontsize=10, color=INK, loc="left")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colour, marker="o", linewidth=2, markersize=7,
                      markeredgecolor=SURFACE, markeredgewidth=2, label=nm)
               for nm, colour, _ in arms]
    if has_bad:
        handles.append(Line2D([], [], color="none", marker="o", markerfacecolor="none",
                              markeredgecolor=CRITICAL, markeredgewidth=2, markersize=9,
                              label="did not train (excluded)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.24)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
