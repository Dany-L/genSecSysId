"""Post-processing feasibility check must run for both L regimes.

``check_input_condition`` rolls the model out, counts input-constraint
violations and (for ``nx == 2``) saves a diagnostic figure. It is called before
``post_process()`` and is NOT wrapped in try/except, so any plotting failure
aborts the whole post-processing run before metrics/SDP results are logged.

When ``L`` is not learnable (``learn_L=False`` => ``L = 0``, globally stable)
there is no regional ellipse/polytope safe set. These tests pin that the check
still counts violations, still emits a figure (a plain trajectory scatter, no
polytope), and never raises.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from sysid.evaluation import check_input_condition  # noqa: E402
from sysid.evaluation.post_processing import _scatter_trajectories  # noqa: E402
from sysid.models import SimpleLure  # noqa: E402


def _make_model(learn_L: bool) -> SimpleLure:
    return SimpleLure(
        nd=1, ne=1, nx=2, nw=1, activation="dzn",
        custom_params={"learn_L": learn_L},
    )


def _run_check(model, out_dir, tag):
    """Run check_input_condition inside a throwaway MLflow run."""
    torch.manual_seed(0)
    inputs_n = torch.randn(3, 20, 1) * 0.1
    x0 = torch.zeros((3, model.nx))
    mlflow.set_tracking_uri(f"file://{out_dir}/mlruns")
    with mlflow.start_run():
        return check_input_condition(
            model, inputs_n, x0, warmup_steps=0, run_output_dir=out_dir,
            tag=tag, title=f"test-{tag}", horizon=10,
        )


def test_check_input_condition_runs_without_learnable_L(tmp_path):
    """learn_L=False: the feasibility check runs, returns a (stable, unstable)
    count and writes the diagnostic figure without touching the polytope path."""
    model = _make_model(learn_L=False)
    assert not model.learn_L

    n_stable, n_unstable = _run_check(model, tmp_path, tag="global")

    assert n_stable + n_unstable == 3
    assert (tmp_path / "ellipse_polytope_global.png").exists()


def test_check_input_condition_runs_with_learnable_L(tmp_path):
    """learn_L=True: unchanged behaviour — the ellipse/polytope plot still runs."""
    model = _make_model(learn_L=True)
    assert model.learn_L

    n_stable, n_unstable = _run_check(model, tmp_path, tag="regional")

    assert n_stable + n_unstable == 3
    assert (tmp_path / "ellipse_polytope_regional.png").exists()


def test_scatter_trajectories_counts_match_violations():
    """The plain-scatter fallback classifies a trajectory unstable iff any c>0,
    mirroring plot_safe_set_trajectories' colouring/return contract."""
    # Two trajectories: the second breaches the constraint at one step.
    x_traj = np.zeros((2, 5, 2))
    c = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0],  # feasible
                  [-1.0, -1.0, 0.5, -1.0, -1.0]])   # one breach
    fig, count_stable, count_unstable = _scatter_trajectories(
        x_traj, c, warmup_steps=0, horizon=5,
    )
    assert count_stable == 1
    assert count_unstable == 1
    plt.close(fig)
