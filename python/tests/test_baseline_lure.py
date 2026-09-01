"""Tests for ``scripts/baseline_lure.py`` — the fixed-Lur'e reference baseline.

The script's job is to make one number set trustworthy: the fit and the
certificate of a hand-built ``.npz`` model, measured through the *same* loaders,
normalizer, warmup and metric a trained run is measured through. The two things
that can silently corrupt that are

1. the physical -> normalized rescaling of θ (get it wrong and the baseline
   quietly reports a different model's error), and
2. the record itself (numpy/NaN leaking into json.dump, or the input-condition
   tally miscounting),

so those get direct tests, plus one end-to-end run on a synthetic dataset.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
import cvxpy as cp

from sysid.data.normalizer import DataNormalizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    """Import scripts/baseline_lure.py (not a package module) by path."""
    spec = importlib.util.spec_from_file_location(
        "baseline_lure", REPO_ROOT / "scripts" / "baseline_lure.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = _load_script()


def _mosek_available() -> bool:
    if "MOSEK" not in cp.installed_solvers():
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize((x - 1) ** 2), [x >= 0]).solve(solver=cp.MOSEK, verbose=False)
        return True
    except Exception:
        return False


requires_mosek = pytest.mark.skipif(
    not _mosek_available(), reason="MOSEK solver not available/licensed"
)


# --------------------------------------------------------------------- fixtures
def _theta(nx: int = 2, nz: int = 2, seed: int = 0) -> dict:
    """A small stable Lur'e system in physical units."""
    rng = np.random.default_rng(seed)
    return {
        "A": np.array([[0.9, 0.05], [-0.05, 0.88]]),
        "B": np.array([[0.0], [0.05]]),
        "B2": 0.01 * rng.standard_normal((nx, nz)),
        "C": np.array([[1.0, 0.0]]),
        "C2": np.array([[2.0, 0.0], [0.5, 0.0]]),
        "D": np.zeros((1, 1)),
        "D12": np.zeros((1, nz)),
        "D21": np.zeros((nz, 1)),
    }


def _dzn(z):
    return np.maximum(np.abs(z) - 1.0, 0.0) * np.sign(z)


def _rollout(theta: dict, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Physical-unit reference rollout: y_k for k = 0..T-1."""
    x, y = x0.astype(float).copy(), []
    for k in range(len(u)):
        uk = np.atleast_1d(u[k])
        w = _dzn(theta["C2"] @ x + theta["D21"] @ uk)
        y.append((theta["C"] @ x + theta["D"] @ uk + theta["D12"] @ w).item())
        x = theta["A"] @ x + theta["B"] @ uk + theta["B2"] @ w
    return np.array(y)


def _fitted_normalizer(u_std: float, y_std: float) -> DataNormalizer:
    n = DataNormalizer(method="scale_only")
    n.fit(np.full((1, 4, 1), u_std), np.full((1, 4, 1), y_std))
    # fit() on a constant array gives std 0; set the scales explicitly instead.
    n.input_std = np.array([[[u_std]]])
    n.output_std = np.array([[[y_std]]])
    return n


# ------------------------------------------------------- physical -> normalized
class TestToNormalizedUnits:
    """The rescaling must be an exact change of units, not a new model."""

    def test_roundtrip_reproduces_the_physical_output(self):
        theta = _theta()
        norm = _fitted_normalizer(u_std=3.0, y_std=0.25)
        theta_n = bl.to_normalized_units(theta, norm)

        rng = np.random.default_rng(1)
        u = 4.0 * rng.standard_normal(50)          # physical input
        x0 = np.array([0.3, -0.2])                  # physical state (never scaled)

        y_phys = _rollout(theta, x0, u)
        # Same physical x0, normalized input -> normalized output.
        e_norm = _rollout(theta_n, x0, u / 3.0)

        assert np.allclose(y_phys, e_norm * 0.25, atol=1e-12)

    def test_state_and_nonlinearity_channels_are_untouched(self):
        """x stays physical, so A/B2/C2 must not move — only the u and y ports."""
        theta = _theta()
        theta_n = bl.to_normalized_units(theta, _fitted_normalizer(3.0, 0.25))
        for key in ("A", "B2", "C2"):
            assert np.allclose(theta_n[key], theta[key])
        assert np.allclose(theta_n["B"], theta["B"] * 3.0)
        assert np.allclose(theta_n["C"], theta["C"] / 0.25)

    def test_no_normalizer_is_the_identity(self):
        theta = _theta()
        theta_n = bl.to_normalized_units(theta, None)
        for key in theta:
            assert np.allclose(theta_n[key], theta[key])

    def test_rejects_offset_normalizations(self):
        """'standard'/'minmax' add an offset the Lur'e structure cannot absorb."""
        norm = DataNormalizer(method="standard")
        norm.fit(np.random.randn(2, 5, 1), np.random.randn(2, 5, 1))
        with pytest.raises(ValueError, match="scale_only"):
            bl.to_normalized_units(_theta(), norm)


# ------------------------------------------------------------------ record bits
class TestRecord:
    def test_jsonable_flattens_numpy_torch_and_nonfinite(self):
        out = bl.jsonable({
            "a": np.arange(4).reshape(2, 2),
            "b": np.float64(1.5),
            "c": torch.tensor([1.0, 2.0]),
            "d": float("nan"),
            "e": np.True_,
            "f": [np.int64(3)],
        })
        assert out["a"] == [[0, 1], [2, 3]]
        assert out["b"] == 1.5
        assert out["c"] == [1.0, 2.0]
        assert out["d"] is None            # NaN is not valid JSON
        assert out["e"] is True and isinstance(out["e"], bool)
        assert out["f"] == [3]
        json.dumps(out)                    # the point of the exercise

    @pytest.mark.parametrize("suffix", [".json", ".yaml"])
    def test_write_record_roundtrips(self, tmp_path, suffix):
        rec = {"certificate": {"s": np.float64(2.5), "P": np.eye(2)}}
        path = tmp_path / f"rec{suffix}"
        bl.write_record(rec, path)
        loaded = (json.loads(path.read_text()) if suffix == ".json"
                  else yaml.safe_load(path.read_text()))
        assert loaded["certificate"]["s"] == 2.5
        assert loaded["certificate"]["P"] == [[1.0, 0.0], [0.0, 1.0]]

    def test_input_violations_counts_trajectories_not_steps(self):
        c = np.array([[-1.0, -2.0, -3.0],     # clean
                      [-1.0, 0.5, 2.0],       # violates twice -> counts once
                      [-1.0, -1.0, -1.0]])
        out = bl.input_violations(c)
        assert out == {"n_trajectories": 3, "n_violating": 1, "max_margin": 2.0}

    def test_input_violations_tolerates_missing_margins(self):
        assert bl.input_violations(None)["n_violating"] is None


class TestDivergedMask:
    """A single blown-up trajectory sends the pooled NRMSE to inf; the mask is
    what keeps the record interpretable when that happens."""

    def test_flags_non_finite_and_blown_up_trajectories(self):
        e = np.ones((4, 5, 1))
        e_hat = np.ones((4, 5, 1))
        e_hat[1, 3, 0] = np.inf          # non-finite
        e_hat[2, :, 0] = 50.0            # finite but 50x the target peak
        e_hat[3, :, 0] = 5.0             # large but under the 10x factor
        mask = bl.diverged_mask(e_hat, e, factor=10.0)
        assert mask.tolist() == [False, True, True, False]

    def test_nan_predictions_count_as_diverged(self):
        e = np.ones((2, 3, 1))
        e_hat = np.ones((2, 3, 1))
        e_hat[0, 1, 0] = np.nan
        assert bl.diverged_mask(e_hat, e, factor=10.0).tolist() == [True, False]

    def test_a_clean_split_flags_nothing(self):
        rng = np.random.default_rng(3)
        e = rng.standard_normal((6, 20, 1))
        assert not bl.diverged_mask(e + 1e-3, e, factor=10.0).any()


class TestNoiseFloor:
    """The floor turns NRMSE into a ratio: 1.0 means the residual IS the sensor
    noise and no model can do better on this data. It is only trustworthy if it
    recovers a known noise level, so that is what these check."""

    @staticmethod
    def _bandlimited_plus_noise(sigma, n=30, T=4000, fs=20.0, seed=11, f_hi=1.2):
        rng = np.random.default_rng(seed)
        t = np.arange(T) / fs
        e = np.empty((n, T, 1))
        for b in range(n):
            sig = sum(
                rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                for f in (0.16, 0.5, f_hi)
            )
            e[b, :, 0] = sig + rng.normal(0.0, sigma, T)
        return e

    def test_recovers_a_known_white_noise_level(self):
        e = self._bandlimited_plus_noise(sigma=0.004)
        out = bl.measured_noise_floor(e, fs=20.0, output_scale=1.0)
        assert out["sigma"] == pytest.approx(0.004, rel=0.05)
        assert out["flatness_ratio"] == pytest.approx(1.0, abs=0.15)
        assert out["n_estimates"] == 30

    def test_nrmse_is_sigma_over_the_metric_denominator(self):
        e = self._bandlimited_plus_noise(sigma=0.004)
        out = bl.measured_noise_floor(e, fs=20.0, output_scale=0.08)
        assert out["nrmse"] == pytest.approx(out["sigma"] / 0.08, rel=1e-12)

    def test_flatness_ratio_flags_signal_leaking_into_the_band(self):
        """Content near Nyquist breaks the white-noise assumption; the ratio is
        the guard that makes an overstated floor visible instead of silent."""
        clean = self._bandlimited_plus_noise(sigma=0.004, f_hi=9.0)
        assert bl.measured_noise_floor(clean, fs=20.0, output_scale=1.0)["flatness_ratio"] > 2.0

    def test_declared_matches_the_generators_formula(self):
        rng = np.random.default_rng(5)
        e = rng.standard_normal((7, 500, 1)) * np.array([0.2, 0.5, 1.0, 2.0, 0.3, 0.7, 1.5]
                                                        ).reshape(-1, 1, 1)
        out = bl.declared_noise_floor(e, snr_db=30.0, output_scale=1.0)
        expected = float(np.sqrt(np.mean((np.std(e, axis=1) / 10 ** 1.5) ** 2)))
        assert out["sigma"] == pytest.approx(expected, rel=1e-12)
        assert out["snr_db"] == 30.0

    def test_measured_and_declared_agree_on_a_generated_split(self):
        """The two estimators are independent — the PSD reads the noise off the
        data, the declared value comes from the SNR the generator was given."""
        sigma, n, T = 0.004, 30, 4000
        e = self._bandlimited_plus_noise(sigma=sigma, n=n, T=T)
        # the SNR that generator would have used for this exact noise level
        snr = 20 * np.log10(float(np.sqrt(np.mean(np.std(e, axis=1) ** 2))) / sigma)
        measured = bl.measured_noise_floor(e, fs=20.0, output_scale=1.0)["sigma"]
        declared = bl.declared_noise_floor(e, snr_db=snr, output_scale=1.0)["sigma"]
        assert measured == pytest.approx(declared, rel=0.05)

    def test_returns_none_when_trajectories_are_too_short(self):
        assert bl.measured_noise_floor(np.zeros((2, 8, 1)), fs=20.0, output_scale=1.0) is None


class TestCertificateRecordRho:
    """rho is the headline of the baseline, so it must survive the one case where
    post_process leaves it unset: an under-covering theta, whose coverage sweep
    finds y_max unreachable and returns None."""

    @staticmethod
    def _model():
        norm = _fitted_normalizer(1.0, 0.5)
        return bl.build_model(bl.to_normalized_units(_theta(), norm),
                              activation="dzn", ts=0.05, alpha=0.99, device="cpu")

    @staticmethod
    def _summary(rho):
        return {
            "y_max": 2.0, "constraints_satisfied": True,
            "max_s": {"y_bar": 1.0, "rho": rho, "coverage_ok": False,
                      "volume": 1.0, "norm_H": 0.0, "max_eig_F": -1e-6},
        }

    def test_derived_from_y_bar_when_the_sweep_returned_none(self):
        rec = bl.certificate_record(self._model(), self._summary(None), sigma_u=1.0)
        assert rec["rho"] == pytest.approx((1.0 / 2.0) ** 2)   # (y_bar/y_max)^nx

    def test_an_existing_rho_is_passed_through_untouched(self):
        rec = bl.certificate_record(self._model(), self._summary(1.234), sigma_u=1.0)
        assert rec["rho"] == 1.234


class TestCertificateUnits:
    """s and u_max live in NORMALIZED input units — u/sigma_u — because that is the
    port the model's B feeds. Reading them as physical numbers is the natural
    mistake (u_max = 99.7 looks absurd until you divide by sigma_u^2), so the
    record has to carry the physical value and the gauge-invariant set."""

    @staticmethod
    def _model(s: float = 4.0):
        m = bl.build_model(_theta(), activation="dzn", ts=0.05, alpha=0.99, device="cpu")
        with torch.no_grad():
            m.s.data = torch.tensor(s, dtype=m.P.dtype)
            m.P.data = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=m.P.dtype)
            m.set_input_bound(99.680360)
        return m

    @staticmethod
    def _summary():
        return {"y_max": 1.0, "constraints_satisfied": True,
                "max_s": {"y_bar": 1.0, "rho": 1.0, "coverage_ok": True,
                          "volume": 1.0, "norm_H": 0.0, "max_eig_F": -1e-6}}

    def test_u_max_physical_undoes_the_input_normalization(self):
        rec = bl.certificate_record(self._model(), self._summary(), sigma_u=0.110395)
        assert rec["u_max"] == pytest.approx(99.680360)
        assert rec["u_max_physical"] == pytest.approx(99.680360 * 0.110395 ** 2)
        assert rec["u_max_physical"] == pytest.approx(1.214818, rel=1e-4)

    def test_set_shape_is_x_over_s_squared(self):
        rec = bl.certificate_record(self._model(s=4.0), self._summary(), sigma_u=1.0)
        assert np.allclose(rec["set_shape"], np.array(rec["X"]) / 16.0)
        # X = {x : x' set_shape x <= 1}: the semi-axes follow from its eigenvalues
        assert np.allclose(rec["semi_axes"],
                           1.0 / np.sqrt(np.linalg.eigvalsh(np.array(rec["set_shape"]))))

    def test_the_units_block_names_the_input_scale(self):
        rec = bl.certificate_record(self._model(), self._summary(), sigma_u=0.110395)
        assert "0.110395" in rec["units"]["input"]
        assert "physical" in rec["units"]["state"]


@requires_mosek
class TestInputRescaling:
    """s = 10.55 in the record against s = 1.24 from the same .npz solved raw reads
    like a scaling bug. It is not: rescaling the input port is an exact
    re-parameterization of the certificate,

        (P, L, M, s) -> (P/su^2, L/su^2, M/su^2, su*s),

    which leaves the certified set alone. What it is NOT is numerically neutral —
    the LMIs' epsilon floors are absolute while P scales by su^2 — and that, not
    the algebra, is why the two solves disagree. These lock in both halves so a
    future eps change cannot quietly move the baseline.
    """

    SIGMA_U = 0.110395

    @staticmethod
    def _regional_theta():
        """_theta() is near-globally stable (B2 ~ 1e-2), so MaxS hits the locality
        epsilon ceiling at s ~ 1000 and there is nothing to compare. Positive
        feedback through q̇ makes the certificate genuinely regional."""
        th = _theta()
        th["B2"] = np.array([[0.0, 0.0], [0.2, 0.1]])
        return th

    def _synth(self, theta, eps=None):
        from dataclasses import replace
        from sysid.optimization import LureCertificateSynthesizer
        m = bl.build_model(theta, activation="dzn", ts=0.05, alpha=0.99, device="cpu")
        s = LureCertificateSynthesizer.from_model(m)
        return s if eps is None else replace(s, eps=eps)

    def _scaled(self, theta):
        return bl.to_normalized_units(theta, _fitted_normalizer(self.SIGMA_U, 1.0))

    def test_the_normalized_solution_transfers_to_a_valid_physical_certificate(self):
        """The map's whole point: the guarantee earned in normalized units is a
        genuine guarantee about the physical system."""
        theta = self._regional_theta()
        sol = self._synth(self._scaled(theta)).max_s()
        assert sol is not None
        c = 1.0 / self.SIGMA_U ** 2
        P, L, M, s = sol.P * c, sol.L * c, sol.M * c, self.SIGMA_U * sol.s

        A, B, B2, C2, D21 = (theta[k] for k in ("A", "B", "B2", "C2", "D21"))
        al, nx, nd = 0.99, 2, 1
        F = np.block([[-(al ** 2) * P, np.zeros((nx, nd)), P @ C2.T + L.T, P @ A.T],
                      [np.zeros((nd, nx)), -np.eye(nd), D21.T, B.T],
                      [C2 @ P + L, D21, -2 * M, M @ B2.T],
                      [A @ P, B, B2 @ M, -P]])
        assert np.max(np.linalg.eigvalsh((F + F.T) / 2)) < 0     # feasible, raw theta
        # and it certifies the very same set
        assert np.allclose(np.linalg.inv(P) / s ** 2,
                           np.linalg.inv(sol.P) / sol.s ** 2)

    def test_the_absolute_epsilon_is_what_makes_the_two_solves_disagree(self):
        """With eps left absolute the normalized solve falls short; scaling eps the
        way the map scales P recovers su*s_norm == s_phys."""
        theta = self._regional_theta()
        c = 1.0 / self.SIGMA_U ** 2
        eps = self._synth(theta).eps

        s_phys = self._synth(theta).max_s().s
        s_fixed_eps = self._synth(self._scaled(theta)).max_s().s
        s_scaled_eps = self._synth(self._scaled(theta), eps=eps / c).max_s().s

        assert self.SIGMA_U * s_fixed_eps < 0.95 * s_phys                    # falls short
        assert self.SIGMA_U * s_scaled_eps == pytest.approx(s_phys, rel=1e-3)  # recovered

    def test_s_physical_is_what_compares_against_a_raw_solve(self):
        """s itself is normalized; only sigma_u*s is on the same footing as an s
        obtained from the unscaled .npz."""
        theta = self._regional_theta()
        s_norm = self._synth(self._scaled(theta)).max_s().s
        s_phys = self._synth(theta).max_s().s
        assert s_norm > 5 * s_phys                                    # wildly different
        assert self.SIGMA_U * s_norm == pytest.approx(s_phys, rel=0.25)  # same ballpark

    def test_maxs_does_not_pin_p_so_set_shape_is_not_a_fingerprint(self):
        """At the optimal s there is a family of feasible P. Recording set_shape is
        useful, but it must not be read as a canonical property of theta."""
        import cvxpy as cp
        theta = self._regional_theta()
        synth = self._synth(theta)
        sol = synth.max_s()
        s = sol.s * 0.999                      # just inside the optimum
        A, B, B2, C2, D21 = (theta[k] for k in ("A", "B", "B2", "C2", "D21"))
        al, nx, nz, nd, eps = 0.99, 2, 2, 1, synth.eps

        P = cp.Variable((nx, nx), symmetric=True)
        L = cp.Variable((nz, nx))
        mv = cp.Variable((nz, 1))
        M = cp.diag(mv)
        F = cp.bmat([[-(al ** 2) * P, np.zeros((nx, nd)), P @ C2.T + L.T, P @ A.T],
                     [np.zeros((nd, nx)), -np.eye(nd), D21.T, B.T],
                     [C2 @ P + L, D21, -2 * M, M @ B2.T],
                     [A @ P, B, B2 @ M, -P]])
        cons = [mv >= eps, F << -eps * np.eye(F.shape[0])]
        for i in range(nz):
            li = L[i, :].reshape((1, -1), order="C")
            cons.append(cp.bmat([[np.array([[1 / s ** 2]]), li], [li.T, P]])
                        >> eps * np.eye(nx + 1))
        cp.Problem(cp.Maximize(cp.log_det(P)), cons).solve(solver=cp.MOSEK)

        maxs_shape = np.linalg.inv(sol.P) / sol.s ** 2
        other_shape = np.linalg.inv(P.value) / s ** 2
        assert not np.allclose(maxs_shape, other_shape, rtol=0.05)


# ------------------------------------------------------------------ end-to-end
def _write_dataset(root: Path, theta: dict, counts: dict, T: int = 40) -> None:
    """A tiny train/validation/test[_div] dataset generated by ``theta`` itself.

    ``x0 = 0`` on every trajectory: the evaluator always rolls out from zeros, so
    a nonzero recorded ``x0`` would show up as a transient mismatch and hide a
    real regression behind it. The drive is large enough that the dead zone
    fires — a silently linear smoke test would not exercise the model class.
    """
    rng = np.random.default_rng(7)
    for split, n in counts.items():
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            length = T // 2 if split.endswith("_div") else T
            u = 8.0 * rng.standard_normal(length)
            x, rows = np.zeros(2), []
            for k in range(length):
                uk = np.atleast_1d(u[k])
                w = _dzn(theta["C2"] @ x + theta["D21"] @ uk)
                rows.append((u[k], x[0], x[1]))
                x = theta["A"] @ x + theta["B"] @ uk + theta["B2"] @ w
            with open(folder / f"traj_{i:03d}.csv", "w") as f:
                f.write("u,q,q_dot\n")
                for r in rows:
                    f.write(f"{r[0]:.8f},{r[1]:.8f},{r[2]:.8f}\n")


def _run_main(argv: list) -> None:
    old_argv, old_dtype = sys.argv, torch.get_default_dtype()
    try:
        sys.argv = ["baseline_lure.py", *argv]
        bl.main()
    finally:
        sys.argv = old_argv
        torch.set_default_dtype(old_dtype)   # main() sets it globally


def _make_case(tmp: Path, T: int = 40) -> tuple:
    theta = _theta()
    data_root = tmp / "data"
    _write_dataset(data_root, theta, {
        "train": 4, "validation": 2, "test": 2,
        "train_div": 2, "validation_div": 1, "test_div": 1,
    }, T=T)
    npz_path = tmp / "lure_model_params.npz"
    np.savez(npz_path, TS=0.05, **theta)
    # the generator's sidecar: its SNR is the noise floor's cross-check
    (data_root / "params.json").write_text(json.dumps({"generation": {"SNR_dB": 30.0}}))
    return npz_path, data_root


@requires_mosek
class TestEndToEnd:
    """One full run: npz + dataset in, one record out."""

    @pytest.fixture(scope="class")
    def record(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("baseline")
        npz_path, data_root = _make_case(tmp)
        out_path = tmp / "baseline.json"
        _run_main(["--npz", str(npz_path), "--data", str(data_root),
                   "--output", str(out_path), "--warmup-steps", "5"])
        return json.loads(out_path.read_text())

    def test_runs_from_npz_and_data_alone(self, tmp_path):
        """--npz and --data are the whole contract; no config, no other flags.

        The record must land next to the dataset, named after the .npz, and the
        default 500-step warmup must still leave metrics to report."""
        npz_path, data_root = _make_case(tmp_path, T=600)
        _run_main(["--npz", str(npz_path), "--data", str(data_root)])
        default_out = data_root / "baseline_lure_model_params.json"
        assert default_out.exists()
        rec = json.loads(default_out.read_text())
        assert rec["protocol"]["warmup_steps"] == 500
        assert rec["metrics"]["test"]["metrics"]["nrmse"] < 1e-6

    def test_all_six_splits_are_reported(self, record):
        assert set(record["metrics"]) == {
            "train", "validation", "test", "train_div", "validation_div", "test_div"
        }
        for split in ("train", "validation", "test"):
            assert record["metrics"][split]["metrics"]["nrmse"] >= 0.0

    def test_the_fit_is_near_exact_on_self_generated_data(self, record):
        """The data came from this very theta, so the baseline must nail it."""
        assert record["metrics"]["train"]["metrics"]["nrmse"] < 1e-6
        assert record["metrics"]["test"]["metrics"]["nrmse"] < 1e-6

    def test_the_nonlinearity_actually_fires(self, record):
        """Guards the fixture: a dataset that stays in the dead band would make
        every assertion above pass on an effectively LTI model."""
        assert record["deadzone_activity"]["firing_rate"] > 0.0
        assert record["deadzone_activity"]["max_abs_z"] > 1.0

    def test_certificate_matrices_are_present_and_consistent(self, record):
        cert = record["certificate"]
        P = np.array(cert["P"])
        X = np.array(cert["X"])
        L = np.array(cert["L"])
        H = np.array(cert["H"])
        assert np.allclose(P @ X, np.eye(2), atol=1e-8)      # X = P^-1
        assert np.allclose(H, L @ X, atol=1e-8)              # H = L P^-1
        assert np.allclose(P, P.T, atol=1e-10)
        assert np.min(np.linalg.eigvalsh(P)) > 0
        assert cert["max_eig_F"] < 0                          # the LMI is strict
        assert cert["constraints_satisfied"] is True

    def test_reported_y_bar_and_rho_match_their_definitions(self, record):
        """y_bar = sigma*s*sqrt(C P C') and rho = (y_bar/y_max)^nx, not free-floating."""
        cert, C = record["certificate"], np.array(record["theta_normalized"]["C"])
        P, s = np.array(cert["P"]), cert["s"]
        sigma = record["normalization"]["output_std"][0]
        y_bar = sigma * s * float(np.sqrt((C @ P @ C.T).item()))
        assert cert["y_bar"] == pytest.approx(y_bar, rel=1e-6)
        assert cert["rho"] == pytest.approx((y_bar / cert["y_max"]) ** 2, rel=1e-6)

    def test_the_input_floor_is_recorded_and_respected(self, record):
        """s >= max_k ||u_k|| is necessary whatever P is; the record must say so."""
        cert = record["certificate"]
        assert cert["s_floor"] == pytest.approx(np.sqrt(record["levels"]["u_max"]))
        assert cert["s_meets_input_floor"] == (cert["s"] >= cert["s_floor"])

    def test_theta_normalized_is_theta_physical_rescaled(self, record):
        u_std = record["normalization"]["input_std"][0]
        y_std = record["normalization"]["output_std"][0]
        phys, norm = record["theta_physical"], record["theta_normalized"]
        assert np.allclose(norm["A"], phys["A"])
        assert np.allclose(norm["B"], np.array(phys["B"]) * u_std)
        assert np.allclose(norm["C"], np.array(phys["C"]) / y_std)

    def test_every_converging_split_carries_its_noise_floor(self, record):
        """Including the declared cross-check, since params.json is present."""
        for split in ("train", "validation", "test"):
            floor = record["metrics"][split]["noise_floor"]
            assert floor["sigma"] >= 0.0
            assert floor["nrmse"] >= 0.0
            assert floor["declared"]["snr_db"] == 30.0
            assert "nrmse_ratio" in floor
        # diverging targets are not band-limited, so no floor is claimed for them
        assert "noise_floor" not in record["metrics"]["test_div"]

    def test_protocol_is_recorded(self, record):
        """A baseline is only comparable if the protocol that produced it is stated."""
        proto = record["protocol"]
        assert proto["warmup_steps"] == 5
        assert proto["dtype"] == "float64"
        assert proto["output_scale"] > 0
