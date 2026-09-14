"""The MLflow file-store maintenance-mode opt-out.

From MLflow 3.16 a file-backed tracking URI (``./mlruns``, i.e.
``mlflow.tracking_uri: null``) raises ``MlflowException`` unless
``MLFLOW_ALLOW_FILE_STORE`` is set -- the file store is in maintenance mode and
the project pushes users toward a database backend.

Every run this project has produced lives in that file store, and several tools
read it directly (``scripts/plot_sigma_tradeoff.py``, ``reporting/runs.py``), so
the opt-out is what keeps the existing history usable. It has to be applied at
*every* entry point that touches MLflow, because a single missed one fails the
whole pipeline at the point where it logs -- i.e. after training has already run.

These tests are about the wiring, not about MLflow's behaviour, so they neither
need a solver nor a tracking server.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sysid.config import allow_file_store

REPO = Path(__file__).resolve().parent.parent


class TestHelper:
    def test_sets_the_flag_when_unset(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
        allow_file_store()
        assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"

    def test_does_not_override_an_explicit_choice(self, monkeypatch):
        """A deliberate migration to sqlite sets this to false. Clobbering it
        would silently drag the user back onto the deprecated backend."""
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
        allow_file_store()
        assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"

    def test_is_idempotent(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
        allow_file_store()
        allow_file_store()
        assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


class TestEveryEntryPointIsCovered:
    """A missed entry point fails only at log time -- after the expensive part.

    Checked by source inspection rather than by running each script: the point
    is that the call is present, and running them needs data, a solver and
    minutes apiece.
    """

    @pytest.mark.parametrize("path", [
        "scripts/train.py",          # does its own MLflow setup, not setup_mlflow_tracking
        "scripts/sweep.py",          # pre-creates the run before train.py is invoked
        "src/sysid/config.py",       # setup_mlflow_tracking -> evaluate/post_process/compare
        "src/sysid/reporting/runs.py",
    ])
    def test_calls_the_opt_out(self, path):
        src = (REPO / path).read_text()
        assert "allow_file_store()" in src, (
            f"{path} touches MLflow but never calls allow_file_store(); a "
            "file-backed tracking URI will raise there on MLflow >= 3.16"
        )

    def test_setup_mlflow_tracking_applies_it(self):
        """The shared path used by evaluate.py, post_process.py and compare.py."""
        src = (REPO / "src/sysid/config.py").read_text()
        body = src[src.index("def setup_mlflow_tracking("):]
        body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
        assert "allow_file_store()" in body


class TestItActuallyUnblocksTheClient:
    def test_file_store_client_works_in_a_clean_environment(self, tmp_path):
        """End to end in a subprocess with the variable unset, which is the
        state a fresh shell is in."""
        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_ALLOW_FILE_STORE"}
        env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        code = (
            "from sysid.config import allow_file_store; allow_file_store()\n"
            "import mlflow\n"
            f"mlflow.set_tracking_uri('file://{tmp_path}/mlruns')\n"
            "mlflow.set_experiment('optout-probe')\n"
            "import mlflow as m\n"
            "with m.start_run():\n"
            "    m.log_metric('x', 1.0)\n"
            "print('OK')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        assert "OK" in r.stdout
        assert "maintenance mode" not in r.stderr
