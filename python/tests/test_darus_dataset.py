"""Tests for the DaRUS publication path of the regionally stable datasets.

Two scripts are pinned here:

1. ``scripts/generate_duffing_dataset.py`` — the standalone Duffing generator
   that replaced the notebook as the source of truth. Its point is to reproduce
   the dataset written by ``notebooks/duffing/duffing_benchmark.ipynb`` at
   045a667 byte for byte, which only holds while the RNG draws happen in the
   same order. The pin test regenerates the first published file and skips when
   the data folder is absent.
2. ``scripts/prepare_darus_dataset.py`` — the packager. The Dataverse JSON is
   checked for the fields DaRUS refuses a dataset without, and ``--verify`` is
   checked to pass on a faithful copy and fail on a tampered one, since the
   README's reproducibility claim rests on it.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"
PUBLISHED_DUFFING = Path("~/genSecSysId-Data/data/Duffing/id").expanduser()


def _load_script_module(name: str, path: Path):
    """Import a scripts/ module by file path without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen_duffing = _load_script_module("generate_duffing_dataset",
                                  SCRIPTS / "generate_duffing_dataset.py")
darus = _load_script_module("prepare_darus_dataset", SCRIPTS / "prepare_darus_dataset.py")


def _small_duffing(out_dir: Path, *extra: str) -> Path:
    gen_duffing.main(["--out-dir", str(out_dir), "--n-conv", "2", "--n-div", "1",
                      "--t-traj", "400", *extra])
    return out_dir


# --------------------------------------------------------------------------
# Duffing generator
# --------------------------------------------------------------------------
def test_duffing_generator_writes_loader_layout(tmp_path):
    out = _small_duffing(tmp_path / "id")
    assert sorted(p.name for p in (out / "raw").iterdir()) == [
        "rand_conv_000.csv", "rand_conv_001.csv", "zero_div_000.csv"]
    for split in darus.SPLIT_FOLDERS:
        assert (out / split).is_dir()

    conv = [pd.read_csv(f) for f in sorted((out / "raw").glob("rand_conv_*.csv"))]
    div = pd.read_csv(out / "raw" / "zero_div_000.csv")
    assert all(list(df.columns) == ["u", "q", "q_dot"] for df in conv + [div])
    # Converging files span the full horizon (the loader np.stack-s them); the
    # diverging one is cut before the threshold crossing.
    assert all(len(df) == 400 for df in conv)
    assert 0 < len(div) < 400
    assert np.abs(div[["q", "q_dot"]].values).max() <= gen_duffing.DIVERGE_THRESH + 0.5

    params = json.loads((out / "params.json").read_text())
    assert params["generation"]["SNR_dB"] == 30.0
    assert params["generation"]["N_rand_conv"] == 2


def test_duffing_generator_is_deterministic(tmp_path):
    a = _small_duffing(tmp_path / "a")
    b = _small_duffing(tmp_path / "b")
    for f in (a / "raw").iterdir():
        assert f.read_bytes() == (b / "raw" / f.name).read_bytes()


def test_duffing_noise_can_be_disabled(tmp_path):
    """Without noise the zero-start trajectory starts at exactly the origin."""
    out = _small_duffing(tmp_path / "id", "--snr-db", "-1")
    first = pd.read_csv(out / "raw" / "zero_div_000.csv").iloc[0]
    assert first["q"] == 0.0 and first["q_dot"] == 0.0
    noisy = pd.read_csv(_small_duffing(tmp_path / "noisy") / "raw" / "zero_div_000.csv")
    assert noisy.iloc[0]["q"] != 0.0


@pytest.mark.skipif(not PUBLISHED_DUFFING.exists(), reason="published Duffing data not present")
def test_duffing_generator_reproduces_published_file(tmp_path):
    """zero_div is drawn first, so its first file pins seed, input, solver and noise."""
    gen_duffing.main(["--out-dir", str(tmp_path), "--n-conv", "0", "--n-div", "1"])
    name = "raw/zero_div_000.csv"
    assert (tmp_path / name).read_bytes() == (PUBLISHED_DUFFING / name).read_bytes()


# --------------------------------------------------------------------------
# DaRUS packager
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def one_d_data(tmp_path_factory):
    """A full OneD dataset from the shipped generator (about a second)."""
    out = tmp_path_factory.mktemp("one_d") / "id"
    subprocess.run([sys.executable, str(SCRIPTS / "generate_one_d_dataset.py"),
                    "--out-dir", str(out)], check=True, stdout=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def one_d_package(one_d_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("darus")
    return darus.build("one_d", one_d_data, out, "someone@example.org", verify=True)


def _fields(block):
    return {f["typeName"]: f for f in block["fields"]}


def _check_field(field):
    """Dataverse rejects a field whose ``multiple`` and value shape disagree."""
    assert field["typeClass"] in ("primitive", "controlledVocabulary", "compound")
    if field["multiple"]:
        assert isinstance(field["value"], list) and field["value"]
    else:
        assert not isinstance(field["value"], list)
    if field["typeClass"] == "compound":
        for entry in field["value"]:
            for name, sub in entry.items():
                assert sub["typeName"] == name
                _check_field(sub)


def test_dataset_json_has_darus_required_fields(one_d_package):
    ds = json.loads((one_d_package / "dataset.json").read_text())["datasetVersion"]
    assert ds["license"]["name"] == "CC BY 4.0"
    citation = _fields(ds["metadataBlocks"]["citation"])
    for name in ["title", "author", "datasetContact", "dsDescription", "subject", "producer"]:
        assert name in citation, name
    assert citation["author"]["value"][0]["authorName"]["value"] == "Frank, Daniel"
    contact = citation["datasetContact"]["value"][0]
    assert contact["datasetContactEmail"]["value"] == "someone@example.org"
    assert citation["producer"]["value"][0]["producerName"]["value"]
    assert "<h2>" in citation["dsDescription"]["value"][0]["dsDescriptionValue"]["value"]
    for block in ds["metadataBlocks"].values():
        for field in block["fields"]:
            _check_field(field)


def test_process_parameters_carry_a_value(one_d_package):
    ds = json.loads((one_d_package / "dataset.json").read_text())["datasetVersion"]
    process = _fields(ds["metadataBlocks"]["process"])
    pars = process["processMethodsPar"]["value"]
    assert len(pars) == len(darus.ONE_D_PARS)
    for p in pars:
        assert ("processMethodsParValue" in p) != ("processMethodsParTextValue" in p)
    by_name = {p["processMethodsParName"]["value"]: p for p in pars}
    assert float(by_name["State matrix"]["processMethodsParValue"]["value"]) == 0.9
    steps = [s["processStepType"]["value"] for s in process["processStep"]["value"]]
    assert steps == ["Generation", "Postprocessing"]


def test_zip_is_self_contained_and_checksummed(one_d_package, one_d_data):
    with zipfile.ZipFile(one_d_package / "OneD.zip") as zf:
        names = set(zf.namelist())
        for required in ["OneD/README.md", "OneD/SHA256SUMS", "OneD/id/params.json",
                         "OneD/code/generate_one_d_dataset.py", "OneD/code/requirements.txt"]:
            assert required in names, required
        # every CSV of every folder, and nothing from outside it
        n_csv = sum(1 for n in names if n.endswith(".csv"))
        assert n_csv == sum(1 for _ in one_d_data.rglob("*.csv"))
        sums = zf.read("OneD/SHA256SUMS").decode().splitlines()
        assert len(sums) == len(names) - 1  # all files but SHA256SUMS itself
        for line in sums:
            digest, rel = line.split("  ", 1)
            assert hashlib.sha256(zf.read(f"OneD/{rel}")).hexdigest() == digest


def test_readme_claims_reproducibility_only_when_verified(one_d_package, one_d_data, tmp_path):
    assert "byte for byte" in (one_d_package / "README.md").read_text()
    unverified = darus.build("one_d", one_d_data, tmp_path, None, verify=False)
    assert "byte for byte" not in (unverified / "README.md").read_text()


def test_verify_fails_on_tampered_data(one_d_data, tmp_path):
    import shutil

    tampered = tmp_path / "id"
    shutil.copytree(one_d_data, tampered)
    victim = tampered / "test" / sorted(p.name for p in (tampered / "test").iterdir())[0]
    victim.write_text(victim.read_text().replace("0", "1", 1))
    with pytest.raises(SystemExit, match="differ after regeneration"):
        darus.build("one_d", tampered, tmp_path / "out", None, verify=True)
