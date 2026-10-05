"""Package the regionally stable benchmark datasets for publication on DaRUS.

DaRUS (https://darus.uni-stuttgart.de) is a Dataverse installation. For each
system this writes, under ``<out-dir>/<Name>/``::

    <Name>.zip        <Name>/README.md, id/ (params.json, raw/, split folders),
                      code/ (generator + requirements.txt), SHA256SUMS
    README.md         the same README, for reading without unpacking
    description.html  the README as HTML, for the web form's Description field
    dataset.json      Dataverse native-API JSON (citation + process metadata,
                      CC BY 4.0), ready for ``POST /api/dataverses/<alias>/datasets``

Only the trajectory CSVs and ``params.json`` are published; other files in the
data folder (baseline fits, Lur'e approximations) are left out.

``--verify`` regenerates the dataset with the bundled generator into a temporary
folder and fails unless every published file is byte-identical. The README only
claims byte-for-byte reproducibility when that check ran.

Dataverse unpacks uploaded zips. To keep ``<Name>.zip`` as one file, as in
doi:10.18419/DARUS-4770, upload it zipped once more.

Usage::

    python scripts/prepare_darus_dataset.py --system all --verify \\
        --contact-email first.last@ki.uni-stuttgart.de
"""

import argparse
import datetime as dt
import filecmp
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy

REPO_PY = Path(__file__).resolve().parents[1]
GITHUB_URL = "https://github.com/Dany-L/genSecSysId"
SPLIT_FOLDERS = ["train", "validation", "test", "train_div", "validation_div", "test_div"]
GROUPS = {"rand_conv": "converging", "zero_div": "diverging"}

AUTHOR = {
    "name": "Frank, Daniel",
    "affiliation": "https://ror.org/04vnq7t77",  # University of Stuttgart
    "orcid": "https://orcid.org/0000-0002-6730-2252",
}
AFFILIATION_NAME = "University of Stuttgart"
GRANT = {"agency": "DFG", "value": "EXC 2075 - 390740016"}
DATAVERSE_ALIAS = "ki_ac_InMotion"

COMMON_KEYWORDS = [
    ("Nonlinear system identification", "http://www.wikidata.org/entity/Q17080460"),
    ("Dynamical system", "http://www.wikidata.org/entity/Q638328"),
    ("Stability theory", "http://www.wikidata.org/entity/Q1756677"),
    ("Recurrent neural network", "http://www.wikidata.org/entity/Q1457734"),
    ("Regional stability", None),
    ("Region of attraction", None),
]
TOPICS = [
    ("Artificial Intelligence and Machine Learning Methods", "443-04"),
    ("Automation, Control Systems, Robotics, Mechatronics, Cyber Physical Systems", "407-01"),
]

# (name, symbol, unit, value) -- a float value goes to the FLOAT field, anything
# else to the textual one.
DUFFING_PARS = [
    ("Damping coefficient", "delta_d", "-", 0.3),
    ("Sampling time", "T_s", "s", 0.05),
    ("ODE solver", "", "", "scipy.integrate.solve_ivp, RK45, rtol=1e-5, atol=1e-7, "
                          "one call per sample, zero-order hold on u"),
    ("Trajectory length", "T", "samples", 4000.0),
    ("Maximum input amplitude", "u_max", "-", 3.5),
    ("Low-pass cutoff frequency", "f_c", "Hz", 2.0),
    ("Low-pass filter order (Butterworth, zero phase)", "", "-", 4.0),
    ("Envelope decay rate", "gamma", "-", 0.9),
    ("Initial state, converging trajectories", "x_0", "-", "U(-0.6, 0.6) per component"),
    ("Initial state, diverging trajectories", "x_0", "-", "(0, 0)"),
    ("Divergence threshold on |q|, |q_dot|", "x_thr", "-", 5.0),
    ("Number of converging trajectories", "N_conv", "-", 100.0),
    ("Number of diverging trajectories", "N_div", "-", 50.0),
    ("Signal-to-noise ratio of the state measurement", "SNR", "dB", 30.0),
    ("Random seed, generation", "", "-", 42.0),
    ("Random seed, split", "", "-", 42.0),
    ("Split ratio train/validation/test", "", "%", "60/10/30, per group"),
]
ONE_D_PARS = [
    ("State matrix", "A", "-", 0.9),
    ("Input matrix", "B", "-", 1.0),
    ("Nonlinearity input matrix", "B_2", "-", 1.1),
    ("Output matrix", "C", "-", 1.0),
    ("Nonlinearity output matrix", "C_2", "-", 1.0),
    ("Feedthrough matrices", "D, D_12, D_21", "-", 0.0),
    ("Nonlinearity", "dzn", "", "dzn(z) = max(|z| - 1, 0) sign(z)"),
    ("Sampling time", "T_s", "-", 1.0),
    ("Trajectory length", "T", "samples", 500.0),
    ("Maximum input amplitude, converging trajectories", "u_max", "-", 0.04),
    ("Maximum input amplitude, diverging trajectories", "u_max", "-", 0.6),
    ("Low-pass cutoff frequency", "f_c", "1/T_s", 0.1),
    ("Low-pass filter order (Butterworth, zero phase)", "", "-", 4.0),
    ("Envelope decay rate", "gamma", "-", 0.9),
    ("Initial state, converging trajectories", "x_0", "-", "U(-0.3, 0.3)"),
    ("Initial state, diverging trajectories", "x_0", "-", 0.0),
    ("Divergence threshold on |x|", "x_thr", "-", 5.0),
    ("Minimum length of a diverging trajectory", "", "samples", 20.0),
    ("Number of converging trajectories", "N_conv", "-", 100.0),
    ("Number of diverging trajectories", "N_div", "-", 50.0),
    ("Measurement noise", "", "", "none"),
    ("Random seed, generation", "", "-", 42.0),
    ("Random seed, split", "", "-", 42.0),
    ("Split ratio train/validation/test", "", "%", "60/10/30, per group"),
]

SYSTEMS = {
    "duffing": {
        "name": "Duffing",
        "data_dir": "~/genSecSysId-Data/data/Duffing/id",
        "generator": "scripts/generate_duffing_dataset.py",
        "title": "Duffing oscillator for nonlinear system identification - regionally stable, "
                 "with converging and diverging trajectories - synthetically generated",
        "keywords": [("Duffing equation", "http://www.wikidata.org/entity/Q675387")],
        "pars": DUFFING_PARS,
        "method": (
            "Simulation of the softening Duffing oscillator",
            "q'' = -delta_d q' - q + q^3 + u, sampled at T_s = 0.05 s with a zero-order hold "
            "on u. Input/initial-state pairs are rejection-sampled and labelled converging or "
            "diverging by a threshold on the noise-free state. White Gaussian noise at 30 dB "
            "SNR is then added to q and q_dot; u is noise free.",
        ),
    },
    "one_d": {
        "name": "OneD",
        "data_dir": "~/genSecSysId-Data/data/OneD/id",
        "generator": "scripts/generate_one_d_dataset.py",
        "title": "Scalar Lur'e system with dead-zone nonlinearity for nonlinear system "
                 "identification - regionally stable, with converging and diverging "
                 "trajectories - synthetically generated",
        "keywords": [("Lur'e system", None), ("Dead zone", None)],
        "pars": ONE_D_PARS,
        "method": (
            "Simulation of a scalar discrete-time Lur'e system",
            "x_{k+1} = 0.9 x_k + u_k + 1.1 dzn(x_k), y_k = x_k with the dead zone "
            "dzn(z) = max(|z| - 1, 0) sign(z). Input/initial-state pairs are "
            "rejection-sampled and labelled converging or diverging by a threshold on the "
            "state. No measurement noise.",
        ),
    },
}


# --- statistics -------------------------------------------------------------


def _n_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f) - 1  # minus the header


def folder_stats(folder: Path) -> Dict:
    lens = [_n_rows(f) for f in sorted(folder.glob("*.csv"))]
    if not lens:
        return {"n": 0, "min": 0, "median": 0, "max": 0, "samples": 0}
    return {"n": len(lens), "min": min(lens), "median": int(np.median(lens)),
            "max": max(lens), "samples": sum(lens)}


def column_stats(raw_dir: Path, prefix: str) -> Dict[str, Dict[str, float]]:
    df = pd.concat([pd.read_csv(f) for f in sorted(raw_dir.glob(f"{prefix}_*.csv"))])
    return {c: {"min": df[c].min(), "max": df[c].max(), "std": df[c].std(ddof=0)}
            for c in df.columns}


def collect_stats(data_dir: Path) -> Dict:
    folders = {name: folder_stats(data_dir / name) for name in ["raw"] + SPLIT_FOLDERS}
    columns = {g: column_stats(data_dir / "raw", g) for g in GROUPS}
    return {"folders": folders, "columns": columns}


# --- README -------------------------------------------------------------------


def _folder_table(stats: Dict) -> str:
    rows = ["| Folder | Content | Trajectories | Length (min / median / max) | Samples |",
            "|---|---|---:|---:|---:|"]
    content = {"raw": "all trajectories", "train": "converging", "validation": "converging",
               "test": "converging", "train_div": "diverging", "validation_div": "diverging",
               "test_div": "diverging"}
    for name, s in stats["folders"].items():
        rows.append(f"| `{name}/` | {content[name]} | {s['n']} | "
                    f"{s['min']} / {s['median']} / {s['max']} | {s['samples']:,} |")
    return "\n".join(rows)


def _column_table(stats: Dict) -> str:
    rows = ["| Group | Column | min | max | std |", "|---|---|---:|---:|---:|"]
    for g, cols in stats["columns"].items():
        for c, s in cols.items():
            rows.append(f"| `{g}` | `{c}` | {s['min']:.4g} | {s['max']:.4g} | {s['std']:.4g} |")
    return "\n".join(rows)


def _env_line(env: Dict) -> str:
    return f"Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}"


DUFFING_README = """\
# {title}

## Overview

Input-state data of a damped, softening Duffing oscillator. The system is
**regionally but not globally stable**: the origin is asymptotically stable, but
the saddle points (±1, 0) bound its basin of attraction, and inputs that push the
state across that boundary make it diverge. The dataset therefore has two parts:
*converging* trajectories that stay bounded over the whole horizon, and
*diverging* trajectories that start at rest and are driven out of the basin.

The data is intended for identifying models with regional stability guarantees.
The converging trajectories are used for fitting and evaluation, the diverging
ones carry the information where the stable region ends. Use the training sets
to fit model parameters, the validation sets for hyperparameter selection, and
the test sets only for the final evaluation.

## System

```
q'' = -delta_d q' - q + q^3 + u,     delta_d = 0.3,     x = (q, q')
```

For u = 0 the origin is a stable focus (eigenvalues -0.15 ± 0.989i) and (±1, 0)
are saddles (eigenvalues 1.27 and -1.57). The equation is non-dimensional; time
is in seconds through the sampling time.

The system is sampled at T_s = 0.05 s with a zero-order hold on u: each sample is
one call to `scipy.integrate.solve_ivp` (RK45, rtol = 1e-5, atol = 1e-7) over one
sampling period.

## Input generation

Each input trajectory is white Gaussian noise at 1/T_s = 20 Hz, filtered with a
zero-phase 4th-order Butterworth low-pass (cutoff 2 Hz, `scipy.signal.filtfilt`,
the first 16 samples discarded). It is then multiplied by the envelope
exp(-0.9 t), with t running from 0 to 1 over the trajectory, and scaled so that
max|u| = a, with a ~ U(0, 3.5) drawn per trajectory.

## Initial states and labelling

Trajectories are rejection-sampled. The label is decided on the noise-free state
with the divergence threshold 5:

* `rand_conv_NNN.csv`: x_0 ~ U(-0.6, 0.6) per component, kept if |q| ≤ 5 and
  |q'| ≤ 5 for all 4000 samples (200 s). 100 kept out of 296 draws.
* `zero_div_NNN.csv`: x_0 = (0, 0), kept if a state component exceeds 5 within
  4000 samples. The trajectory ends with the last sample before the threshold is
  crossed, so these files are ragged. 50 kept out of 81 draws.

## Noise

White Gaussian noise is added to q and q' (not to u), independently per
trajectory and channel, with standard deviation std(x_i) / 10^(30/20), i.e. an
SNR of 30 dB. It is added to every stored sample, k = 0 included, so the
`zero_div` files do not start at exactly zero.

## Folders and statistics

{folder_table}

The split is a random permutation (seed 42), 60/10/30 per group. The split
folders contain copies of the files in `raw/`.

Value ranges over `raw/`:

{column_table}

## File format

One CSV per trajectory, comma separated, header `u,q,q_dot`, values rounded to 8
decimals. Row k holds the input u_k and the measured state x_k at time
t_k = k T_s; x_(k+1) is the state after holding u_k for one sampling period.
`id/params.json` holds the generation parameters in machine-readable form.

In the accompanying code the input is `u`, the output `q`, and `(q, q_dot)` the
state used for initial conditions.

## Reproduction

{reproduction}

```
pip install -r code/requirements.txt
python code/generate_duffing_dataset.py --out-dir regenerated/id
```

The generator is also part of the genSecSysId repository ({github}), commit
`{commit}`.

## License

CC BY 4.0
"""

ONE_D_README = """\
# {title}

## Overview

Input-state data of a scalar discrete-time Lur'e system with a dead-zone
nonlinearity. The system is **regionally but not globally stable**: inside the
dead zone it is a stable linear system, outside it is unstable. The dataset has
two parts: *converging* trajectories that stay bounded over the whole horizon,
and *diverging* trajectories that start at zero and are driven out of the
stable region.

The system is small enough that its stable region is known in closed form,
which makes it a testbed for identification methods with regional stability
guarantees. Use the training sets to fit model parameters,
the validation sets for hyperparameter selection, and the test sets only for the
final evaluation.

## System

```
x_(k+1) = A x_k + B u_k + B2 dzn(C2 x_k + D21 u_k)
y_k     = C x_k + D u_k + D12 dzn(C2 x_k + D21 u_k)

A = 0.9, B = 1.0, B2 = 1.1, C = C2 = 1.0, D = D12 = D21 = 0
dzn(z) = max(|z| - 1, 0) sign(z)
```

For |x| ≤ 1 the map is x+ = 0.9 x + u. For |x| > 1 it is x+ = 2.0 x ∓ 1.1 + u,
which is unstable, so for u = 0 the basin of attraction of the origin is
|x| < 1.1. The sampling time is T_s = 1 (dimensionless) and y = x.

For reference, `id/params.json` records a regional stability certificate of the
true system: at contraction rate alpha = 0.99 the certified invariant set is
|x| ≤ 1.0, and inputs with amplitude up to sigma* = 0.0588 keep the state in it.
This set coincides with the linear region. The converging trajectories start
inside it and their inputs stay below sigma*, so they never excite the
nonlinearity (max|x| = {max_x_conv:.3f}). Only the diverging trajectories do.

## Input generation

Each input trajectory is white Gaussian noise filtered with a zero-phase
4th-order Butterworth low-pass (cutoff 0.1 / T_s, `scipy.signal.filtfilt`, the
first 16 samples discarded). It is then multiplied by the envelope exp(-0.9 t),
with t running from 0 to 1 over the trajectory, and scaled so that max|u| = a,
with a ~ U(0, a_max) drawn per trajectory: a_max = 0.04 (≈ 0.68 sigma*) for the
converging and a_max = 0.6 for the diverging group.

## Initial states and labelling

Trajectories are rejection-sampled with the divergence threshold |x| > 5:

* `rand_conv_NNN.csv`: x_0 ~ U(-0.3, 0.3), kept if |x| ≤ 5 for all 500 samples.
* `zero_div_NNN.csv`: x_0 = 0, kept if |x| exceeds 5 within 500 samples and the
  trajectory is at least 20 samples long. The trajectory ends with the last
  sample before the threshold is crossed, so these files are ragged.

## Noise

None. The data is noise free.

## Folders and statistics

{folder_table}

The split is a random permutation (seed 42), 60/10/30 per group. The split
folders contain copies of the files in `raw/`.

Value ranges over `raw/`:

{column_table}

## File format

One CSV per trajectory, comma separated, header `u,x`, values rounded to 8
decimals. Row k holds the input u_k and the state x_k = y_k; x_(k+1) is the
state after applying u_k. `id/params.json` holds the system, the generation
parameters and the reference certificate in machine-readable form.

In the accompanying code the input is `u`, the output `x`, and `x` the state
used for initial conditions.

## Reproduction

{reproduction}

```
pip install -r code/requirements.txt
python code/generate_one_d_dataset.py --out-dir regenerated/id
```

The generator is also part of the genSecSysId repository ({github}), commit
`{commit}`.

## License

CC BY 4.0
"""


def render_readme(system: str, stats: Dict, env: Dict, commit: str, verified: bool,
                  params: Dict) -> str:
    cfg = SYSTEMS[system]
    if verified:
        reproduction = ("`code/` contains the generator. With " + _env_line(env) +
                        " it reproduces every file in `id/` byte for byte:")
    else:
        reproduction = ("`code/` contains the generator. The data was generated with "
                        + _env_line(env) + ":")
    fields = dict(title=cfg["title"], folder_table=_folder_table(stats),
                  column_table=_column_table(stats), reproduction=reproduction,
                  github=GITHUB_URL, commit=commit)
    if system == "duffing":
        return DUFFING_README.format(**fields)
    return ONE_D_README.format(max_x_conv=params["measured"]["max_abs_x_converging"],
                               **fields)


def markdown_to_html(md: str) -> str:
    """README body without the title, as HTML for the Dataverse description."""
    import markdown  # only needed here; part of the "dev" extra in setup.py

    body = md.split("\n", 1)[1] if md.startswith("# ") else md
    return markdown.markdown(body, extensions=["tables", "fenced_code"])


# --- Dataverse JSON -----------------------------------------------------------


def _prim(name: str, value, multiple: bool = False) -> Dict:
    return {"typeName": name, "multiple": multiple, "typeClass": "primitive", "value": value}


def _cv(name: str, value, multiple: bool = False) -> Dict:
    return {"typeName": name, "multiple": multiple, "typeClass": "controlledVocabulary",
            "value": value}


def _compound(name: str, entries: List[Dict[str, Dict]]) -> Dict:
    return {"typeName": name, "multiple": True, "typeClass": "compound", "value": entries}


def _sub(**fields) -> Dict[str, Dict]:
    """One compound entry from ``typeName=value`` pairs, dropping ``None`` values."""
    return {k: _prim(k, v) for k, v in fields.items() if v is not None}


def citation_block(system: str, html: str, contact_email: Optional[str],
                   date: str) -> Dict:
    cfg = SYSTEMS[system]
    author = _sub(authorName=AUTHOR["name"], authorAffiliation=AUTHOR["affiliation"],
                  authorIdentifier=AUTHOR["orcid"])
    author["authorIdentifierScheme"] = _cv("authorIdentifierScheme", "ORCID")
    contact = _sub(datasetContactName=AUTHOR["name"],
                   datasetContactAffiliation=AFFILIATION_NAME,
                   datasetContactEmail=contact_email)
    keywords = [
        _sub(keywordValue=k, keywordTermURI=uri, keywordVocabulary="Wikidata" if uri else None)
        for k, uri in COMMON_KEYWORDS + cfg["keywords"]
    ]
    topics = [_sub(topicClassValue=v, topicClassVocab="DFGFO",
                   topicClassVocabURI=f"https://w3id.org/dfgfo/2024/{code}")
              for v, code in TOPICS]
    fields = [
        _prim("title", cfg["title"]),
        _compound("author", [author]),
        _compound("datasetContact", [contact]),
        _compound("dsDescription", [_sub(dsDescriptionValue=html)]),
        _cv("subject", ["Computer and Information Science", "Engineering"], multiple=True),
        _compound("keyword", keywords),
        _compound("topicClassification", topics),
        _compound("producer", [_sub(producerName=AUTHOR["name"],
                                    producerAffiliation=AFFILIATION_NAME)]),
        _compound("grantNumber", [_sub(grantNumberAgency=GRANT["agency"],
                                       grantNumberValue=GRANT["value"])]),
        _prim("depositor", AUTHOR["name"]),
        _compound("dateOfCollection", [_sub(dateOfCollectionStart=date,
                                            dateOfCollectionEnd=date)]),
        _prim("kindOfData", ["Synthetic data", "Simulation data"], multiple=True),
        _prim("relatedMaterial", [f"Generator and identification code: {GITHUB_URL}"],
              multiple=True),
    ]
    return {"displayName": "Citation Metadata", "fields": fields}


def process_block(system: str, env: Dict, commit: str, date: str) -> Dict:
    cfg = SYSTEMS[system]
    pars = []
    for name, symbol, unit, value in cfg["pars"]:
        entry = _sub(processMethodsParName=name, processMethodsParSymbol=symbol or None,
                     processMethodsParUnit=unit or None)
        if isinstance(value, float):
            entry["processMethodsParValue"] = _prim("processMethodsParValue", repr(value))
        else:
            entry["processMethodsParTextValue"] = _prim("processMethodsParTextValue", str(value))
        pars.append(entry)
    method_name, method_descr = cfg["method"]
    methods = [
        _sub(processMethodsName=method_name, processMethodsDescription=method_descr,
             processMethodsPars=", ".join(p[0] for p in cfg["pars"])),
        _sub(processMethodsName="Train/validation/test split",
             processMethodsDescription="Converging and diverging trajectories are each "
             "shuffled with numpy.random.default_rng(42).permutation and split 60/10/30 "
             "into train/validation/test and train_div/validation_div/test_div."),
    ]
    software = [
        _sub(processSoftwareName="genSecSysId", processSoftwareVersion=commit,
             processSoftwareURL=GITHUB_URL, processSoftwareLicence="MIT"),
        _sub(processSoftwareName="Python", processSoftwareVersion=env["python"],
             processSoftwareURL="https://www.python.org", processSoftwareLicence="PSF-2.0"),
        _sub(processSoftwareName="NumPy", processSoftwareVersion=env["numpy"],
             processSoftwareURL="https://numpy.org", processSoftwareLicence="BSD-3-Clause"),
        _sub(processSoftwareName="SciPy", processSoftwareVersion=env["scipy"],
             processSoftwareURL="https://scipy.org", processSoftwareLicence="BSD-3-Clause"),
    ]
    sw_line = f"genSecSysId {commit} ({cfg['generator']}); {_env_line(env)}"
    steps = []
    for i, (kind, method, outputs) in enumerate([
        ("Generation", method_name, ["id/raw/*.csv", "id/params.json"]),
        ("Postprocessing", "Train/validation/test split", [f"id/{f}/*.csv" for f in SPLIT_FOLDERS]),
    ], start=1):
        step = _sub(processStepId=str(i), processStepDate=date, processStepMethods=method,
                    processStepSoftware=sw_line)
        step["processStepType"] = _cv("processStepType", kind)
        step["processStepOutput"] = _prim("processStepOutput", outputs, multiple=True)
        steps.append(step)
    fields = [_compound("processMethods", methods), _compound("processMethodsPar", pars),
              _compound("processSoftware", software), _compound("processStep", steps)]
    return {"displayName": "Process Metadata", "fields": fields}


def dataset_json(system: str, html: str, contact_email: Optional[str], env: Dict,
                 commit: str, date: str) -> Dict:
    return {"datasetVersion": {
        "license": {"name": "CC BY 4.0", "uri": "http://creativecommons.org/licenses/by/4.0"},
        "metadataBlocks": {
            "citation": citation_block(system, html, contact_email, date),
            "process": process_block(system, env, commit, date),
        },
    }}


# --- packaging ----------------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_PY,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generator_is_committed(generator: Path) -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", str(generator)], cwd=REPO_PY,
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def published_files(data_dir: Path) -> List[Path]:
    """Relative paths of everything that goes into ``id/``: params.json and the CSVs."""
    files = [Path("params.json")]
    for folder in ["raw"] + SPLIT_FOLDERS:
        files += sorted(p.relative_to(data_dir) for p in (data_dir / folder).glob("*.csv"))
    return files


def verify_reproduction(generator: Path, data_dir: Path) -> List[str]:
    """Regenerate into a temp folder; return the published files that differ."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(generator), "--out-dir", tmp], check=True,
                       stdout=subprocess.DEVNULL)
        return [str(rel) for rel in published_files(data_dir)
                if not (Path(tmp) / rel).exists()
                or not filecmp.cmp(data_dir / rel, Path(tmp) / rel, shallow=False)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(system: str, data_dir: Path, out_dir: Path, contact_email: Optional[str],
          verify: bool, generator: Optional[Path] = None) -> Path:
    cfg = SYSTEMS[system]
    generator = generator or REPO_PY / cfg["generator"]
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__}
    commit = git_commit()
    if not generator_is_committed(generator):
        print(f"  WARNING: {generator.name} has uncommitted changes; commit {commit} does "
              "not contain the bundled version. Commit, push and re-run before uploading.")

    verified = False
    if verify:
        print(f"  verifying reproduction with {generator.name} ...")
        differing = verify_reproduction(generator, data_dir)
        if differing:
            raise SystemExit(f"{len(differing)} file(s) differ after regeneration, "
                             f"e.g. {differing[:3]}")
        verified = True

    with open(data_dir / "params.json") as f:
        params = json.load(f)
    date = dt.date.fromtimestamp((data_dir / "params.json").stat().st_mtime).isoformat()
    stats = collect_stats(data_dir)
    readme = render_readme(system, stats, env, commit, verified, params)
    html = markdown_to_html(readme)

    sys_dir = out_dir / cfg["name"]
    if sys_dir.exists():
        shutil.rmtree(sys_dir)
    stage = sys_dir / "_stage" / cfg["name"]
    for rel in published_files(data_dir):
        (stage / "id" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / rel, stage / "id" / rel)
    (stage / "code").mkdir()
    shutil.copy2(generator, stage / "code" / generator.name)
    (stage / "code" / "requirements.txt").write_text(
        f"# Python {env['python']}\nnumpy=={env['numpy']}\nscipy=={env['scipy']}\n")
    (stage / "README.md").write_text(readme)
    sums = [f"{sha256(p)}  {p.relative_to(stage).as_posix()}"
            for p in sorted(stage.rglob("*")) if p.is_file()]
    (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    zip_path = sys_dir / f"{cfg['name']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage.parent).as_posix())
    shutil.rmtree(sys_dir / "_stage")

    (sys_dir / "README.md").write_text(readme)
    (sys_dir / "description.html").write_text(html)
    with open(sys_dir / "dataset.json", "w") as f:
        json.dump(dataset_json(system, html, contact_email, env, commit, date), f, indent=2)

    n_csv = sum(s["n"] for s in stats["folders"].values())
    print(f"  {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n_csv} CSVs, "
          f"verified={verified})")
    return sys_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    ap.add_argument("--data-dir", help="dataset root (only with a single --system)")
    ap.add_argument("--out-dir", default="~/genSecSysId-Data/darus")
    ap.add_argument("--contact-email", help="datasetContactEmail; DaRUS requires one")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate and require byte-identical files (Duffing: ~2 min)")
    args = ap.parse_args(argv)

    systems = list(SYSTEMS) if args.system == "all" else [args.system]
    if args.data_dir and len(systems) > 1:
        ap.error("--data-dir needs a single --system")
    if not args.contact_email:
        print("WARNING: no --contact-email; dataset.json lacks the required "
              "datasetContactEmail.")
    out_dir = Path(args.out_dir).expanduser()
    for system in systems:
        print(f"{SYSTEMS[system]['name']}:")
        data_dir = Path(args.data_dir or SYSTEMS[system]["data_dir"]).expanduser()
        build(system, data_dir, out_dir, args.contact_email, args.verify)
    print(f"\nCreate a draft in the '{DATAVERSE_ALIAS}' collection with\n"
          "  curl -H \"X-Dataverse-key: $API_TOKEN\" -X POST "
          f"https://darus.uni-stuttgart.de/api/dataverses/{DATAVERSE_ALIAS}/datasets "
          "--upload-file <Name>/dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
