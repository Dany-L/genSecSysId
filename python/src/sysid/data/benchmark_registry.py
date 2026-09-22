"""Registry mapping `nonlinear_benchmarks` datasets onto this repo's layout.

One :class:`BenchmarkSpec` per dataset says where the data comes from, how it is
split, and what the columns are called. ``scripts/prepare_benchmark.py`` walks
this registry; nothing else about a benchmark lives in code.

What the package actually gives us (measured, v0.1.2)
----------------------------------------------------
Its return shape is not uniform -- a single record, a list, a
``(train_val, test)`` tuple of lists, or a bare flat list -- so
:func:`normalize_records` flattens all of it into two lists of :class:`Record`.

Two facts drive the split policy:

* **No benchmark ships a validation set.** They give ``(train_val, test)``, so
  validation is always carved out of ``train_val`` here.
* **5 of 10 have no official train/test split at all** (the package prints a
  warning). Of the supported ones only the F-16 is affected; it returns a flat
  list of 14 records, so its split is *declared* below rather than taken.

``state_initialization_window_length`` is set on the package's TEST records
only. It is the benchmark's own washout spec and seeds ``training.warmup_steps``.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Record(NamedTuple):
    """One input/output record, already 2-D."""

    name: str
    u: np.ndarray  # (N, nd)
    y: np.ndarray  # (N, ne)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Everything benchmark-specific about materializing one dataset."""

    name: str
    #: ``() -> (train_records, test_records, sampling_time)``.
    fetch: Optional[Callable[..., Tuple[List[Record], List[Record], Optional[float]]]] = None
    input_cols: Sequence[str] = ("u",)
    output_cols: Sequence[str] = ("y",)
    #: Which test record lands in ``test/``. ``None`` -> all of them (only valid
    #: when they share a length; see the ragged rule in benchmark_prep).
    default_test_record: Optional[str] = None
    #: The benchmark's own washout length, from its test records.
    init_window: Optional[int] = None
    #: State dimension for the generated starter config. Explicit rather than
    #: inferred: for Silverbox it is the known plant order, elsewhere it is a
    #: truncation level and the config comment reports how much Hankel energy
    #: that keeps so it can be revisited.
    default_nx: int = 4
    #: Dead-zone width of the generated config's model.
    default_nw: int = 16
    #: For a DECLARED split (the package returns a flat list with no split of
    #: its own): which record to fit, and which become test/siblings. Named
    #: rather than positional so the choice is readable and checkable.
    declared_train_record: Optional[str] = None
    declared_test_records: Sequence[str] = ()
    #: True when the package defines the train/test split; False when we declare it.
    official_split: bool = True
    reference: str = ""
    notes: str = ""
    #: Upstream names to try beyond `name` -- the package renames things
    #: between versions (ParWHF in 0.1.2 became ParWH in 1.0.1).
    loader_aliases: Sequence[str] = ()
    #: Set to a reason string to register a dataset we deliberately do not support.
    unsupported: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.unsupported is None


# ── flattening the package's return shapes ────────────────────────────────────
def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _flatten(obj) -> list:
    """Depth-first flatten of arbitrarily nested lists/tuples of records."""
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return [obj]


def normalize_records(
    obj, prefix: str = "record"
) -> Tuple[List[Record], List[Record], Optional[float]]:
    """``(train_records, test_records, sampling_time)`` from any package return.

    A 2-tuple is read as ``(train_val, test)``; anything else is treated as a
    flat list with no split, which the caller must then split itself.

    Names come from POSITION, never from ``.name``: v0.1.2 labels Silverbox's
    second test record 'test SB multisine' when it is in fact the arrow record,
    so the attribute cannot be trusted.
    """
    def to_records(part, tag) -> List[Record]:
        items = _flatten(part)
        return [
            Record(f"{tag}_{i:04d}", _as_2d(r.u), _as_2d(r.y))
            for i, r in enumerate(items)
        ]

    sampling_time = None
    for rec in _flatten(obj):
        ts = getattr(rec, "sampling_time", None)
        if ts is not None:
            sampling_time = float(ts)
            break

    if isinstance(obj, tuple) and len(obj) == 2:
        return (
            to_records(obj[0], f"{prefix}_train"),
            to_records(obj[1], f"{prefix}_test"),
            sampling_time,
        )
    return to_records(obj, prefix), [], sampling_time


def apply_declared_split(
    spec: "BenchmarkSpec", records: Sequence[Record]
) -> Tuple[List[Record], List[Record]]:
    """Split a flat record list according to the spec's declared choice.

    Used when the package provides no split of its own. Only the declared
    records are materialized: the F-16 has 20 records of 49k-117k samples and
    writing all of them as siblings would cost hundreds of MB of CSV for
    records nothing references.
    """
    by_name = {r.name: r for r in records}
    missing = [
        n for n in (spec.declared_train_record, *spec.declared_test_records)
        if n and n not in by_name
    ]
    if missing:
        raise KeyError(
            f"{spec.name}: declared record(s) {missing} not found. "
            f"Available: {sorted(by_name)}"
        )
    return (
        [by_name[spec.declared_train_record]],
        [by_name[n] for n in spec.declared_test_records],
    )


def rename(records: Sequence[Record], names: Sequence[str]) -> List[Record]:
    """Attach meaningful names to positionally-ordered records."""
    if len(names) != len(records):
        raise ValueError(f"{len(names)} names for {len(records)} records")
    return [Record(n, r.u, r.y) for n, r in zip(names, records)]


# ── locating a loader across package versions ─────────────────────────────────
# Which submodule a benchmark lives in is NOT stable across versions of
# nonlinear_benchmarks. v0.1.2 puts CED/EMPS/Silverbox/... at the top level and
# F16/ParWHF/BoucWen in `not_splitted_benchmarks`; other versions move them.
# Hard-coding the submodule produced
#     AttributeError: module 'nonlinear_benchmarks.not_splitted_benchmarks'
#     has no attribute 'ParWHF'
# on a server with a different build, so the lookup searches instead -- ending
# with the package's own `all_benchmarks` lists, which name the functions
# regardless of where they are defined.
_LOADER_SUBMODULES = (
    "not_splitted_benchmarks",
    "benchmarks",
    "splitted_benchmarks",
)
_LOADER_REGISTRIES = (
    "all_benchmarks",
    "all_not_splitted_benchmarks",
    "all_splitted_benchmarks",
)


def package_version() -> str:
    """Installed nonlinear_benchmarks version, or 'unknown'.

    Recorded in metadata.json because the package's splits are NOT stable
    across versions: 0.1.2 gives ParWH 200 records of 16384 samples and warns
    there is no official split; 1.0.1 gives 100 of 32768 and declares the split
    official. Silverbox is identical in both. A result is only comparable to
    another produced by the same version.
    """
    try:
        import importlib.metadata as md

        return md.version("nonlinear_benchmarks")
    except Exception:
        try:
            import nonlinear_benchmarks as nb

            return str(getattr(nb, "__version__", "unknown"))
        except Exception:
            return "unknown"


def has_official_split(loader_name: str, aliases: Sequence[str] = ()) -> Optional[bool]:
    """Does the INSTALLED package consider this benchmark officially split?

    Derived rather than declared, because it moved: ParWH is absent from
    ``all_splitted_benchmarks`` in 0.1.2 and present in 1.0.1. ``None`` when the
    package exposes no such list to ask.
    """
    try:
        import nonlinear_benchmarks as nb
    except Exception:
        return None
    listed = getattr(nb, "all_splitted_benchmarks", None)
    if not listed:
        return None
    names = {getattr(f, "__name__", "") for f in listed}
    return bool(names & {loader_name, *aliases})


def available_loaders() -> Dict[str, str]:
    """``{loader name: where it was found}`` for the installed package."""
    import nonlinear_benchmarks as nb

    found: Dict[str, str] = {}
    for sub in ("", *_LOADER_SUBMODULES):
        mod = nb if sub == "" else getattr(nb, sub, None)
        if mod is None:
            continue
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr, None)
            if callable(obj) and getattr(obj, "__module__", "").startswith(
                "nonlinear_benchmarks"
            ):
                found.setdefault(attr, sub or "nonlinear_benchmarks")
    for reg in _LOADER_REGISTRIES:
        for fn in getattr(nb, reg, None) or ():
            name = getattr(fn, "__name__", None)
            if name:
                found.setdefault(name, reg)
    return found


def find_loader(name: str, aliases: Sequence[str] = ()):
    """The package's loader function for ``name``, wherever it lives."""
    import nonlinear_benchmarks as nb

    candidates = [name, *aliases]
    modules = [nb]
    for sub in _LOADER_SUBMODULES:
        mod = getattr(nb, sub, None)
        if mod is not None:
            modules.append(mod)

    for cand in candidates:
        for mod in modules:
            fn = getattr(mod, cand, None)
            if callable(fn):
                return fn
    for reg in _LOADER_REGISTRIES:
        for fn in getattr(nb, reg, None) or ():
            if getattr(fn, "__name__", None) in candidates:
                return fn

    have = available_loaders()
    version = getattr(nb, "__version__", "unknown")
    raise AttributeError(
        f"nonlinear_benchmarks (version {version}) has no loader named "
        f"{' or '.join(repr(c) for c in candidates)}. It does have: "
        f"{', '.join(sorted(have))}. If one of those is the right dataset under "
        f"another name, add it to the spec's `loader_aliases`."
    )


# ── per-benchmark fetch hooks ─────────────────────────────────────────────────
def _pair(loader_name: str, *, test_names=None, atleast_2d=True, prefix=None,
          aliases: Sequence[str] = ()):
    """Fetch for a package benchmark that returns ``(train_val, test)``."""
    def fetch(force_download: bool = False):
        loader = find_loader(loader_name, aliases)
        kwargs = {"force_download": force_download}
        if atleast_2d:
            kwargs["atleast_2d"] = True
        kwargs = _supported_kwargs(loader, kwargs)
        train, test, ts = normalize_records(loader(**kwargs), prefix or loader_name)
        if test_names:
            test = rename(test, test_names)
        return train, test, ts
    return fetch


def _fetch_f16(force_download: bool = False):
    """The F-16, read from its ``.mat`` files rather than via the wrapper.

    ``nonlinear_benchmarks.F16()`` returns ONE accelerometer (``output_index``)
    and drops the record names, and this project fits all three channels with
    the config choosing the subset. So the files are read directly, which also
    keeps the amplitude level in each record's name -- the split below is stated
    in terms of those levels.
    """
    from pathlib import Path

    loadmat = _find_loadmat()
    f16 = find_loader("F16")
    paths = {
        Path(p).stem: p
        for p in f16(**_supported_kwargs(
            f16, {"data_file_locations": True, "force_download": force_download}
        ))
    }
    records, sampling_times = [], set()
    for name in sorted(paths):
        mat = loadmat(paths[name])
        force = np.asarray(mat["Force"][0], dtype=float).reshape(-1, 1)
        accel = np.asarray(mat["Acceleration"], dtype=float)
        if accel.shape[0] != 3:
            raise RuntimeError(f"{name}: expected 3 accelerometer rows, got {accel.shape}")
        y = accel.T
        if len(force) != len(y):
            raise RuntimeError(f"{name}: {len(force)} inputs vs {len(y)} outputs")
        records.append(Record(name, force, y))
        sampling_times.add(round(1.0 / float(mat["Fs"][0, 0]), 12))
    if len(sampling_times) != 1:
        raise RuntimeError(f"F16 records disagree on sampling time: {sampling_times}")
    return records, [], sampling_times.pop()


def _supported_kwargs(fn, kwargs: Dict) -> Dict:
    """Drop kwargs the installed version's signature does not accept."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _find_loadmat():
    """``loadmat`` moved between versions too; find it or fall back to scipy."""
    import nonlinear_benchmarks as nb

    for sub in ("benchmarks", "not_splitted_benchmarks", "utilities"):
        mod = getattr(nb, sub, None)
        fn = getattr(mod, "loadmat", None) if mod is not None else None
        if callable(fn):
            return fn
    from scipy.io import loadmat as scipy_loadmat

    return scipy_loadmat


# ── the registry ──────────────────────────────────────────────────────────────
F16_TRAIN_RECORD = "F16Data_FullMSine_Level3"
F16_TEST_RECORD = "F16Data_FullMSine_Level4_Validation"

REGISTRY: Dict[str, BenchmarkSpec] = {
    "CED": BenchmarkSpec(
        name="CED",
        default_nx=4,
        default_nw=12,
        fetch=_pair("CED"),
        init_window=4,
        reference="Coupled electric drives; see nonlinearbenchmark.org.",
        notes=(
            "Two short records per side (400 train, 100 test samples). The "
            "shortest benchmark here by two orders of magnitude, so windowing "
            "has to stay well under 400."
        ),
    ),
    "EMPS": BenchmarkSpec(
        name="EMPS",
        default_nx=4,
        default_nw=16,
        fetch=_pair("EMPS"),
        init_window=20,
        reference="Electro-Mechanical Positioning System; see nonlinearbenchmark.org.",
        notes="A positioning system dominated by friction.",
    ),
    "Silverbox": BenchmarkSpec(
        name="Silverbox",
        default_nx=2,
        default_nw=20,
        fetch=_pair("Silverbox",
                    test_names=("multisine", "arrow", "arrow_no_extrapolation")),
        default_test_record="multisine",
        init_window=50,
        reference=(
            "T. Wigren and J. Schoukens, 'Three free data sets for development "
            "and benchmarking in nonlinear system identification', ECC 2013."
        ),
        notes=(
            "An electronic Duffing oscillator: 2nd-order LTI with a cubic static "
            "nonlinearity in feedback, which is why nx=2 is the plant order and "
            "not a hyperparameter. The three test records are RAGGED and the "
            "'arrow' one is the interesting one: its amplitude ramps PAST the "
            "training range (u to 0.149 vs 0.101, y to 0.300 vs 0.216), so it "
            "leaves the region any regional certificate was fitted over. "
            "'arrow_no_extrapolation' is its truncation that stays inside."
        ),
    ),
    "WienerHammerBenchMark": BenchmarkSpec(
        name="WienerHammerBenchMark",
        default_nx=6,
        default_nw=20,
        fetch=_pair("WienerHammerBenchMark"),
        init_window=50,
        reference="Wiener-Hammerstein benchmark; see nonlinearbenchmark.org.",
        notes="A static nonlinearity sandwiched between two linear blocks.",
    ),
    "ParWHF": BenchmarkSpec(
        name="ParWHF",
        default_nx=6,
        default_nw=20,
        fetch=_pair("ParWHF", atleast_2d=False,
                    aliases=("ParWH", "Parallel_Wiener_Hammerstein")),
        loader_aliases=("ParWH", "Parallel_Wiener_Hammerstein"),
        official_split=False,
        reference="Parallel Wiener-Hammerstein benchmark; see nonlinearbenchmark.org.",
        notes=(
            "UPSTREAM NAME CHANGED: 'ParWHF' in nonlinear_benchmarks 0.1.2, "
            "'ParWH' from 1.0.1. The DATA changed with it -- 0.1.2 returns 200 "
            "train / 12 test realizations of 16384 samples and warns there is "
            "no official split; 1.0.1 returns 100 / 5 of 32768 and lists it as "
            "officially split. Same total samples, paired up. The actual counts "
            "for THIS run are in metadata.json under 'records', along with the "
            "package version that produced them. The realizations are "
            "independent experiments either way, so the train/validation split "
            "is by record rather than within a record."
        ),
    ),
    "F16": BenchmarkSpec(
        name="F16",
        default_nx=9,
        default_nw=24,
        fetch=_fetch_f16,
        output_cols=("y1", "y2", "y3"),
        default_test_record=F16_TEST_RECORD,
        declared_train_record=F16_TRAIN_RECORD,
        declared_test_records=(F16_TEST_RECORD, "F16Data_FullMSine_Level6_Validation"),
        official_split=False,
        reference=(
            "J.P. Noel and M. Schoukens, 'F-16 aircraft benchmark based on "
            "ground vibration test data', Workshop on Nonlinear System "
            "Identification Benchmarks, Brussels, 2017."
        ),
        notes=(
            "THE BENCHMARK DEFINES NO TRAIN/TEST SPLIT -- the package returns a "
            f"flat list of 14 records. Declared here: fit {F16_TRAIN_RECORD}, "
            f"test on {F16_TEST_RECORD}, one amplitude level up, so the test "
            "set is a mild EXTRAPOLATION rather than a resample. Amplitude "
            "grows monotonically with the level (|F|max 43 -> 337 N over the "
            "FullMSine levels), which is what makes the level the extrapolation "
            "axis. All three accelerometers are written; data.output_col picks "
            "the subset, and nx >= ne is required or the certified output set "
            "is degenerate."
        ),
    ),
    # ── registered but deliberately not supported ────────────────────────────
    "Cascaded_Tanks": BenchmarkSpec(
        name="Cascaded_Tanks",
        unsupported=(
            "the signals are not zero-mean (levels around an operating point, "
            "u in [0.4, 6.5], y in [2.9, 10.0]) and the Lur'e model's "
            "equilibrium is the origin. Only 1024 samples per record either way."
        ),
    ),
    "BoucWen": BenchmarkSpec(
        name="BoucWen",
        unsupported=(
            "ships NO estimation data -- only 2 validation signals and a "
            "MATLAB-encrypted .p simulator. Its own protocol forbids estimating "
            "on the test data, so it needs a reimplementation of the published "
            "Newmark integration to generate training data."
        ),
    ),
    "Industrial_robot": BenchmarkSpec(
        name="Industrial_robot",
        unsupported="6-input/6-output MIMO; out of the SISO scope.",
    ),
    "WienerHammerstein_Process_Noise": BenchmarkSpec(
        name="WienerHammerstein_Process_Noise",
        unsupported="423 MB download; out of scope.",
    ),
}

SUPPORTED = [k for k, v in REGISTRY.items() if v.supported]


def get(name: str) -> BenchmarkSpec:
    """Look a spec up, with a helpful error listing what is available."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown benchmark {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        )
    spec = REGISTRY[name]
    if not spec.supported:
        raise ValueError(f"{name} is not supported: {spec.unsupported}")
    return spec
