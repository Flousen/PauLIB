#!/usr/bin/env python3
"""
paper_bench.py — unified benchmark suite for the PauLIB paper.

Benchmarks
----------
  run_multiplication_benchmark  PauliString × PauliString throughput  (paulib vs PauliEngine vs Qiskit)
  run_outer_product_benchmark   PauliSum × PauliSum outer product (paulib vs PauliEngine vs Qiskit)
  run_memory_benchmark          Memory & construction time at 500 qubits, large term counts (paulib vs PauliEngine/Qiskit)

Results are printed to stdout and saved as CSV in bench/results/.

Run from the repo root:
    PYTHONPATH=python python bench/paper_bench.py

Supported qubit counts for paulib (must use these exactly):
    1-8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 500
"""

from __future__ import annotations

import csv
import os
import random
import re
import subprocess
import sys
import timeit
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# ---------------------------------------------------------------------------
# glibc < 2.32 compatibility: __libc_single_threaded was added in glibc 2.32.
# Prebuilt pauliengine wheels reference it; we inject a stub so dlopen succeeds.
# ---------------------------------------------------------------------------
def _preload_libc_single_threaded_stub() -> None:
    import ctypes
    import tempfile
    stub_so = Path(tempfile.gettempdir()) / "_pauliengine_glibc_shim.so"
    if not stub_so.exists():
        stub_c = stub_so.with_suffix(".c")
        stub_c.write_text("char __libc_single_threaded = 0;\n")
        r = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(stub_so), str(stub_c)],
            capture_output=True,
        )
        if r.returncode != 0:
            return
    try:
        ctypes.CDLL(str(stub_so), ctypes.RTLD_GLOBAL)
    except OSError:
        pass

_preload_libc_single_threaded_stub()  # preload glibc shim before pauliengine dlopen
try:
    import pauliengine as _pe_probe  # noqa: F401
except ImportError:
    pass  # pauliengine not installed; benchmarks will skip it

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS = 5  # best-of-N timing runs
WARMUP = 3   # discarded warm-up runs to avoid cold-cache bias

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def random_pauli_terms(
    n_qubits: int, n_terms: int, density: float = 0.3, seed: int = 42
) -> List[Tuple[str, float]]:
    rng = random.Random(seed)
    paulis = "XYZ"
    terms = []
    for _ in range(n_terms):
        chars = [rng.choice(paulis) if rng.random() < density else "I" for _ in range(n_qubits)]
        terms.append(("".join(chars), rng.uniform(-1.0, 1.0)))
    return terms


def timed(fn, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    all_times = timeit.repeat(fn, number=1, repeat=warmup + repeats)
    return min(all_times[warmup:])


def fmt_time(t: Optional[float]) -> str:
    if t is None:
        return "(skipped)    "
    if t >= 1.0:
        return f"{t:.3f} s     "
    if t >= 1e-3:
        return f"{t * 1e3:.2f} ms   "
    if t >= 1e-6:
        return f"{t * 1e6:.1f} µs   "
    return f"{t * 1e9:.0f} ns   "


def speedup_str(t_ref: Optional[float], t: Optional[float]) -> str:
    if t_ref is None or t is None or t == 0:
        return ""
    r = t_ref / t
    return f"({r:.1f}x faster)" if r >= 1.0 else f"({1 / r:.1f}x slower)"


def _write_csv(filename: str, rows: list) -> None:
    if not rows:
        return
    path = RESULTS_DIR / filename
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Result verification against PauliEngine
#
# Before timing, benchmarks A and B check that PauLIB and PauliEngine produce
# identical algebraic results for the exact inputs being benchmarked. The
# outcome is printed and recorded in the CSV ("PauliEngine identical").
# ---------------------------------------------------------------------------

def _split_phase_prefix(s: str) -> Tuple[complex, str]:
    """PauLIB to_string() prefix ('', 'i', '-', '-i') -> (phase, bare chars)."""
    if s.startswith("-i"):
        return -1j, s[2:]
    if s.startswith("i"):
        return 1j, s[1:]
    if s.startswith("-"):
        return -1.0, s[1:]
    return 1.0, s


def _sparse_key(chars: str) -> tuple:
    """Dense Pauli chars -> canonical sparse key ((qubit, char), ...)."""
    return tuple((q, ch) for q, ch in enumerate(chars) if ch != "I")


def _pe_string(s: str, c: complex):
    import pauliengine as pe
    return pe.PauliString(complex(c), {i: ch for i, ch in enumerate(s) if ch != "I"})


def _paulib_sum_to_dict(soa) -> dict:
    """PauliSum -> {sparse key: complex coefficient}, like terms summed."""
    out: dict = {}
    for i in range(len(soa)):
        ps, coeff = soa[i]
        phase, chars = _split_phase_prefix(ps.to_string())
        key = _sparse_key(chars)
        out[key] = out.get(key, 0j) + phase * coeff
    return out


_PE_PAULI_RE = re.compile(r"([XYZ])\((\d+)\)")


def _pe_hamiltonian_to_dict(h) -> dict:
    """QubitHamiltonian -> {sparse key: complex coefficient}.

    Parsed from to_string(); format per line: '<re> + <im>i : X(0)Y(2)'.
    Coefficients are printed with 6 decimals, which bounds the verification
    tolerance in verify_outer_product.
    """
    out: dict = {}
    for line in h.to_string().splitlines():
        if not line.strip():
            continue
        left, _, paulis = line.partition(" : ")
        re_s, _, im_s = left.partition(" + ")
        coeff = complex(float(re_s), float(im_s.rstrip("i")))
        key = tuple(sorted((int(q), ch) for ch, q in _PE_PAULI_RE.findall(paulis)))
        out[key] = out.get(key, 0j) + coeff
    return out


def _report_verify(row: dict, ok: bool, detail: str, max_dev: float) -> None:
    if ok:
        print(f"    PauliEngine result check: identical ✓ (max deviation {max_dev:.1e})")
    else:
        print(f"    PauliEngine result check: *** MISMATCH *** {detail}")
    row["PauliEngine identical"] = ok


def verify_multiplication(row: dict, terms_a, terms_b) -> None:
    """Check every pairwise product a[i]·b[i] against PauliEngine."""
    try:
        import paulib
        import pauliengine  # noqa: F401
    except ImportError:
        print("    PauliEngine result check: skipped (pauliengine not installed)")
        row["PauliEngine identical"] = None
        return
    try:
        max_dev = 0.0
        for (sa, ca), (sb, cb) in zip(terms_a, terms_b):
            pl = paulib.PauliString(sa)
            pl.multiply(paulib.PauliString(sb))
            phase, chars = _split_phase_prefix(pl.to_string())

            pe_coeff, pe_paulis = (_pe_string(sa, ca) * _pe_string(sb, cb)).to_dictionary()
            pe_key = tuple(sorted((int(q), ch) for q, ch in pe_paulis.items()))

            if pe_key != _sparse_key(chars):
                _report_verify(row, False,
                               f"pauli mismatch: paulib {pl.to_string()!r} vs "
                               f"PauliEngine {pe_paulis}", max_dev)
                return
            dev = abs(pe_coeff - phase * ca * cb)
            max_dev = max(max_dev, dev)
            if dev > 1e-12:
                _report_verify(row, False,
                               f"coefficient mismatch: paulib {phase * ca * cb} vs "
                               f"PauliEngine {pe_coeff}", max_dev)
                return
        _report_verify(row, True, "", max_dev)
    except Exception as exc:
        print(f"    PauliEngine result check: FAILED to run: {exc}")
        row["PauliEngine identical"] = None


def verify_outer_product(row: dict, terms_a, terms_b) -> None:
    """Check the full outer product against PauliEngine, like terms combined."""
    try:
        import paulib
        import pauliengine as pe
    except ImportError:
        print("    PauliEngine result check: skipped (pauliengine not installed)")
        row["PauliEngine identical"] = None
        return
    try:
        d_pl = _paulib_sum_to_dict(paulib.PauliSum(terms_a) * paulib.PauliSum(terms_b))
        ha = pe.QubitHamiltonian([_pe_string(s, c) for s, c in terms_a])
        hb = pe.QubitHamiltonian([_pe_string(s, c) for s, c in terms_b])
        d_pe = _pe_hamiltonian_to_dict(ha * hb)

        # QubitHamiltonian.to_string() truncates coefficients to 6 decimals,
        # so deviations below ~1e-5 are print truncation, not algebra errors
        # (a wrong phase would deviate by O(|coefficient|) ≈ 0.1–1).
        tol = 1e-5
        max_dev = 0.0
        for key in set(d_pl) | set(d_pe):
            va, vb = d_pl.get(key, 0j), d_pe.get(key, 0j)
            dev = abs(va - vb)
            max_dev = max(max_dev, dev)
            if dev > tol:
                label = "".join(f"{ch}({q})" for q, ch in key) or "I"
                _report_verify(row, False,
                               f"term {label}: paulib {va} vs PauliEngine {vb}", max_dev)
                return
        _report_verify(row, True, "", max_dev)
    except Exception as exc:
        print(f"    PauliEngine result check: FAILED to run: {exc}")
        row["PauliEngine identical"] = None


# ---------------------------------------------------------------------------
# Benchmark A: single PauliString × PauliString throughput
# ---------------------------------------------------------------------------

def _bench_multiplication_paulib_loop(terms_a, terms_b):
    import paulib
    pairs_b = [paulib.PauliString(s) for s, _ in terms_b]
    # multiply() is in-place — pre-build one fresh copy of A per run
    pool = [
        [paulib.PauliString(s) for s, _ in terms_a]
        for _ in range(WARMUP + REPEATS + 2)
    ]
    it = iter(pool)

    def run():
        a = next(it)
        for i in range(len(a)):
            a[i].multiply(pairs_b[i])

    return timed(run)


def _bench_multiplication_paulib_soa(terms_a, terms_b):
    import paulib
    soa_b = paulib.PauliSum(terms_b)
    pool = [paulib.PauliSum(terms_a) for _ in range(WARMUP + REPEATS + 2)]
    it = iter(pool)

    def run():
        next(it).elementwise_multiply(soa_b)

    return timed(run)


def _bench_multiplication_pauliengine(terms_a, terms_b):
    import pauliengine as pe

    def make(terms):
        return [
            pe.PauliString(complex(c), {i: ch for i, ch in enumerate(s) if ch != "I"})
            for s, c in terms
        ]

    sa, sb = make(terms_a), make(terms_b)

    def run():
        for i in range(len(sa)):
            _ = sa[i] * sb[i]

    return timed(run)


def _bench_multiplication_qiskit(terms_a, terms_b):
    from qiskit.quantum_info import Pauli
    # Qiskit uses reversed qubit order (rightmost char = qubit 0); irrelevant for throughput.
    pa = [Pauli(s) for s, _ in terms_a]
    pb = [Pauli(s) for s, _ in terms_b]

    def run():
        for i in range(len(pa)):
            _ = pa[i].compose(pb[i])

    return timed(run)


def run_multiplication_benchmark(configs: List[Tuple[int, int]]) -> None:
    print(f"\n{'=' * 72}")
    print("Benchmark A — PauliString × PauliString throughput")
    print(f"{'=' * 72}")
    rows = []
    for n_qubits, n_terms in configs:
        terms_a = random_pauli_terms(n_qubits, n_terms, seed=42)
        terms_b = random_pauli_terms(n_qubits, n_terms, seed=99)
        print(f"\n  {n_terms} pairs · {n_qubits} qubits")
        row: dict = {"n_qubits": n_qubits, "n_terms": n_terms}
        verify_multiplication(row, terms_a, terms_b)

        for label, fn in [
            ("paulib loop",  lambda: _bench_multiplication_paulib_loop(terms_a, terms_b)),
            ("paulib SoA",   lambda: _bench_multiplication_paulib_soa(terms_a, terms_b)),
            ("PauliEngine",  lambda: _bench_multiplication_pauliengine(terms_a, terms_b)),
            ("Qiskit",       lambda: _bench_multiplication_qiskit(terms_a, terms_b)),
        ]:
            try:
                t = fn()
                per_op = t / n_terms
                print(f"    {label:<18} total {fmt_time(t)}  per-op {fmt_time(per_op)}")
                row[label] = t
                row[f"{label} per-op"] = per_op
            except Exception as exc:
                print(f"    {label:<18} FAILED: {exc}")
                row[label] = None
                row[f"{label} per-op"] = None

        rows.append(row)

    _write_csv("bench_A_ps_mul.csv", rows)


# ---------------------------------------------------------------------------
# Benchmark B: PauliSum × PauliSum outer product
# ---------------------------------------------------------------------------

def _bench_outer_product_paulib_soa(terms_a, terms_b):
    import paulib
    sa = paulib.PauliSum(terms_a)
    sb = paulib.PauliSum(terms_b)
    return timed(lambda: sa * sb)


def _bench_outer_product_pauliengine(terms_a, terms_b):
    import pauliengine as pe

    def make_h(terms):
        strings = [
            pe.PauliString(complex(c), {i: ch for i, ch in enumerate(s) if ch != "I"})
            for s, c in terms
        ]
        return pe.QubitHamiltonian(strings)

    ha, hb = make_h(terms_a), make_h(terms_b)
    return timed(lambda: ha * hb)


def _bench_outer_product_qiskit(terms_a, terms_b):
    from qiskit.quantum_info import SparsePauliOp
    ha = SparsePauliOp.from_list(terms_a)
    hb = SparsePauliOp.from_list(terms_b)
    return timed(lambda: (ha @ hb).simplify())


def run_outer_product_benchmark(configs: List[Tuple[int, int]]) -> None:
    print(f"\n{'=' * 72}")
    print("Benchmark B — PauliSum × PauliSum (outer product, N×N → N² terms)")
    print(f"{'=' * 72}")
    rows = []
    for n_qubits, n_terms in configs:
        terms_a = random_pauli_terms(n_qubits, n_terms, seed=42)
        terms_b = random_pauli_terms(n_qubits, n_terms, seed=99)
        print(f"\n  {n_terms}×{n_terms} → {n_terms**2} terms · {n_qubits} qubits")
        row: dict = {"n_qubits": n_qubits, "n_terms": n_terms}
        verify_outer_product(row, terms_a, terms_b)

        t_ref = None
        for label, fn in [
            ("paulib SoA",  lambda: _bench_outer_product_paulib_soa(terms_a, terms_b)),
            ("PauliEngine", lambda: _bench_outer_product_pauliengine(terms_a, terms_b)),
            ("Qiskit",      lambda: _bench_outer_product_qiskit(terms_a, terms_b)),
        ]:
            try:
                t = fn()
                s = speedup_str(t_ref, t)
                print(f"    {label:<18} {fmt_time(t)}  {s}")
                if t_ref is None:
                    t_ref = t
                row[label] = t
            except Exception as exc:
                print(f"    {label:<18} FAILED: {exc}")
                row[label] = None

        rows.append(row)

    _write_csv("bench_B_sum_mul.csv", rows)


# ---------------------------------------------------------------------------
# Benchmark F: 500-qubit stress test — memory & construction at scale
# ---------------------------------------------------------------------------

# PauliEngine stores each non-I qubit as a Python dict entry; at density 0.3
# and 500 qubits that is ~150 dict entries per term, so memory blows up fast.
# Cap it at 10k terms to avoid OOM hanging the benchmark process.
_MEMORY_PE_LIMIT = 10_000

# Memory is measured in a fresh subprocess per (library, n_terms) so that C++
# heap reuse across iterations does not suppress the RSS delta for PauLIB.
_MEMORY_BENCHMARK_CODE = """\
import sys, json, gc, random, timeit, ctypes, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, {python_path!r})

# glibc < 2.32 shim so pauliengine dlopen succeeds in subprocesses too
def _load_libc_shim():
    stub_so = Path(tempfile.gettempdir()) / '_pauliengine_glibc_shim.so'
    if not stub_so.exists():
        stub_c = stub_so.with_suffix('.c')
        stub_c.write_text('char __libc_single_threaded = 0;\\n')
        r = subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(stub_so), str(stub_c)],
                           capture_output=True)
        if r.returncode != 0:
            return
    try:
        ctypes.CDLL(str(stub_so), ctypes.RTLD_GLOBAL)
    except OSError:
        pass
_load_libc_shim()

def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return 0.0

rng = random.Random(42)
paulis = 'XYZ'
terms = [
    (''.join([rng.choice(paulis) if rng.random() < 0.3 else 'I'
              for _ in range({n_qubits})]),
     rng.uniform(-1.0, 1.0))
    for _ in range({n_terms})
]

gc.collect()
before = rss_mb()
obj = {build_expr}
gc.collect()
# Return freed construction-time allocations (pybind11 string copies, etc.) to OS
try:
    import ctypes; ctypes.CDLL('libc.so.6').malloc_trim(0)
except Exception:
    pass
after = rss_mb()
times = timeit.repeat(lambda: {build_expr}, number=1, repeat={warmup}+{repeats})
build_s = min(times[{warmup}:])
print(json.dumps({{'mem_mb': max(0.0, after - before), 'build_s': build_s}}))
"""

_MEMORY_BUILD_EXPRS = {
    "paulib SoA":  "__import__('paulib').PauliSum(terms)",
    "PauliEngine": (
        "[__import__('pauliengine').PauliString(complex(c),"
        " {i: ch for i, ch in enumerate(s) if ch != 'I'})"
        " for s, c in terms]"  # QubitHamiltonian too slow to repeat; measure list build
    ),
    "Qiskit": (
        "__import__('qiskit.quantum_info', fromlist=['SparsePauliOp'])"
        ".SparsePauliOp.from_list(terms)"
    ),
}

def _measure_memory_subprocess(python_path: str, label: str,
                                n_qubits: int, n_terms: int) -> Optional[dict]:
    code = _MEMORY_BENCHMARK_CODE.format(
        python_path=python_path,
        n_qubits=n_qubits,
        n_terms=n_terms,
        build_expr=_MEMORY_BUILD_EXPRS[label],
        warmup=WARMUP,
        repeats=REPEATS,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip().splitlines()[-1])
        import json
        return json.loads(result.stdout.strip())
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def run_memory_benchmark() -> None:
    N = 500
    term_counts = [10_000, 100_000, 1_000_000]
    print(f"\n{'=' * 72}")
    print(f"Benchmark F — Large qubit count ({N} qubits): memory & construction time")
    print(f"{'=' * 72}")

    python_path = str(Path(__file__).parent.parent / "python")
    rows = []
    for n_terms in term_counts:
        print(f"\n  {n_terms:,} terms · {N} qubits")
        row: dict = {"n_qubits": N, "n_terms": n_terms}

        libs = [
            ("paulib SoA",  True),
            ("PauliEngine", n_terms <= _MEMORY_PE_LIMIT),
            ("Qiskit",      True),
        ]

        for label, enabled in libs:
            if not enabled:
                print(f"    {label:<18} (skipped — too many terms for sparse dict format)")
                row[f"{label} mem_mb"] = None
                row[f"{label} build_s"] = None
                continue
            try:
                data = _measure_memory_subprocess(python_path, label, N, n_terms)
                mb, t = data["mem_mb"], data["build_s"]
                print(f"    {label:<18} {mb:7.1f} MB    build {fmt_time(t)}")
                row[f"{label} mem_mb"] = mb
                row[f"{label} build_s"] = t
            except Exception as exc:
                print(f"    {label:<18} FAILED: {exc}")
                row[f"{label} mem_mb"] = None
                row[f"{label} build_s"] = None

        rows.append(row)

    _write_csv("bench_F_large_qubits.csv", rows)


# n_qubits can be any value up to paulib.MAX_QUBITS (1024)
MULTIPLICATION_CONFIGS = [(500, 100), (500, 200), (500, 500), (500, 1000)]
OUTER_PRODUCT_CONFIGS = [(500, 25), (500, 50), (500, 100), (500, 200)]

if __name__ == "__main__":
    print(f"PauLIB paper benchmarks  ·  best-of-{REPEATS} (after {WARMUP} warm-up)")
    print(f"Results → {RESULTS_DIR}/")

    run_multiplication_benchmark(MULTIPLICATION_CONFIGS)
    run_outer_product_benchmark(OUTER_PRODUCT_CONFIGS)
    run_memory_benchmark()
