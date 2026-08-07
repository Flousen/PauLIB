"""
paulib — Python bindings for the PauLIB high-performance Pauli string library.

Usage
-----
from paulib import PauliString, PauliSum

ps  = PauliString("XYZZ")
soa = PauliSum([("XYZZ", 0.5), ("ZZZY", 1.2)])

Any qubit count up to MAX_QUBITS is supported: the compiled extension
provides one template instantiation per 64-bit word count (capacity 64,
128, ..., 1024) and each object tracks its actual qubit count at runtime.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

# The compiled extension is placed in this package directory by CMake.
from . import _paulib  # type: ignore[import]

SUPPORTED_CAPACITIES: List[int] = sorted(_paulib.SUPPORTED_CAPACITIES)
"""Capacities (qubit counts rounded up to whole 64-bit words) with template
instantiations in the compiled extension."""

MAX_QUBITS: int = SUPPORTED_CAPACITIES[-1]
"""Largest supported qubit count."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_qubits(s: str) -> int:
    """Count Pauli operators in a string, skipping the optional phase prefix."""
    i = 0
    if i < len(s) and s[i] in "+-":
        i += 1
    if i < len(s) and s[i] == "i":
        i += 1
    return len(s) - i


def _capacity_for(n: int) -> int:
    """Smallest instantiated capacity that fits n qubits."""
    if n < 1:
        raise ValueError("Qubit count must be at least 1.")
    for cap in SUPPORTED_CAPACITIES:
        if n <= cap:
            return cap
    raise ValueError(
        f"Qubit count {n} exceeds the maximum of {MAX_QUBITS} qubits."
    )


def _cls(prefix: str, cap: int):
    return getattr(_paulib, f"{prefix}_{cap}")


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def PauliString(s: str):
    """
    Create a PauliString from a string representation.

    Format: ``[+|-][i]<pauli_chars>``
    where *pauli_chars* are ``X``, ``Y``, ``Z``, or ``I``.

    Examples::

        PauliString("XYZZ")     # 4 qubits
        PauliString("+iXYZZ")   # 4 qubits with phase
    """
    cap = _capacity_for(_count_qubits(s))
    return _cls("PauliString", cap)(s)


def PauliSum(
    terms: Union[str, Sequence[str], Sequence[Tuple[str, float]]],
    coefficient: float = 1.0,
):
    """
    Create a PauliSum (Structure-of-Arrays) with ``double`` coefficients.

    The SoA layout is SIMD-friendly and provides fast time evolution
    (``evolve()``).

    Parameters
    ----------
    terms:
        - A single Pauli string (coefficient defaults to 1.0, or use *coefficient*).
        - A list of Pauli strings (all coefficients set to 1.0).
        - A list of ``(pauli_string, coefficient)`` pairs.

    Examples::

        soa = PauliSum([("XYZZ", 0.5), ("ZZZY", 1.2)])
    """
    if isinstance(terms, str):
        cap = _capacity_for(_count_qubits(terms))
        return _cls("PauliSum", cap)([(terms, coefficient)])

    terms = list(terms)
    if not terms:
        raise ValueError(
            "Cannot infer qubit count from an empty list. "
            "Use a non-empty list or pass a single string."
        )

    strings = [t if isinstance(t, str) else t[0] for t in terms]
    cap = _capacity_for(max(_count_qubits(s) for s in strings))
    return _cls("PauliSum", cap)(terms)


def random_pauli_sum(n_qubits: int, num_terms: int, seed: int = 42):
    """Return a random :class:`PauliSum` with *n_qubits* qubits."""
    cap = _capacity_for(n_qubits)
    return getattr(_paulib, f"random_pauli_sum_{cap}")(num_terms, seed, n_qubits)


TruncationOptions = _paulib.TruncationOptions
"""Truncation settings for Pauli propagation and operator dynamics:
``max_weight``, ``coefficient_cutoff``, ``max_terms``."""


def _cap_of(obj) -> int:
    """Capacity of a bound PauliSum/PauliString object (from its type name)."""
    return int(type(obj).__name__.rsplit("_", 1)[1])


def _fn(name: str, obj):
    return getattr(_paulib, f"{name}_{_cap_of(obj)}")


def trace_product(a, b) -> float:
    """Re tr(A*B), computed without forming the product."""
    return _fn("trace_product", a)(a, b)


def commutator(a, b):
    """[A, B], canonicalized."""
    return _fn("commutator", a)(a, b)


def i_commutator(a, b):
    """i[A, B], canonicalized (Hermitian for Hermitian inputs)."""
    return _fn("i_commutator", a)(a, b)


def propagate_expectation(observable, rotations, options=None) -> float:
    """Pauli propagation: ``<0|U† O U|0>`` for the rotation circuit U.

    ``rotations`` lists gates ``exp(-i theta/2 P)`` in application order
    (term = generator P, coefficient = angle theta).
    """
    opts = options if options is not None else TruncationOptions()
    return _fn("propagate_expectation", observable)(observable, rotations, opts)


__all__ = [
    "SUPPORTED_CAPACITIES",
    "MAX_QUBITS",
    "PauliString",
    "PauliSum",
    "random_pauli_sum",
    "TruncationOptions",
    "trace_product",
    "commutator",
    "i_commutator",
    "propagate_expectation",
    "_paulib",
]
