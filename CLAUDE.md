# PauLIB: A High-Performance Library for Processing Pauli Strings

C++20 library for high-performance Pauli string and Pauli sum operations used
in quantum computing (VQE, quantum algorithms). Core library is header-only;
GPU (CUDA) and MPI are optional, separately-compiled add-ons.

This document is written for integrating PauLIB into another project. For
instructions on building/testing PauLIB itself, see the end of this file.

## Integrating into a project

The core library is a header-only CMake `INTERFACE` target named `paulib`
(see `src/CMakeLists.txt`). Typical integration as a submodule:

```cmake
add_subdirectory(external/paulib)
target_link_libraries(your_target PRIVATE paulib)
```

`paulib` propagates its include path (`include/`), C++20, `-O3 -march=native
-ffast-math`, and `OpenMP::OpenMP_CXX`. Your target does not need to add
anything else for CPU use.

Requirements:
- GCC or Clang with C++20 support.
- OpenMP (`find_package(OpenMP REQUIRED)` — the top-level `CMakeLists.txt`
  already does this; a consumer that only does `add_subdirectory` gets it
  transitively).
- The `eigen` submodule (`external/eigen`), only needed if you use
  `GSim.h`. Not needed for the rest of the library.
- Without CMake: it's plain headers, so adding `include/` to your include
  path and compiling with `-std=c++20 -fopenmp` (plus Eigen's include path
  if you use `GSim.h`) also works.

If you cloned this repo as a submodule, initialize its own submodules first:
```bash
git submodule update --init --recursive
```

Single include for everything CPU-side:
```cpp
#include "PauLib.h"
using namespace paulib;
```
GPU and MPI headers are separate and must be included explicitly (see
"Optional layers" below) — they are not pulled in by `PauLib.h`.

## Choosing template parameters: `N` and `T`

Both core types are templated on:
- **`N`** — compile-time qubit *capacity*. Storage cost is `ceil(N/64)`
  `size_t` words per term, so pick the largest qubit count you will ever
  need for that instantiation (e.g. `PauliSum<64, double>` for up to 64
  qubits, `PauliSum<128, double>` for up to 128, etc.). Different `N`
  instantiate different, unrelated types — you cannot mix a `PauliSum<64,
  double>` and a `PauliSum<128, double>` in one expression.
- **`T`** — coefficient type, normally `double`. The core CPU library is
  generic in `T`; the GPU and MPI layers currently hard-require
  `T = double` (enforced by `static_assert`).

The **logical/runtime qubit count** (`num_qubits()`) can be smaller than the
capacity `N`, and is tracked separately per object:
- Inferred automatically as the longest string passed to a constructor, when
  no qubit count is given.
- Set explicitly with the `(qubits, ...)` constructor overloads, which
  `throw std::invalid_argument` if `qubits > N`.

## Quickstart

```cpp
#include "PauLib.h"
using namespace paulib;

// A 4-qubit sum; capacity N=64 is plenty of headroom.
PauliSum<64, double> h(std::vector<std::pair<std::string, double>>{
    {"XXII", 1.0}, {"IZZI", -0.5}, {"IIXY", 0.25}});

// Rotations: term = generator, coefficient = angle theta (exp(-i*theta/2*P)).
PauliSum<64, double> rot(std::vector<std::pair<std::string, double>>{
    {"IZZI", M_PI / 4}});
h.evolve(rot);                       // Heisenberg-conjugate h under the gate
std::cout << h;                      // uses operator<< / get_str()

double e = h.zero_state_expectation();  // <0...0| h |0...0>
```

## Core types

### `PauliString<N>` — a single Pauli operator

Bit-packed: `x[ceil(N/64)]`, `z[ceil(N/64)]` word arrays plus a `flags`
phase/sign byte. Construct from a string; an optional `-`, `i`, or `-i`
prefix sets sign/phase:

```cpp
PauliString<8> p("-iXYZI");   // throws std::invalid_argument if the
                              // operator-character count exceeds N
p.to_string();                // round-trips back to "-iXYZI"
p.num_qubits();               // 4 here (inferred from string length)
```

Key operations: `p.multiply(other)` / `p *= other` / `p * other`,
`p.commute(other)`, `p.anti_commute(other)`, `inner_product(a, b)`.

Gate methods use **physics names**, not common gate names — this is the one
naming detail most likely to trip up code written against the bulk API:

| Physics name (on `PauliString<N>`) | Common equivalent |
|---|---|
| `apply_H_XZ` | Hadamard `H` |
| `apply_sqrt_Z` | `S` |
| `apply_ZC_X` | `CNOT` |
| `apply_ZC_Z` | `CZ` |
| `apply_ZC_Y` | `CY` |

`apply_H_YZ`, `apply_H_XY`, `apply_sqrt_X`, `apply_sqrt_Y`, `apply_X/Y/Z/I`,
`apply_XC_X/Y/Z`, `apply_YC_X/Y/Z`, `apply_SWAP`, `apply_iSWAP` round out the
set (defined in `PauliStringClifford.h`). Every gate is available both as a
member (`p.apply_H_XZ(0)`) and as a free function
(`apply_H_XZ(p, 0)`).

### `PauliSum<N, T>` — a weighted sum of Pauli strings (bulk container)

Struct-of-Arrays: `x_data`/`z_data` (per-word `vector<size_t>`), `flags`
(`vector<uint8_t>`), `value` (`vector<T>`, the coefficients) — all public,
same length, index `i` selects one term across all four arrays.

```cpp
PauliSum<64, double> s(std::vector<std::pair<std::string, double>>{
    {"XX", 1.0}, {"-ZZ", 0.5}});
PauliSum<64, double> t(10, "XI", 2.0);       // explicit 10-qubit sum, 1 term
s.get_length();                              // term count
s.get_pauli_string(0);                       // -> PauliString<64> for term 0
s.get_str();                                 // "1.000000 * XX\n0.500000 * -ZZ\n"
```

Arithmetic: `+`, `+=`, `-`, `-=`, unary `-`, scalar `*`/`*=` (all combine
duplicate terms via `sort_and_combine()` where relevant). `s.multiply(rhs)` /
`s * rhs` is the full Cartesian outer product (`n*m` result terms);
`s.elementwise_multiply(rhs)` is the same-length pairwise product (`n`
terms) — pick whichever matches what you're computing, they are not
interchangeable. Free functions `commutator(a, b)` / `i_commutator(a, b)`
compute `[a,b]` / `i[a,b]` without forming the full product.

Gate methods on `PauliSum` use **common gate names** (the opposite
convention from `PauliString` above): `apply_H`, `apply_S`, `apply_CNOT`,
`apply_CZ`, `apply_CY`, plus `apply_H_YZ/XY`, `apply_sqrt_X/Y`,
`apply_X/Y/Z/I`, `apply_XC_X/Y/Z`, `apply_YC_X/Y/Z`, `apply_SWAP`,
`apply_iSWAP`, and the parametrized single-qubit rotations `apply_RZ(c,
theta)` / `apply_RX(c, theta)` / `apply_RY(c, theta)` (angle convention:
`theta` in `exp(-i*theta/2*P)`, i.e. standard `RZ`/`RX`/`RY` gate angles).
These are member-function only — no free-function wrappers.

`evolve(rotations)` is the general multi-term circuit-conjugation primitive:
each term of `rotations` is `(generator P_l, angle theta_l)` and is applied
as the Heisenberg conjugation of gate `exp(-i*theta_l/2*P_l)` — same angle
convention as `apply_RZ` etc., but for an arbitrary Pauli generator instead
of a fixed single-qubit axis.

Truncation & reduction primitives (all defined in `PauliSumUtils.h`):
`sort_and_combine()`, `canonicalize()` (folds sign into coefficient,
combines, **drops exact zeros** — `sort_and_combine()` alone does *not*
drop them), `trim(max_terms)` / `trim(max_terms, keep)`, `cutoff(eps)`,
`truncate_weight(max_weight)`, `term_weight(i)`, `frobenius_norm2()`, `trace()`, `trace_product(a, b)`,
`zero_state_expectation()`, `basis_state_expectation(bits)` (as
`zero_state_expectation()`, but `<x|·|x>` for an arbitrary computational
basis state `x`, given as a `'0'`/`'1'` string in qubit order).

`greedy_group()` partitions terms into mutually-commuting groups (useful for
VQE measurement grouping); it is OMP-thread-parallel, so the exact partition
can vary with thread count, but every returned partition is valid (each
group mutually commutes, every index appears exactly once).

## Common tasks

**Build a Hamiltonian and get its ground-state-adjacent expectation:**
```cpp
PauliSum<64, double> h(std::vector<std::pair<std::string, double>>{
    {"ZIII", -1.05}, {"IZII", -1.05}, {"XXII", 1.0}});
h.canonicalize();
double e = h.zero_state_expectation();  // <0...0| h |0...0>
```

**Trotterized circuit + expectation:**
```cpp
PauliSum<64, double> obs(4, "ZIII", 1.0);
PauliSum<64, double> layer(std::vector<std::pair<std::string, double>>{
    {"XXII", 0.3}, {"IIZZ", 0.7}});
obs.evolve(layer);
```

**Truncated Heisenberg backpropagation through a circuit
(`PauliPropagation.h`):**
```cpp
TruncationOptions<double> opts;
opts.max_weight = 4;
opts.max_terms  = 1u << 16;
double e = propagate_expectation(obs, circuit_rotations, opts);
```

**g-sim for circuits whose generators close under commutation
(`GSim.h`, needs Eigen):**
```cpp
auto gens = /* vector<PauliString<N>> */;
GSim<64> sim(gens);                       // throws if DLA exceeds max_dim
auto e = sim.zero_state_vector();
sim.apply_gate(e, sim.basis_index(PauliString<64>("XIII")), theta);
```

## Semantics & gotchas

- **Coefficients are real.** `T` is a real type; `i^k` phase lives in the
  `flags` byte (bit 0 = imaginary, bit 1 = sign). Reductions that must
  return a real number (`zero_state_expectation`, `basis_state_expectation`,
  `trace`, `frobenius_norm2`'s assumptions, `trace_product`) **silently skip
  terms whose imaginary-phase bit is set** rather than erroring — if a
  computation should be purely real but isn't, check for stray `i` terms
  rather than assuming a bug elsewhere.
- **`sort_and_combine()` vs `canonicalize()`.** `sort_and_combine()` merges
  duplicate `(pauli, phase)` keys by summing coefficients but leaves
  exact-cancellation results as an explicit zero-coefficient term (no
  pruning), and does **not** fold the sign bit into the coefficient (so `+X`
  and `-X` stay separate terms). `canonicalize()` does both: folds sign into
  the coefficient, combines, and drops exact zeros. Use `canonicalize()`
  unless you specifically need the raw sign-preserving behavior.
- **Angle convention.** `evolve()`, `apply_RZ`, `apply_RX`, `apply_RY` all
  take `theta` for gate `exp(-i*theta/2*P)` — the standard RZ/RX/RY gate
  angle (not the "already-halved" angle).
- **Gate naming asymmetry.** `PauliString<N>` gate names are physics names
  (`apply_H_XZ`, `apply_sqrt_Z`, `apply_ZC_X`); `PauliSum<N,T>` gate names
  are common names (`apply_H`, `apply_S`, `apply_CNOT`). See the tables
  above — this is intentional, not a bug, but easy to typo across the two
  APIs.
- **Exceptions on construction.** `PauliString(str)` throws
  `std::invalid_argument` if the string is empty or its operator-character
  count exceeds `N`; the `PauliSum`/`PauliString` `(qubits, ...)`
  constructors throw `std::invalid_argument` if `qubits > N`;
  `PauliString::set_pauli` throws `std::out_of_range` for an index `>=
  num_qubits()`. `trace_product` throws `std::invalid_argument` on a
  qubit-count mismatch between its two operands.
- **Determinism regardless of thread count.** `sort_and_combine`, `evolve`,
  the commutator, and the truncation passes are OpenMP-parallel above
  internal size thresholds, but use fixed-chunk-count parallel
  sort/merge/reduce strategies specifically so that output — including
  floating-point summation order — is identical for every
  `OMP_NUM_THREADS`. You do not need to pin thread count for reproducible
  results.
- **Threading is automatic.** There is nothing to opt into beyond linking
  OpenMP (already handled by the `paulib` CMake target); parallelism only
  activates above internal size thresholds, so small inputs run serially
  with no overhead.

## Optional layers

### GPU (CUDA)

Separate target `paulib_gpu`, built automatically when CMake detects
`nvcc` (targets `sm_90` — H100/GH200). Link both:
```cmake
target_link_libraries(your_target PRIVATE paulib paulib_gpu)
```
```cpp
#include "gpu/GpuPauliSum.h"
using namespace paulib;

GpuPauliSum<32> g_lhs(lhs_soa), g_rhs(rhs_soa);   // upload CPU SoA to HBM
GpuPauliSum<32> g_out = gpu_multiply(g_lhs, g_rhs);   // outer product on GPU
GpuPauliSum<32> g_comm = gpu_commutator(g_lhs, g_rhs, /*multiply_by_i=*/true);
PauliSum<32> result = g_out.download();
```
Constraints: `T = double` only; up to 1024 qubits (`ArraySize` 1–16); each
operand capped at `2^31-1` terms (throws `std::runtime_error` otherwise).
`gpu_commutator`'s output is uncombined and in nondeterministic order —
`canonicalize()`/`sort_and_combine()` after `download()` to get canonical
form.

### MPI

Optional, separately included — the core library has zero MPI dependency.
```cpp
#include "mpi/MpiPauliSum.h"
using namespace paulib;

auto sharded = MpiPauliSum<64>::from_replicated(h, MPI_COMM_WORLD);  // h replicated on every rank
auto next = mpi_i_commutator(H, sharded);   // H replicated, operand sharded
sharded.trim_global(max_terms);
PauliSum<64, double> full = sharded.gather();  // replicate back (tests/small sums)
```
Terms are owned by rank = hash of their `(x, z)` pattern (phase excluded, so
`+P`/`-P`/`iP` share an owner and combine locally). SPMD convention:
replicated inputs (`H`, `O0`, options) must be constructed identically on
every rank. `T = double` only. Link `MPI::MPI_CXX`; `mpi_sanity_check` is
built and registered as ctest entries (2 and 4 ranks) whenever an MPI
implementation is found.

## Reference

### Encoding

- `I` = X=0,Z=0 · `X` = X=1,Z=0 · `Z` = X=0,Z=1 · `Y` = X=1,Z=1
- Phase/sign in the `flags` byte: bit 0 = imaginary phase, bit 1 = sign;
  together they form the exponent `k` in `i^k` (0 = +1, 1 = +i, 2 = -1, 3 =
  -i).

### Naming conventions

- Types: PascalCase (`PauliString`, `PauliSum`).
- Functions: snake_case (`inner_product`, `sort_and_combine`, `apply_H_XZ`).
- Template params: `N` (qubit capacity), `T` (coefficient type).

### Headers

| Header | Purpose |
|--------|---------|
| `PauLib.h` | Single include for everything (CPU) |
| `PauliString.h` | Convenience include: PauliStringCore + Utils + Clifford |
| `PauliStringCore.h` | Core `PauliString<N>` struct |
| `PauliStringUtils.h` | String↔PauliString conversions |
| `PauliStringClifford.h` | Full single-string Clifford gate set: H (3 variants), S/SX/SY, X/Y/Z/I, the generalized controlled-Pauli family (CNOT, CZ, CY, XC_*, YC_*), SWAP/iSWAP |
| `PauliSum.h` | Convenience include: all SoA headers |
| `PauliSumCore.h` | Core `PauliSum<N, T>` struct, arithmetic operators (`+`, `-`, scalar `*`) |
| `PauliSumOperations.h` | Bulk multiply (`multiply`), `elementwise_multiply`, `evolve` (RZ/RX/RY), commutator |
| `PauliSumClifford.h` | Bulk (SoA/OMP) version of the same Clifford gate set as `PauliStringClifford.h` |
| `PauliSumUtils.h` | Utilities (string conversion, inner product, …) |
| `GSim.h` | g-sim: `lie_closure()` + `GSim<N,T>` Lie-algebraic simulator (uses Eigen) |
| `PauliPropagation.h` | `TruncationOptions` + `propagate()` / `propagate_expectation()`: truncated Heisenberg backpropagation through rotation circuits |
| `gpu/GpuPauliSum.h` | RAII HBM buffer manager + `gpu_multiply()` / `gpu_commutator()` — compiled by g++ |
| `gpu/paulib_gpu_api.h` | `extern "C"` kernel launcher declarations — safe for both g++ and nvcc |
| `mpi/MpiPauliSum.h` | Optional MPI layer: sharded `MpiPauliSum` (hash-owned terms), `mpi_i_commutator`, `trim_global`. Core library has no MPI dependency; link `MPI::MPI_CXX` |

## Building & testing PauLIB itself

```bash
cmake -S . -B build
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure -j "$(nproc)"
```

Dependencies: GCC/Clang with C++20. On Debian/Ubuntu:
```bash
sudo apt install -y build-essential cmake git
```
Submodules (GoogleTest, Eigen, Google Benchmark) must be initialized:
```bash
git submodule update --init --recursive
```
GPU support is auto-detected (nvcc on `PATH`); no extra CMake flags needed.
On Debian/Ubuntu: `sudo apt install -y nvidia-cuda-toolkit`.

Compiler flags: `-O3 -march=native -ffast-math` (the `-ffast-math` is
intentional — it enables vectorization of the floating-point loops). CUDA
kernel: `--use_fast_math -O3`, target `sm_90`.
