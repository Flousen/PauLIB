# PauLIB: A High-Performance Library for Processing Pauli Strings

C++20 library for high-performance Pauli string and Pauli sum operations used
in quantum computing (VQE, quantum algorithms). Core library is header-only;
GPU (CUDA) and MPI are optional, separately-compiled add-ons.

This document is written for integrating PauLIB into another project. For
instructions on building/testing PauLIB itself, see the end of this file.

## Integrating into a project

The core library is a header-only CMake `INTERFACE` target `paulib::paulib`
(see `src/CMakeLists.txt`). Either as a submodule:

```cmake
add_subdirectory(external/paulib)
target_link_libraries(your_target PRIVATE paulib::paulib)
```

or installed (`cmake --install build --prefix <prefix>`, then
`-DCMAKE_PREFIX_PATH=<prefix>` in your project):

```cmake
find_package(paulib 0.1 REQUIRED)            # COMPONENTS mpi gpu, if needed
target_link_libraries(your_target PRIVATE paulib::paulib)
```

The install puts the headers into `<prefix>/include/paulib/` (with the
bundled Eigen under `include/paulib/eigen/`) and the CMake package into
`<prefix>/lib/cmake/paulib/`; `paulib::mpi` and `paulib::gpu` are part of it
when they were built. As a subproject, PauLIB builds no tests, benchmarks or
demo app by default (`BUILD_BENCHMARKS`, `PAULIB_BUILD_APP`; tests only as
top-level project). The plain name `paulib` still works in-tree.

`paulib::paulib` propagates its include path, C++20, `-O3 -march=native
-ffast-math`, and `OpenMP::OpenMP_CXX`. Your target does not need to add
anything else for CPU use.

Requirements:
- GCC or Clang with C++20 support.
- OpenMP (`find_package(OpenMP REQUIRED)` — the top-level `CMakeLists.txt`
  and the installed `paulibConfig.cmake` already do this; consumers get it
  transitively).
- The `eigen` submodule (`external/eigen`): `PauLib.h` includes `GSim.h`,
  which uses Eigen. Installed along with the headers.
- Without CMake: it's plain headers, so adding `include/` and
  `external/eigen` to your include path and compiling with `-std=c++20
  -fopenmp` also works.

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
- **`T`** — coefficient type: `double` (normal case) or
  `std::complex<double>`. The core CPU library is generic in `T`, including
  complex `T` (see "Coefficients" below); `GSim` requires a real `T`, and
  the GPU and MPI layers hard-require `T = double` (enforced by
  `static_assert`). Angles, norms and cutoffs are always real:
  `real_t<T>` (`PauliScalar.h`) is `T` for real `T` and `double` for
  `std::complex<double>`.

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
`p.commute(other)`, `p.anti_commute(other)`, `symplectic_product(a, b)` (the
symplectic inner product of the binary representation: `true` = 1 = the
strings anticommute, i.e. the same as `anti_commute`; not the inner product of
two sums, which is `inner_product(A, B)` on `PauliSum`).

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
duplicate terms via `sort_and_combine()` where relevant). `+=`/`-=` are
`add_combine(rhs, c)` (`*this += c * rhs`): each operand is sorted on its
own — not at all if `is_sorted()` already holds — and the two are merged,
which gives append + `sort_and_combine()` bit for bit, but costs one pass
instead of a sort for a collated operand (the common case in `evolve`).
`adjoint(sum)` (A†: conjugated coefficients, sign flipped where the phase is
imaginary); qubit maps such as `remap_qubits` and `tensor_product` are in
`PauliQubitMap.h` (below). `s.multiply(rhs)` /
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
theta)` / `apply_RX(c, theta)` / `apply_RY(c, theta)`: the Heisenberg
conjugation `U† O U` by the standard gate `U = exp(-i*theta/2*P_c)`, the
same as `evolve` with generator `P_c` and angle `theta` (e.g. `RZ` maps `X`
to `cos(theta) X - sin(theta) Y`).
These are member-function only — no free-function wrappers.

`evolve(rotations)` is the general multi-term circuit-conjugation primitive:
each term of `rotations` is `(generator P_l, angle theta_l)` and is applied
as the Heisenberg conjugation `U† O U` of gate `exp(-i*theta_l/2*P_l)`,
with the standard gate angle, as `apply_RZ` etc.
`evolve_imaginary(generators)` is the imaginary-time step
`O -> e^{-tau P/2} O e^{-tau P/2}` per `(P_l, tau_l)` (evolve with
`theta = -i tau`): terms that **commute** with P become
`cosh(tau) Q - sinh(tau) P Q`, anticommuting terms stay — the opposite split
of a rotation. Not unitary (coefficients and trace grow); P must be
Hermitian.

Truncation & reduction primitives (all defined in `PauliSumUtils.h`):
`sort_and_combine()`, `canonicalize()` (folds the phase into the
coefficient — only the sign for real `T` — combines, **drops exact zeros**
— `sort_and_combine()` alone does *not* drop them), `trim(max_terms)` /
`trim(max_terms, keep)`, `cutoff(eps)`, `truncate_weight(max_weight)`,
`term_weight(i)`, `frobenius_norm2()`, `l1_norm()` (`sum |a_j|`, the
lambda of qubitization/qDRIFT), `weight_histogram()` (`sum |a_j|^2` per
weight), `trace()`, `trace_product(a, b)` (`tr(AB)`, no conjugation,
includes `2^n`), `inner_product(a, b)` (`tr(A†B)/2^n`, the normalized
Frobenius inner product; `trace_product` and `inner_product` use a merge
instead of a hash table when both sums are sorted, e.g. canonicalized),
`is_sorted()` (terms in the order of `sort_and_combine`),
`zero_state_expectation()`, `basis_state_expectation(bits)` (as
`zero_state_expectation()`, but `<x|·|x>` for an arbitrary computational
basis state `x`, given as a `'0'`/`'1'` string in qubit order),
`product_state_expectation(bloch)` (one Bloch vector `{rx, ry, rz}` per
qubit; `{0, 0, 1}` everywhere is the zero state).
`lookup(ps)` / `lookup("XZI")` returns the coefficient of a given Pauli
string in the sum (0 if absent). It sums all terms with that `(x, z)`
pattern, so the sum does not have to be combined first. Each term's phase
is taken relative to the query's phase (`{-X: 0.5}` gives `lookup("X") ==
-0.5`), and for real `T` terms that differ from the query by `±i` are
skipped (for complex `T` they contribute `±i` times their coefficient).

`greedy_group()` partitions terms into mutually-commuting groups (useful for
VQE measurement grouping); `greedy_group(Commutation::qubitwise)` requires
qubit-wise commutation instead (same letter or I on every qubit, so one
product basis measures a group; `XX` and `YY` commute but not qubit-wise).
It is OMP-thread-parallel, so the exact partition
can vary with thread count, but every returned partition is valid (each
group mutually commutes, every index appears exactly once).

### Commutation in bulk and noise (`PauliSumCommutation.h`)

Free functions. `anticommutes_mask(sum, P)` (`vector<uint8_t>`, 1 =
anticommutes), `commutes_with_all(sum, P)`, `anticommutes(sum, i, j)` (one
entry of the anticommutation matrix A = G Λ Gᵀ, O(W)). A is never stored: a
row is `anticommutes_mask(sum, P_j)`, or, word-wide and much cheaper for
low-weight terms, `anticommutes_mask(to_table(sum), P_j)`. Pauli channels in the Heisenberg picture rescale
coefficients only: `damp_anticommuting(sum, P, factor)` (adjoint of
`rho -> (1-p) rho + p P rho P` is `factor = 1 - 2p`),
`apply_pauli_channel(sum, q, px, py, pz)`, `apply_depolarizing(sum, q, p)`
(X/Y/Z on `q` damped by `1-p`, PauliPropagation.jl's convention),
`apply_dephasing(sum, q, p)` (X/Y damped by `1-p`), `add_noise(sum, p)`
(depolarizing on every qubit: factor `(1-p)^weight`), and the primitive
`damp_local(sum, q, {fI, fX, fZ, fY})`. `signed_norm2(sum, V)` is
`tr(A† V A V)/2^n = sum_j s_j |a_j|^2` (`s_j = -1` where term j anticommutes
with the Pauli `V`): the OTOC of a Hermitian `A = W(t)` without building
`V A V`; like `frobenius_norm2()` it assumes a collated sum.

### Writing kernels: skeletons (`PauliOps.h`, `PauliSkeletons.h`)

New term-wise kernels should use a skeleton instead of a hand-written OpenMP
loop. The skeleton handles the thread threshold, chunking and
reproducibility; the kernel supplies a lambda on a `TermView`:
```cpp
// value[i] = f(term)            (strings unchanged: order/collation kept)
transform_values(sum, [&](const TermView<N, T>& t) { return t.value() * 0.5; });
// keep terms where pred holds   (order kept)
filter_terms(sum, [](const TermView<N, T>& t) { return weight(t) <= 4; });
// mask[i] = pred(term)
auto m = term_mask(sum, [&](const TermView<N, T>& t) { return anticommutes(t, P); });
// sum of f(term); per-bin sums
auto r = reduce_terms<double>(sum, [](const TermView<N, T>& t) { return abs2(t.value()); });
auto h = reduce_terms_binned<double>(sum, bins, key, f);
```
Skeletons that change strings or create terms:
```cpp
// change the term in place through a TermRef (string, phase, value)
transform_terms(sum, [&](const TermRef<N, T>& t) { if (anticommutes(t, P)) t.flip_sign(); });
// 0..k new terms per term, returned in term order (evolve, evolve_imaginary)
auto added = expand_terms(sum, [&](const TermRef<N, T>& t, TermEmitter<N, T>& out) {
    out.push(t, t.value()).multiply(P); });
// one output term per pair, at index i*|b| + j (tensor_product)
auto out = for_each_pair(a, b, [](const TermView<N, T>& ta, const TermView<N, T>& tb,
                                  const TermRef<N, T>& o) { /* write o */ });
// sum of f over pairs with the same string, both sums sorted, O(|a|+|b|)
auto r = merge_reduce<T>(a, b, [](const TermView<N, T>& ta, const TermView<N, T>& tb) { ... });
```
`transform_terms` does not request vectorisation; `transform_terms<true>`
does, for branch-free bodies only (`adjoint`). On a branchy body the forced
request either fails (`-Wpass-failed`) or runs masked and slower than the
scalar loop (Clifford angles of `evolve`: 3.3 vs 1.3 µs on 1024 terms).
`TermView` gives `x(k)`, `z(k)`, `flags()`, `value()`, `index()`; `TermRef`
additionally `x_word(k)`, `z_word(k)`, `flags_ref()`, `value_ref()`,
`flip_sign()`, `multiply(ps)`. A `TermRef` reads through the `PauliSum`;
a version with raw base pointers (like `SumView`) was tried on 9 Nov 2026
and reverted: faster with GCC, but with Clang 22 on the M4 Pro the Clifford
angles of `evolve` took 4.8 instead of 1.3 µs on 1024 terms (also with the
pointers held by value). The primitives `anticommutes(a, b)`,
`qubitwise_commutes(a, b)`, `weight`, `has_x`, `is_identity`,
`local_pauli(t, word, bit)` work on
TermView, TermRef and PauliString; `contributes(t)` / `weighted_value(t)`
(phase handling of reductions) on TermView.
The reductions are **bit-identical for every thread count**: fixed-length
chunks (`ExecPolicy::chunk_terms`, default 2^12) summed in order, chunk
results combined in chunk order. Threads start only above
`ExecPolicy::parallel_threshold` (default 2^15 terms), by a real branch (an
OpenMP `if` clause still starts a team). `default_policy()` is process-wide;
tests lower both values (`tests/KernelConformance.h`, which also checks a
kernel for bit-identity with 1, 3 and 8 threads). Do not put
`#pragma omp simd` on term loops over a `TermView`: it is lowered before
inlining and keeps the loop scalar. The skeletons mark their loops with
`PAULIB_INDEPENDENT_ITERATIONS` (`clang loop vectorize(assume_safety)` for
Clang, `GCC ivdep` for GCC; each compiler ignores the other's pragma).

Pitfalls found in October 2026 (none of them fails a test, each cost 10 %
to a factor of 3):
- A vectorisation pragma for one compiler only is silently ignored by the
  other (`GCC ivdep` under Clang).
- `#pragma omp parallel for simd` around a lambda body: Clang does not
  vectorise the outlined loop. Use a parallel loop over blocks with
  `omp simd` inside (`detail::table_parallel_words` in `PauliTable.h`).
- `#pragma omp parallel for if(cond)` costs about 0.1 us per call even when
  `cond` is false (GCC/libgomp 116 ns, Clang/libomp 56 ns, against 7-17 ns
  for an explicit branch). Use an explicit `if`/`else` where the region runs
  per gate or per pivot on small data (`gf2_rref` was 3.3 times slower on a
  20 x 40 matrix). `evolve` and `canonicalize` run on skeletons since
  29 Oct 2026; the clause remains only in `PauliTable` conversions and GSim.
- A forced vectorisation request on a body that cannot vectorise gives a
  `-Wpass-failed` warning only in some translation units (it depends on
  inlining): check the full benchmark TU, not each kernel alone. On a body
  with a branch that does vectorise, the masked loop can be slower than the
  scalar one: measure before forcing.
- Check: build with Clang and `-Wunknown-pragmas -Wpass-failed` (lists every
  requested vectorisation that failed) and with GCC and
  `-fopt-info-vec-missed`. Known and harmless: `cutoff` with a complex `T`
  (`std::abs` calls `hypot`).

### Linear algebra over GF(2) (`PauliGF2.h`)

No separate bit matrix: the terms of a `PauliSum` are the rows `(x_j | z_j)`
of G (column c < n = X bit of qubit c, column n + c = Z bit), and the
elimination runs on the X/Z word arrays of a copy of the sum (phases and
coefficients ignored). `gf2_rref(sum)` (in place, returns pivot columns;
terms rank.. become the identity; flags/values no longer describe the
strings), `gf2_rank(sum)`, `gf2_row_basis(sum)` and `gf2_nullspace(sum)`
(basis of `{v : G v = 0}`, ordinary dot product) return `PauliSum`s with
phase +1 and coefficient 1. `GF2Basis<N>(n, max_inputs)`: incremental
reduced basis of `PauliString`s that records which inputs combine to a basis
string. Pauli level: `pauli_rank(sum)` / `pauli_rank(strings, n)`,
`independent_subset(strings, n)` (indices, greedy in input order),
`in_span(strings, P, n)` / `in_span(sum, P)`, `commutant(sum)` (a
`PauliSum`) / `commutant(strings, n)` (a vector): basis of all strings
commuting with every input; dimension `2n - rank`; on a Hamiltonian these
are its Z2 symmetries for tapering), and
`stabilizer_expectation(sum, generators)` (`tr(rho A)` for the stabilizer
state of commuting, independent, Hermitian generators with signs; fewer
than `n` generators give the maximally mixed code state; throws
`std::invalid_argument` for invalid generators).

### Generation (`PauliGenerate.h`)

- Random: `random_pauli_string<N>(n, rng)` (uniform over `4^n`, phase +1),
  `TermSampler sampler(sum)` (`sample(rng)` draws index `j` with
  probability `|a_j|/lambda`, `lambda()`; the qDRIFT draw),
  `sample_terms(sum, k, seed)`, `random_clifford(n, rng)` (uniform
  Clifford as a `CliffordCircuit`, van den Berg sweep, O(n²) gates).
- Structured: `strings_up_to_weight<N, T>(n, k, max_terms)`,
  `chain_edges(n, periodic)`, `square_lattice_edges(Lx, Ly, periodic)`,
  `two_site_sum<N, T>(n, edges, "ZZ", J)`, `one_site_sum<N, T>(n, "X", h)`
  (compose models with `+`), Jordan–Wigner `jw_majorana<N>(n, j)`,
  `jw_annihilation<N, T>(n, m)`, `jw_creation<N, T>(n, m)` (the `±i` lives
  in the phase flags, so real `T` works), and Bravyi–Kitaev
  `bk_majorana<N>(n, j)`, `bk_annihilation`, `bk_creation` (Fenwick-tree
  sets of OpenFermion's `bravyi_kitaev`, any n; weight O(log n);
  `a_m = (gamma_2m + i gamma_2m+1)/2` as for JW).
- `CliffordCircuit` = `vector<CliffordGate>` with kinds `H, S, SDG, X, Y,
  Z, CNOT(control, target), CZ, SWAP` in application order.
  `apply_circuit(p, circuit)` works on `PauliString`, `PauliSum` and
  `PauliTable` and computes the Heisenberg `U† P U` (gates in reverse
  order, each with the `apply_*` convention `P -> G† P G`);
  `apply_circuit(p, adjoint(circuit))` gives `U P U†`.

### Qubit-major table (`PauliTable.h`)

`PauliTable` is the transposed (Stim-style) layout for gate-heavy work on a
fixed set of strings: `xbits`, `zbits` hold n rows of `words()` words,
**row = qubit** (`x_row(q)`, `z_row(q)`, `get_x/set_x(q, j)`), bit `j` =
term `j`; phases in the bit planes `imag`, `sign` (bits 0 and 1 of
the flags byte). No coefficients — keep them by term index. Runtime sizes,
not templated on `N`.
```cpp
PauliTable t = to_table(sum);              // 64x64 bit transposes
apply_circuit(t, circuit);                 // same gates/convention as PauliSum
auto back = to_sum<64, double>(t, sum.value);
```
Gates `apply_H/S/H_YZ/H_XY/sqrt_X/sqrt_Y/X/Y/Z/I/CNOT/CZ/SWAP` update 64
terms per word (`P -> G† P G`, as on `PauliSum`), plus `apply_S_dag`,
`apply_sqrt_X_dag`, `apply_sqrt_Y_dag` (`P -> G P G†`); ~90× the `PauliSum` gate
rate on one thread. `get<N>(j)`/`set(j, ps)`/`copy_term` are O(n) bit
operations. `anticommutes_mask(table, ps)` returns a bit vector over terms.
`left_multiply(mask, P or term index)`: term_j ← P·term_j for the masked
terms at once. `product(mask, px, pz)`: ordered product of the masked terms,
returns the phase exponent. Threads only from `detail::kTableParallelWords`
= 2^18 words per row (below, starting a team costs more than the gate).
`to_table`/`to_sum` are slow relative to the gates (allocation-bound) — convert
once. `transpose64` (64 × 64 bit block) lives in `PauliTable.h`.

### Stabilizer tableau (`Tableau.h`)

`Tableau<N> tab(n)` — CHP (Aaronson–Gottesman) on a `PauliTable` with 2n
rows (destabilizers `0..n-1`, stabilizers `n..2n-1`), starting in `|0…0>`.
The tableau evolves the **state**: its gates map every row `P -> G P G†`
(Schrödinger picture, the inverse of the `PauliTable` gates for S, √X, √Y),
and `apply_circuit(tab, c)` (own overload) runs the gates in circuit order,
giving `U|0…0>`. `tab.measure_z(q, rng)` →
`MeasureResult{outcome, deterministic}` (`outcome = true` means `-1`/`|1>`),
`tab.reset(q, rng)`, `tab.peek(P)` → `+1/-1/0` for a Hermitian `P` (throws
otherwise), `stabilizers()`, `destabilizers()`. Z measurements only; measure
X/Y via a basis change.

### Qubit maps (`PauliQubitMap.h`)

QMP of the thesis: change the qubits on which the strings act, not the
letters. Letters are stored directly (Y = bits x = z = 1), so moving or
dropping a letter changes no phase. None of them combines; `restrict_qubits`
and `fix_qubits` can create duplicates (`canonicalize()` afterwards).
- `remap_qubits(sum, n_out, map)`: qubit q → `map[q]`; `kDropQubit` removes
  a qubit that is the identity in every term (throws otherwise); embedding,
  permutation. Bit loop over the support of each term (~30 M terms/s).
- `shift_qubits(sum, k)`: cyclic q → (q + k) mod n, k may be negative (the
  translation T_k of PauliStrings.jl). `shift_lattice(sum, Lx, Ly, dx, dy)`:
  periodic Lx × Ly lattice, qubit x + Lx·y; both word-wide (masked
  multi-word shifts, no loop over qubits).
- `restrict_qubits(sum, qubits)`: `qubits[j]` becomes qubit j, all other
  letters are dropped (P_I ⊗ P_rest → P_I; Sun et al. T_I, contextual
  subspace). To keep only terms supported on a set (TRN), use `filter_terms`.
- `fix_qubits(sum, {{q, 'X'|'Y'|'Z', ±1}, ...})`: every term must act on q
  with I or that Pauli (throws otherwise); the Pauli is replaced by its
  eigenvalue (sign bit) and q is removed; the last step of tapering.
- `tensor_product(a, b)`: A ⊗ B, the qubits of `a` first, `|a|·|b|` terms.

### Gates and channels on k qubits (`PauliLocalMap.h`)

General non-Clifford conjugation (ROT beyond Pauli rotations): a gate or
channel on k ≤ 4 qubits maps a string to up to 4^k strings that differ only
there. `heisenberg_ptm(kraus, k)` builds the Pauli transfer matrix of
`E†(P) = Σ K† P K` (a gate U: `{U}`; amplitude damping: two Kraus
operators) from row-major 2^k × 2^k `DenseMatrix`es, bit j of the basis
index = `qubits[j]`; entries below 1e-14 are zeroed.
`apply_ptm(sum, qubits, ptm)` applies it in place, with the convention of
`evolve` (U† O U): the term stays in place scaled by the diagonal entry
(removed if 0), the other images are new terms merged by `add_combine`, so a
combined sum stays combined and the result is thread-count independent.
Local index: digit j (2 bits) is the letter on `qubits[j]`, coded 0 = I,
1 = X, 2 = Z, 3 = Y; `ptm.at(out, in)`. A CCZ maps `X` on a target to four
strings; a Pauli rotation as PTM gives `evolve`'s result up to rounding.

### Qubit tapering (`PauliTapering.h`)

`z2_symmetries(h)`: independent, mutually commuting strings commuting with
every term (commutant basis, kept greedily if they commute with those kept).
`Z2Tapering<N> tap(symmetries, sector, n)` (sector values ±1, the
eigenvalues of the symmetries) builds a Clifford `U` (`tap.clifford()`,
H/S/CNOT sweep) with `U s_i tau_i U† = ±Z_{q_i}`; `tap.apply(op)` returns
`U op U†` with each `Z_{q_i}` replaced by its eigenvalue and the qubits
`q_i` removed (`tap.qubits()`, via `fix_qubits`), canonicalized, on `tap.num_qubits_out()`
qubits; its spectrum is that of `op` in the sector. Throws
`std::invalid_argument` for anticommuting/dependent symmetries or an `op`
that does not commute with them.

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

- **Coefficients.** A term is `value * i^k * P` with `k` in the `flags`
  byte (bit 0 = imaginary, bit 1 = sign). With a **real `T`**, reductions
  returning `T` (`zero_state_expectation`, `basis_state_expectation`,
  `product_state_expectation`, `stabilizer_expectation`, `trace`, `lookup`)
  **silently skip terms whose contribution is imaginary**, and
  `trace_product`/`inner_product` return the real part — if a
  computation should be purely real but isn't, check for stray `i` terms.
  With **`T = std::complex<double>`** nothing is skipped, and
  `canonicalize()` folds the whole `i^k` into the coefficient, so every
  flag is 0 afterwards and `P` and `iP` merge into one term.
  `frobenius_norm2()`, `l1_norm()` and `weight_histogram()` return
  `real_t<T>`. The scalar `*` operators deduce `T` from both sides, so
  write `s * std::complex<double>(2.0)`, not `s * 2.0`, for complex `T`.
- **Conjugation direction: Heisenberg everywhere.** The Clifford `apply_*`
  gates, `apply_circuit`, `evolve` and `apply_RZ/RX/RY` all map
  `P -> U† P U`, so `apply_S(q)` equals `apply_RZ(q, π/2)` (X → -Y, Y → X),
  `apply_sqrt_X`/`apply_sqrt_Y` equal `apply_RX`/`apply_RY(q, π/2)`.
  Before October 2026 the Clifford gates used `G P G†`; only S, √X, √Y and
  iSWAP change (the other gates are self-inverse). The one exception is
  `Tableau`, which evolves a state (`G P G†`, circuit order). For a
  stabilizer state `U|0>` given as strings, use
  `apply_circuit(Z_j, adjoint(c))`.
- **`sort_and_combine()` vs `canonicalize()`.** `sort_and_combine()` merges
  duplicate `(pauli, phase)` keys by summing coefficients but leaves
  exact-cancellation results as an explicit zero-coefficient term (no
  pruning), and does **not** fold the sign bit into the coefficient (so `+X`
  and `-X` stay separate terms). `canonicalize()` does both: folds sign into
  the coefficient, combines, and drops exact zeros. Use `canonicalize()`
  unless you specifically need the raw sign-preserving behavior.
- **Angle convention.** `evolve()`, `apply_RZ`, `apply_RX`, `apply_RY` all
  take `theta` for gate `exp(-i*theta/2*P)` — the standard RZ/RX/RY gate
  angle. (Before October 2026, `apply_RX/RY/RZ` used `theta/2`.)
- **Two "inner products".** `PauliString::symplectic_product` is the
  commutation check (returns `true` for anticommuting strings);
  `inner_product(A, B)` on two `PauliSum`s is `tr(A†B)/2^n`. The former
  `PauliString::inner_product` (which returned `true` for *commuting*
  strings) was renamed and its boolean inverted.
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
  the commutator, the truncation passes and every kernel on a skeleton
  (norms, traces, basis- and zero-state expectations, `weight_histogram`,
  `signed_norm2`, the noise channels) are OpenMP-parallel above
  internal size thresholds, but use fixed-chunk-count parallel
  sort/merge/reduce strategies (`evolve_imaginary`, `add_combine`,
  `adjoint`, the qubit maps of `PauliQubitMap.h`, `apply_ptm` and the merge paths of
  `inner_product`/`trace_product` included) specifically so that output — including
  floating-point summation order — is identical for every
  `OMP_NUM_THREADS`. You do not need to pin thread count for reproducible
  results.
- **Threading is automatic.** There is nothing to opt into beyond linking
  OpenMP (already handled by the `paulib` CMake target); parallelism only
  activates above internal size thresholds, so small inputs run serially
  with no overhead.

## Optional layers

### GPU (CUDA)

Separate target `paulib::gpu` (in-tree also `paulib_gpu`), built
automatically when CMake detects `nvcc` (targets `sm_90` — H100/GH200).
Link both:
```cmake
target_link_libraries(your_target PRIVATE paulib::paulib paulib::gpu)
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
every rank. `T = double` only. Link `paulib::mpi` (= `paulib` +
`MPI::MPI_CXX`); `mpi_sanity_check` is
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
- Functions: snake_case (`symplectic_product`, `sort_and_combine`, `apply_H_XZ`).
- Template params: `N` (qubit capacity), `T` (coefficient type).

### Headers

| Header | Purpose |
|--------|---------|
| `PauLib.h` | Single include for everything (CPU) |
| `PauliScalar.h` | Coefficient traits: `real_t<T>`, `is_complex_v<T>`, `abs2`, `phase_factor<T>(flags)` |
| `PauliString.h` | Convenience include: PauliStringCore + Utils + Clifford |
| `PauliStringCore.h` | Core `PauliString<N>` struct |
| `PauliStringUtils.h` | String↔PauliString conversions |
| `PauliStringClifford.h` | Full single-string Clifford gate set: H (3 variants), S/SX/SY, X/Y/Z/I, the generalized controlled-Pauli family (CNOT, CZ, CY, XC_*, YC_*), SWAP/iSWAP |
| `PauliSum.h` | Convenience include: all SoA headers |
| `PauliSumCore.h` | Core `PauliSum<N, T>` struct, arithmetic operators (`+`, `-`, scalar `*`) |
| `PauliSumOperations.h` | Bulk multiply (`multiply`), `elementwise_multiply`, `evolve` (RZ/RX/RY), `evolve_imaginary`, commutator, collate (`sort_and_combine`, `add_combine`, `is_sorted`), `adjoint`, `greedy_group` |
| `PauliQubitMap.h` | Qubit maps (QMP): `remap_qubits`, `shift_qubits`, `shift_lattice`, `restrict_qubits`, `fix_qubits`, `tensor_product` |
| `PauliLocalMap.h` | Gates and channels on k ≤ 4 qubits: `heisenberg_ptm`, `apply_ptm` (non-Clifford conjugation beyond Pauli rotations) |
| `PauliSumClifford.h` | Bulk (SoA/OMP) version of the same Clifford gate set as `PauliStringClifford.h` |
| `PauliSumUtils.h` | Utilities (string conversion, inner product, …) |
| `PauliOps.h` | `SumView`, `TermView`, `TermRef` and the word-level primitives (`anticommutes`, `qubitwise_commutes`, `weight`, `has_x`, `is_identity`, `local_pauli`) shared by all kernels |
| `PauliSkeletons.h` | `ExecPolicy`, `default_policy()`, skeletons `transform_values`, `term_mask`, `filter_terms`, `reduce_terms`, `reduce_terms_binned`, `transform_terms`, `expand_terms`, `for_each_pair`, `merge_reduce` (reproducible for every thread count) |
| `PauliSumCommutation.h` | COM in bulk (`anticommutes_mask`, `anticommutes(sum, i, j)`) and Pauli-channel damping (`apply_pauli_channel`, `apply_depolarizing`, `add_noise`, …), `signed_norm2` (OTOC) |
| `PauliGF2.h` | GF(2) linear algebra on the word arrays of a `PauliSum`: `gf2_rref/rank/row_basis/nullspace`, `GF2Basis<N>`, `commutant`, `in_span`, `stabilizer_expectation` |
| `PauliGenerate.h` | Generation: random strings, `TermSampler`, `random_clifford` + `CliffordCircuit`/`apply_circuit`, weight-k enumeration, lattice Hamiltonians, Jordan–Wigner, Bravyi–Kitaev |
| `PauliTapering.h` | Qubit tapering: `z2_symmetries`, `Z2Tapering<N>` (Clifford, removed qubits, `apply`) |
| `PauliTable.h` | Qubit-major `PauliTable` (bits + phase), `to_table`/`to_sum`, `transpose64`, word-wide Clifford gates, `anticommutes_mask`, `left_multiply`, `product` |
| `Tableau.h` | `Tableau<N>`: CHP stabilizer simulator with `measure_z`, `reset`, `peek` |
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

### Python bindings

`pyproject.toml` (scikit-build-core + pybind11) builds the `paulib` Python
package from the repo root; the version comes from `project()` in
`CMakeLists.txt`:
```bash
pip install .            # or: pip install -e . for development
python -c "import paulib; print(paulib.MAX_QUBITS)"
```
It configures CMake with `BUILD_PYTHON_BINDINGS=ON` and tests, benchmarks
and app off, and installs only the CMake component `python` (extension +
`python/paulib/__init__.py`), no C++ headers. Needs the Python development
headers (`python3-dev`) and the Eigen submodule. Compiler and OpenMP
location go through `CMAKE_ARGS`, e.g. on macOS with Homebrew LLVM:
`CMAKE_ARGS="-DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++
-DOpenMP_ROOT=$(brew --prefix libomp)" pip install .`. The extension is
compiled with `-march=native`, so a wheel only runs on CPUs like the build
machine. The bindings instantiate 16 capacities (64 … 1024 qubits), so the
build takes several minutes.

Without pip: `cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON` writes the
extension directly into `python/paulib/`, importable with
`PYTHONPATH=python`. After `pip install .`, do not run scripts that sit
directly in `python/`: that directory then comes first on `sys.path`, and
`python/paulib/` (without the extension) hides the installed package.
Example: `python/examples/example.py`.
