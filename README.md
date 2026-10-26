<p align="center">
  <img src="assets/logo.png" alt="PauLIB" width="420">
</p>

PauLIB is a C++ library for high-performance processing of large numbers
of Pauli strings, with GPU (CUDA) and MPI extensions and Python bindings.
The benchmark suite lives in [`bench/`](bench/).

## Build & test

```bash
git clone --recurse-submodules <repo-url> PauLIB
cd PauLIB
cmake -S . -B build
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure -j "$(nproc)"
```

Needs CMake 3.18+ and a C++20 compiler (GCC or Clang). A CUDA compiler on
`PATH` is auto-detected and enables the GPU target — no extra flags. For the
benchmark suite and Python bindings, see `CLAUDE.md`.

## Usage

```cpp
#include "PauLib.h"
using namespace paulib;

PauliSum<64, double> h(std::vector<std::pair<std::string, double>>{
    {"XXII", 1.0}, {"IZZI", -0.5}});
h.canonicalize();
double e = h.zero_state_expectation();
```

Full integration guide and API reference: [CLAUDE.md](CLAUDE.md).
