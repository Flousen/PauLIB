#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <PauLib.h>

#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace paulib;

// ---------------------------------------------------------------------------
// The classes are instantiated once per 64-bit word count (capacity N = 64,
// 128, ..., 1024); the actual qubit count is a runtime value tracked in
// num_qubits_. The Python factories in paulib/__init__.py pick the smallest
// capacity that fits and the objects report/print their logical size.
// ---------------------------------------------------------------------------

namespace {

template <typename Obj>
void check_qubit(const Obj& o, std::size_t q) {
    if (q >= o.num_qubits())
        throw py::index_error("qubit index " + std::to_string(q) +
                              " out of range for " +
                              std::to_string(o.num_qubits()) + " qubits");
}

template <typename PSS>
void check_term(const PSS& s, std::size_t i) {
    if (i >= s.get_length())
        throw py::index_error("term index " + std::to_string(i) +
                              " out of range for " +
                              std::to_string(s.get_length()) + " terms");
}

template <typename A, typename B>
void check_same_qubits(const A& a, const B& b) {
    if (a.num_qubits() != b.num_qubits())
        throw py::value_error("qubit count mismatch: " +
                              std::to_string(a.num_qubits()) + " vs " +
                              std::to_string(b.num_qubits()));
}

} // namespace

// ---------------------------------------------------------------------------
// Per-capacity registration helpers
// ---------------------------------------------------------------------------

template <std::size_t N>
void register_pauli_string(py::module_& m) {
    using PS = PauliString<N>;
    std::string name = "PauliString_" + std::to_string(N);

    py::class_<PS>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("s"))
        .def("num_qubits", &PS::num_qubits)
        .def("inner_product", [](const PS& a, const PS& b) { check_same_qubits(a, b); return a.inner_product(b); }, py::arg("other"))
        .def("commute",      [](const PS& a, const PS& b) { check_same_qubits(a, b); return a.commute(b); },       py::arg("other"))
        .def("anti_commute", [](const PS& a, const PS& b) { check_same_qubits(a, b); return a.anti_commute(b); },  py::arg("other"))
        .def("multiply",     [](PS& a, const PS& b) { check_same_qubits(a, b); a.multiply(b); },                   py::arg("other"))
        .def("get_sign",  &PS::getSign)
        .def("get_phase", &PS::getPhase)
        .def("set_sign",  &PS::setSign,  py::arg("v"))
        .def("set_phase", &PS::setPhase, py::arg("v"))
        .def("set_pauli", &PS::set_pauli, py::arg("qubit"), py::arg("p"),
             "Set qubit to 'X', 'Y', 'Z', or 'I'.")
        .def("to_string", &PS::to_string)
        .def("__str__",   &PS::to_string)
        .def("__repr__",  [](const PS& ps) {
            return "PauliString(\"" + ps.to_string() + "\")";
        })
        // Clifford gates
        .def("apply_H",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_H_XZ(q); },   py::arg("qubit"))
        .def("apply_H_YZ", [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_H_YZ(q); },   py::arg("qubit"))
        .def("apply_H_XY", [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_H_XY(q); },   py::arg("qubit"))
        .def("apply_S",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_sqrt_Z(q); }, py::arg("qubit"))
        .def("apply_SX",   [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_sqrt_X(q); }, py::arg("qubit"))
        .def("apply_SY",   [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_sqrt_Y(q); }, py::arg("qubit"))
        .def("apply_X",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_X(q); },      py::arg("qubit"))
        .def("apply_Y",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_Y(q); },      py::arg("qubit"))
        .def("apply_Z",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_Z(q); },      py::arg("qubit"))
        .def("apply_I",    [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.apply_I(q); },      py::arg("qubit"))
        .def("apply_CNOT", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_ZC_X(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_CZ",   [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_ZC_Z(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_CY",   [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_ZC_Y(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_X", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_XC_X(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_Y", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_XC_Y(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_Z", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_XC_Z(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_X", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_YC_X(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_Y", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_YC_Y(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_Z", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_YC_Z(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_SWAP", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_SWAP(c, t);
            }, py::arg("qubit1"), py::arg("qubit2"))
        .def("apply_iSWAP", [](PS& ps, std::size_t c, std::size_t t) {
                check_qubit(ps, c); check_qubit(ps, t); ps.apply_iSWAP(c, t);
            }, py::arg("qubit1"), py::arg("qubit2"))
        // Phase-multiply helpers
        .def("multiply_iX", [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.multiply_iX(q); }, py::arg("qubit"))
        .def("multiply_iZ", [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.multiply_iZ(q); }, py::arg("qubit"))
        .def("multiply_iY", [](PS& ps, std::size_t q) { check_qubit(ps, q); ps.multiply_iY(q); }, py::arg("qubit"))
        // Commutativity with single-qubit Paulis
        .def("commute_with_Z", [](const PS& ps, std::size_t q) { check_qubit(ps, q); return ps.commute_with_Z(q); }, py::arg("qubit"))
        .def("commute_with_X", [](const PS& ps, std::size_t q) { check_qubit(ps, q); return ps.commute_with_X(q); }, py::arg("qubit"))
        .def("commute_with_Y", [](const PS& ps, std::size_t q) { check_qubit(ps, q); return ps.commute_with_Y(q); }, py::arg("qubit"))
        // Operators
        .def("__mul__",  [](const PS& a, const PS& b) { check_same_qubits(a, b); return a * b; })
        .def("__imul__", [](PS& a, const PS& b) -> PS& { check_same_qubits(a, b); a *= b; return a; })
        .def("__eq__", [](const PS& a, const PS& b) {
            return a.num_qubits() == b.num_qubits() &&
                   a.x == b.x && a.z == b.z && a.flags == b.flags;
        });
}

template <std::size_t N>
void register_pauli_sum(py::module_& m) {
    using PSS = PauliSum<N, double>;
    using PS  = PauliString<N>;
    std::string name = "PauliSum_" + std::to_string(N);

    py::class_<PSS>(m, name.c_str())
        .def(py::init([](std::size_t num_qubits) {
                if (num_qubits > N)
                    throw py::value_error("qubit count exceeds capacity " + std::to_string(N));
                PSS s;
                s.num_qubits_ = num_qubits;
                return s;
            }), py::arg("num_qubits") = N)
        .def(py::init<const std::vector<std::string>&>(), py::arg("pauli_strings"))
        .def(py::init<const std::vector<std::pair<std::string, double>>&>(), py::arg("terms"))
        // Properties
        .def_property_readonly("coefficients", [](const PSS& ps) { return ps.value; })
        // Accessors
        .def("num_qubits",       &PSS::num_qubits)
        .def("__len__",          &PSS::get_length)
        .def("get_str",          &PSS::get_str)
        .def("__str__",          &PSS::get_str)
        .def("__repr__",         [](const PSS& ps) { return ps.get_str(); })
        .def("get_pauli_string", [](const PSS& s, std::size_t i) { check_term(s, i); return s.get_pauli_string(i); }, py::arg("i"))
        .def("__getitem__", [](const PSS& ps, std::size_t i) -> std::pair<PS, double> {
            check_term(ps, i);
            return {ps.get_pauli_string(i), ps.value[i]};
        })
        // Mutators
        .def("reserve",   &PSS::reserve, py::arg("capacity"))
        .def("push_back", [](PSS& s, const PS& ps, double coeff) {
                if (ps.num_qubits() > s.num_qubits())
                    throw py::value_error("Pauli string has more qubits than the sum");
                s.push_back(ps, coeff);
            }, py::arg("pauli_string"), py::arg("coefficient") = 1.0)
        .def("collate", &PSS::sort_and_combine)
        // Multiplication
        .def("multiply", [](PSS& a, const PSS& b) { check_same_qubits(a, b); a.multiply(b); },
             py::arg("rhs"), "Multiply this Pauli sum by rhs (polynomial product).")
        .def("elementwise_multiply", [](PSS& a, const PSS& b) {
                check_same_qubits(a, b);
                if (a.get_length() != b.get_length())
                    throw py::value_error("elementwise_multiply requires equal lengths");
                a.elementwise_multiply(b);
            }, py::arg("rhs"),
             "Element-wise product: result[i] = self[i] * rhs[i]. Both sums must have equal length.")
        .def("multiply_at", [](PSS& s, std::size_t i, const PS& ps) {
                check_term(s, i); check_same_qubits(s, ps); s.multiply(i, ps);
            }, py::arg("i"), py::arg("pauli_string"),
             "Left-multiply the i-th term by a PauliString.")
        // Time evolution
        .def("evolve", [](PSS& s, const PSS& rot) { check_same_qubits(s, rot); s.evolve(rot); },
             py::arg("rotations"),
             "Apply a sequence of Pauli rotations (evolve under exp(-i*theta*P)).")
        // Commutativity
        .def("commute", [](PSS& s, std::size_t i, const PS& ps) {
                check_term(s, i); check_same_qubits(s, ps); return s.commute(i, ps);
            }, py::arg("i"), py::arg("pauli_string"))
        .def("commute", [](PSS& s, std::size_t i, std::size_t j) {
                check_term(s, i); check_term(s, j); return s.commute(i, j);
            }, py::arg("i"), py::arg("j"))
        .def("commute_with_Z", [](const PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); return s.commute_with_Z(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("commute_with_X", [](const PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); return s.commute_with_X(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("commute_with_Y", [](const PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); return s.commute_with_Y(i, q); }, py::arg("i"), py::arg("qubit"))
        // Pauli setters
        .def("set_Z", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.set_Z(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("set_X", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.set_X(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("set_Y", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.set_Y(i, q); }, py::arg("i"), py::arg("qubit"))
        // Phase-multiply helpers
        .def("multiply_iX", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.multiply_iX(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("multiply_iZ", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.multiply_iZ(i, q); }, py::arg("i"), py::arg("qubit"))
        .def("multiply_iY", [](PSS& s, std::size_t i, std::size_t q) { check_term(s, i); check_qubit(s, q); s.multiply_iY(i, q); }, py::arg("i"), py::arg("qubit"))
        // Expectation value in the |0^N> state
        .def("zero_state_expectation", &PSS::zero_state_expectation,
             "<0^N|H|0^N>: sum of coefficients of purely I/Z terms.")
        // Trace / norm
        .def("trace", &PSS::trace,
             "Matrix trace: 2^n * sum of identity-term coefficients.")
        .def("frobenius_norm2", &PSS::frobenius_norm2,
             "tr(O†O)/2^n for a canonicalized sum (sum of squared coefficients).")
        // Truncation & algebra primitives
        .def("term_weight", [](const PSS& s, std::size_t i) { check_term(s, i); return s.term_weight(i); },
             py::arg("i"), "Number of non-identity sites of term i.")
        .def("canonicalize", &PSS::canonicalize,
             "Fold sign flags into coefficients, combine duplicates, drop zeros.")
        .def("trim", [](PSS& s, std::size_t m) { s.trim(m); },
             py::arg("max_terms"), "Keep only the max_terms largest-|coefficient| terms.")
        .def("trim", [](PSS& s, std::size_t m, const PSS& keep) { check_same_qubits(s, keep); s.trim(m, keep); },
             py::arg("max_terms"), py::arg("keep"),
             "trim(), but strings appearing in `keep` are never dropped.")
        .def("cutoff", &PSS::cutoff, py::arg("eps"),
             "Drop terms with |coefficient| < eps.")
        .def("truncate_weight", &PSS::truncate_weight, py::arg("max_weight"),
             "Drop terms acting on more than max_weight qubits.")
        .def("scale", &PSS::scale, py::arg("c"), "Multiply all coefficients by c.")
        // Clifford gates
        .def("apply_H",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_H(q); }, py::arg("qubit"))
        .def("apply_H_YZ", [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_H_YZ(q); }, py::arg("qubit"))
        .def("apply_H_XY", [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_H_XY(q); }, py::arg("qubit"))
        .def("apply_S",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_S(q); }, py::arg("qubit"))
        .def("apply_SX",   [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_sqrt_X(q); }, py::arg("qubit"))
        .def("apply_SY",   [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_sqrt_Y(q); }, py::arg("qubit"))
        .def("apply_X",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_X(q); }, py::arg("qubit"))
        .def("apply_Y",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_Y(q); }, py::arg("qubit"))
        .def("apply_Z",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_Z(q); }, py::arg("qubit"))
        .def("apply_I",    [](PSS& s, std::size_t q) { check_qubit(s, q); s.apply_I(q); }, py::arg("qubit"))
        .def("apply_CNOT", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_CNOT(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_CZ",   [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_CZ(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_CY",   [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_CY(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_X", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_XC_X(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_Y", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_XC_Y(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_XC_Z", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_XC_Z(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_X", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_YC_X(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_Y", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_YC_Y(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_YC_Z", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_YC_Z(c, t);
            }, py::arg("control"), py::arg("target"))
        .def("apply_SWAP", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_SWAP(c, t);
            }, py::arg("qubit1"), py::arg("qubit2"))
        .def("apply_iSWAP", [](PSS& s, std::size_t c, std::size_t t) {
                check_qubit(s, c); check_qubit(s, t); s.apply_iSWAP(c, t);
            }, py::arg("qubit1"), py::arg("qubit2"))
        // Rotation gates
        .def("apply_RZ", [](PSS& s, std::size_t q, double a) { check_qubit(s, q); s.apply_RZ(q, a); }, py::arg("qubit"), py::arg("angle"))
        .def("apply_RX", [](PSS& s, std::size_t q, double a) { check_qubit(s, q); s.apply_RX(q, a); }, py::arg("qubit"), py::arg("angle"))
        .def("apply_RY", [](PSS& s, std::size_t q, double a) { check_qubit(s, q); s.apply_RY(q, a); }, py::arg("qubit"), py::arg("angle"))
        // Operators
        .def("__mul__",  [](const PSS& a, const PSS& b) { check_same_qubits(a, b); return a * b; })
        .def("__imul__", [](PSS& a, const PSS& b) -> PSS& { check_same_qubits(a, b); a *= b; return a; })
        .def("__mul__",  [](const PSS& a, double c) { return a * c; })
        .def("__rmul__", [](const PSS& a, double c) { return c * a; })
        .def("__imul__", [](PSS& a, double c) -> PSS& { a *= c; return a; })
        .def("__add__",  [](const PSS& a, const PSS& b) { check_same_qubits(a, b); return a + b; })
        .def("__iadd__", [](PSS& a, const PSS& b) -> PSS& { check_same_qubits(a, b); a += b; return a; })
        .def("__sub__",  [](const PSS& a, const PSS& b) { check_same_qubits(a, b); return a - b; })
        .def("__isub__", [](PSS& a, const PSS& b) -> PSS& { check_same_qubits(a, b); a -= b; return a; })
        .def("__neg__",  [](const PSS& a) { return -a; });
}

template <std::size_t N>
void register_n(py::module_& m) {
    register_pauli_string<N>(m);
    register_pauli_sum<N>(m);
    using PSS = PauliSum<N, double>;
    using Opts = TruncationOptions<double>;

    std::string suffix = "_" + std::to_string(N);
    m.def(("random_pauli_sum" + suffix).c_str(),
          &random_pauli_sum<N, double>,
          py::arg("num_terms"), py::arg("seed") = 42ULL, py::arg("num_qubits") = N,
          ("Generate a random PauliSum with double coefficients (capacity " +
           std::to_string(N) + " qubits).").c_str());

    m.def(("trace_product" + suffix).c_str(),
          [](const PSS& a, const PSS& b) { return trace_product(a, b); },
          py::arg("a"), py::arg("b"),
          "Re tr(A*B), computed without forming the product.");
    m.def(("commutator" + suffix).c_str(),
          [](const PSS& a, const PSS& b) {
              check_same_qubits(a, b);
              auto c = commutator(a, b);
              c.canonicalize();
              return c;
          }, py::arg("a"), py::arg("b"), "[A, B], canonicalized.");
    m.def(("i_commutator" + suffix).c_str(),
          [](const PSS& a, const PSS& b) {
              check_same_qubits(a, b);
              auto c = i_commutator(a, b);
              c.canonicalize();
              return c;
          }, py::arg("a"), py::arg("b"), "i[A, B], canonicalized (Hermitian for Hermitian inputs).");
    m.def(("propagate_expectation" + suffix).c_str(),
          [](const PSS& observable, const PSS& rotations, const Opts& opts) {
              check_same_qubits(observable, rotations);
              return propagate_expectation(observable, rotations, opts);
          }, py::arg("observable"), py::arg("rotations"),
             py::arg("options") = Opts{},
          "Pauli propagation: <0|U† O U|0> for the rotation circuit U "
          "(gates exp(-i theta/2 P) in application order).");
}

// Fold-expression dispatcher — one call registers all supported capacities.
template <std::size_t... Ns>
void register_all(py::module_& m) {
    (register_n<Ns>(m), ...);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

// py::mod_gil_not_used() suppresses the Python 3.13t free-threaded warning.
// The GIL is still held on the Python side; OpenMP threads inside C++ are unaffected.
PYBIND11_MODULE(_paulib, m, py::mod_gil_not_used()) {
    m.doc() = "PauLIB — high-performance Pauli string operations (pybind11 bindings).";

    // Shared by all capacities (T = double).
    py::class_<TruncationOptions<double>>(m, "TruncationOptions")
        .def(py::init<>())
        .def_readwrite("max_weight", &TruncationOptions<double>::max_weight,
                       "Drop strings acting on more than this many qubits.")
        .def_readwrite("coefficient_cutoff", &TruncationOptions<double>::coefficient_cutoff,
                       "Drop strings with |coefficient| below this value.")
        .def_readwrite("max_terms", &TruncationOptions<double>::max_terms,
                       "Keep at most this many strings (largest |coefficient|).");

    // One instantiation per 64-bit word count; any qubit count up to 1024 is
    // supported by dispatching to the smallest capacity that fits.
    register_all<
        64, 128, 192, 256, 320, 384, 448, 512,
        576, 640, 704, 768, 832, 896, 960, 1024
    >(m);

    // Capacities (in qubits) for which template instantiations are available.
    m.attr("SUPPORTED_CAPACITIES") = std::vector<std::size_t>{
        64, 128, 192, 256, 320, 384, 448, 512,
        576, 640, 704, 768, 832, 896, 960, 1024
    };
}
