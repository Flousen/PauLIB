#pragma once

// Coefficient-type traits. PauliSum<N, T> stores a term as value * i^k * P,
// with k in the flags byte. T may be real (double, float) or complex
// (std::complex<double>). With a real T, terms with odd k (imaginary phase)
// cannot be absorbed into a real result and reductions skip them; with a
// complex T, the phase is folded in and nothing is skipped.

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>

namespace paulib {

template <typename T>
struct scalar_traits {
    using real = T;
    static constexpr bool is_complex = false;
};

template <typename R>
struct scalar_traits<std::complex<R>> {
    using real = R;
    static constexpr bool is_complex = true;
};

// Real type underlying T: T itself for real T, R for std::complex<R>.
template <typename T>
using real_t = typename scalar_traits<T>::real;

template <typename T>
inline constexpr bool is_complex_v = scalar_traits<T>::is_complex;

// |v|^2 without a square root.
template <typename T>
inline real_t<T> abs2(const T& v) {
    if constexpr (is_complex_v<T>) return std::norm(v);
    else return v * v;
}

// True if a term with these flags contributes to a T-valued reduction: always
// for complex T, only for even k (phase +1 or -1) for real T.
template <typename T>
constexpr bool phase_representable(uint8_t flags) {
    if constexpr (is_complex_v<T>) return true;
    else return (flags & 0x01) == 0;
}

// i^k for k = flags & 3. For real T only the sign is returned; callers check
// phase_representable() first.
template <typename T>
inline T phase_factor(uint8_t flags) {
    using R = real_t<T>;
    if constexpr (is_complex_v<T>) {
        switch (flags & 0x03) {
            case 0:  return T{R{1}, R{0}};
            case 1:  return T{R{0}, R{1}};
            case 2:  return T{R{-1}, R{0}};
            default: return T{R{0}, R{-1}};
        }
    } else {
        return (flags & 0x02) ? T{-1} : T{1};
    }
}

template <typename T>
inline std::string scalar_to_string(const T& v) {
    if constexpr (is_complex_v<T>)
        return "(" + std::to_string(v.real()) + (v.imag() < 0 ? "-" : "+") +
               std::to_string(v.imag() < 0 ? -v.imag() : v.imag()) + "i)";
    else
        return std::to_string(v);
}

} // namespace paulib
