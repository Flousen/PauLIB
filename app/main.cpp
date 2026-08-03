#include <iostream>

#include "PauLib.h"

using namespace paulib;

// Minimal quickstart: build a small Pauli sum, add another to it, evolve it
// under a rotation, and take a commutator. See README.md for the full API.
int main() {
    PauliSum<5, double> a({"IIXII"});
    PauliSum<5, double> b({"IIZII"});

    PauliSum<5, double> h = a + b;
    std::cout << "H = IIXII + IIZII =\n" << h;

    PauliSum<5, double> rotations(
        std::vector<std::pair<std::string, double>>{{"IIZII", M_PI / 4}});
    h.evolve(rotations);
    std::cout << "\nH after evolving under exp(-i * pi/4 * IIZII) =\n" << h;

    std::cout << "\ni[IIXII, IIZII] =\n" << i_commutator(a, b);
}
