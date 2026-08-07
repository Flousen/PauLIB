"""
Simple paulib example: VQE-style Hamiltonian manipulation.
Run from the python/ directory: python example.py
"""

import math
from paulib import PauliString, PauliSum

# --- PauliString basics ---------------------------------------------------

print("=== PauliString ===")
ps1 = PauliString("XYZZ")
ps2 = PauliString("ZZXY")
print(f"ps1          = {ps1}")
print(f"ps2          = {ps2}")
print(f"ps1 * ps2    = {ps1 * ps2}")
print(f"commute      = {ps1.commute(ps2)}")

ps1.apply_H(0)          # Hadamard on qubit 0
print(f"ps1 after H  = {ps1}")

# --- Gate operations -------------------------------------------------------

print("\n=== Gate operations ===")
H = PauliSum([
    ("ZIII", -1.0),
    ("IZII", -1.0),
    ("IIZI", -1.0),
    ("IIIZ", -1.0),
    ("XXXX",  0.5),
    ("YYYY",  0.5),
    ("ZZZZ",  0.25),
])
print(f"Hamiltonian ({len(H)} terms):\n{H}")

H.collate()
print(f"After collate ({len(H)} terms):\n{H}")

angle = math.pi / 4
H.apply_RZ(0, angle)
print(f"\nAfter RZ(0, pi/4) ({len(H)} terms):\n{H}")

# --- Time evolution --------------------------------------------------------

print("\n=== Time evolution ===")
# Observable: ZZ on qubits 0,1
observable = PauliSum([("ZZII", 1.0)])

# Rotation generator: X on qubit 0, angle pi/8
rotations = PauliSum([("XIII", math.pi / 8)])

print(f"Before evolve: {observable}")
observable.evolve(rotations)
print(f"After evolve:  {observable}")
