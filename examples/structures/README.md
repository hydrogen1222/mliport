# Example structures

## `li10gep2s12_primitive.vasp`

A 50-atom Li10GeP2S12 (LGPS) primitive cell used by the README quickstart and
the acceptance smoke tests. It is a real periodic cell: the formula is
Li20Ge2P4S24 per cell, all three axes are periodic, and the minimum
interatomic distance is 2.02 A.

Provenance: derived deterministically from a 2x2x2 LGPS supercell
(`local-data/LGPS.vasp` on the acceptance host)
(400 atoms, Ge16Li160P32S192) by halving each lattice vector, folding the
fractional coordinates into the new cell, and removing periodic duplicates.
Tiling the primitive back to 2x2x2 reproduces the source supercell within
3e-8 in fractional coordinates. The source supercell used for the derivation
had SHA-256 `f004de0b8b0f5687d7251acfa246347846452281d1add1d12ff8331fb9aa17bb`;
the committed primitive has SHA-256
`0e0702be3a31778839f730430c6d7989118df316c4976a1a98f945c32fd3561a`.

Any ASE-readable periodic structure works with mliport; this one is small
enough for CPU smoke tests and is used by `validation/acceptance/`.
