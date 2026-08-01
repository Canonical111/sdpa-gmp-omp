# sdpa-gmp-omp

OpenMP-threaded, reproducible fork of sdpa-gmp.

Fork of [nakatamaho/sdpa-gmp](https://github.com/nakatamaho/sdpa-gmp) at `ca110db`, carrying
two patches (kernel fixes + Schur-complement threading). Reported upstream; not adopted there.

**Measured** (requested 200-bit = actual 256-bit precision, EPYC 7232P, external wall clock,
median of 3 pinned repeats): **223.2 s → 94.5 s (2.36×)** over 5 SDPLIB problems; arch0
160.2 → 59.6 s (2.69×). Iteration counts and objectives identical across every repeat and
configuration. Two upstream *regressions* become gains: control1 (threading was 1.10× slower,
now 1.79× faster) and truss5 (upstream threading was net negative).

Every modified file carries an in-file, dated change notice (GPLv2 §2a): `sdpa_newton.cpp`,
`sdpa_parts.cpp`, `mplapack/Rgemm_NN_omp.cpp`, `mplapack/Rgemm_NT_omp.cpp`;
`mplapack/mplapack_omp_tuning.h` is new.

Build: `autoreconf -fi && ./configure --enable-openmp=yes && make -j$(nproc)` (bundles GMP and
SPOOLES; if SPOOLES stops with a `struct timezone` error, set `CC = gcc` in its `Make.inc` —
see CI for the exact commands). Threshold caveat: the OpenMP work thresholds were calibrated
on double-double at ~256-bit cost; other precisions should re-run the calibration sweep.

License: GPL v2, unchanged (`COPYING`); original SDPA authors retain copyright.
