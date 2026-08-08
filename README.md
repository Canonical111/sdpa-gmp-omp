# sdpa-gmp-omp

OpenMP-threaded, reproducible fork of sdpa-gmp.

Fork of [nakatamaho/sdpa-gmp](https://github.com/nakatamaho/sdpa-gmp) (upstream README preserved as [README-UPSTREAM.md](README-UPSTREAM.md)) at `ca110db`, carrying
two patches (kernel fixes + Schur-complement threading). Reported upstream; not adopted there.

**Measured** (requested 200-bit = actual 256-bit precision; 5 SDPLIB problems, external wall
clock, median of 3 pinned repeats):

| | EPYC 7232P (8 cores) | i9-13900K (24 cores) | M1 Max (8P+2E) |
|---|---|---|---|
| upstream, serial | 223.2 s | 92.7 s | 172.0 s |
| **this fork** | **94.5 s (2.36×)** | **27.6 s (3.36×)** | **63.3 s (2.72×)** |

Iteration counts and objectives are identical across every repeat, every configuration and
all three machines. Two upstream *regressions* become gains: control1 (upstream threading
1.10× slower than serial; this fork 1.79× faster) and truss5 (upstream threading net
negative on every machine tested). The i9 and M1 fork binaries are the ones built by this
README's own instructions from a fresh clone. Full tables, methodology and raw per-repeat
data: [BENCHMARKS.md](BENCHMARKS.md) and [`bench/`](bench/).

Every modified file carries an in-file, dated change notice (GPLv2 §2a): `sdpa_newton.cpp`,
`sdpa_parts.cpp`, `mplapack/Rgemm_NN_omp.cpp`, `mplapack/Rgemm_NT_omp.cpp`;
`mplapack/mplapack_omp_tuning.h` is new.

Build: `autoreconf -fi && ./configure --enable-openmp=yes && make -j$(nproc)` (bundles GMP and
SPOOLES; if SPOOLES stops with a `struct timezone` error, set `CC = gcc` in its `Make.inc` —
see CI for the exact commands). Threshold caveat: the OpenMP work thresholds were calibrated
on double-double at ~256-bit cost; other precisions should re-run the calibration sweep.

Prefer it automated? This repository packages an agent skill —
[`.claude/skills/install-sdpa-omp/`](.claude/skills/install-sdpa-omp/) — that performs the
whole installation (compiler detection, the SPOOLES rescue, OpenMP verification, smoke test)
on Linux and macOS, verified on x86-64 and Apple Silicon. Claude Code discovers it
automatically in a clone; `bash .claude/skills/install-sdpa-omp/scripts/install.sh` also
works standalone.

### macOS (Apple Silicon) — verified on an M1 Max

The bundled GMP 6.2.1's own test suite bus-errors on Apple Silicon, so use Homebrew's GMP
via the configure option upstream already provides. `/usr/bin/gcc` is Apple clang (no
OpenMP); Homebrew GCC is required:

```bash
brew install gcc gmp autoconf automake libtool
GCC=$(ls $(brew --prefix gcc)/bin/gcc-[0-9]* | head -1)   # resolve the current version
GXX=$(ls $(brew --prefix gcc)/bin/g++-[0-9]* | head -1)   # (brew install may have just upgraded it)
autoreconf -fi
./configure CC="$GCC" CXX="$GXX" --enable-openmp=yes \
            --with-system-gmp --with-gmp-includedir=/opt/homebrew/include \
            --with-gmp-libdir=/opt/homebrew/lib
make -j8 || true   # first pass stops inside SPOOLES (Apple's c99 rejects the flags) -- expected
M=external/spooles/work/internal/Make.inc
sed -i '' 's|^# CC = gcc|  CC = '$GCC'|' $M
sed -i '' 's|^  CFLAGS += -O2 -funroll-all-loops|  CFLAGS += -O2 -funroll-all-loops -Wno-error=int-conversion -Wno-error=implicit-function-declaration -Wno-error=incompatible-pointer-types|' $M
( cd external/spooles/work/internal && find . -name '*.o' -delete && rm -f spooles.a \
  && make global -f makefile && mkdir -p ../../../i/SPOOLES/lib \
  && cp spooles.a ../../../i/SPOOLES/lib/libspooles.a )
make -j8
otool -L sdpa_gmp | grep gomp   # must print libgomp
```

License: GPL v2, unchanged (`COPYING`); original SDPA authors retain copyright.

## Exit status

Scripts that loop over problems can rely on the exit code:

| outcome | exit |
|---|---|
| solver ran to any stopping condition (`pdOPT`, `pdFEAS`, `pFEAS`, `dFEAS`, `pdINF`, `pINF_dFEAS`, `pFEAS_dINF`, `pUNBD`, `dUNBD`, `noINFO`) | **0** |
| iteration limit reached | **0** |
| infeasibility / unboundedness detected | **0** |
| malformed input, unreadable file, invalid parameter | **1** with a diagnostic (line-numbered for data files) |
| numerical failure with nothing valid to print -- no iteration completed, or the updated `X`/`Z` left the positive-definite cone (the in-memory iterate is invalid) | **2**, `solveStatus = FAILURE` in the result file, no solution section |
| late Schur-complement factorisation failure after `k` good iterations -- the last **valid** iterate is printed and labelled | **3**, `solveStatus = PARTIAL`, `failureIteration = k` in the result file |

Infeasibility and the iteration limit are valid mathematical results, not errors. Upstream
exited 0 on *every* path -- including fatal errors -- so a crashed run was indistinguishable
from a solved one in any harness that checks exit codes.
