# Building sdpa-gmp-omp

Verified by execution on Ubuntu x86-64 and macOS Apple Silicon. Every trap in the troubleshooting
table is a real failure that was hit, not a hypothetical.

This is the only installation document. There used to be an `INSTALL` (no extension) alongside it
carrying generic Autoconf boilerplate that said nothing about this solver; it was untracked on
2026-08-20. Verified in a fresh clone that nothing needs it: `autoreconf -fi`,
`automake --add-missing` and `./configure` all succeed without it. An older clone may still have
one on disk; it is ignored.

## The short version

```bash
bash .claude/skills/install-sdpa-omp/scripts/install.sh
```

That does the whole job: detects the compiler, applies the SPOOLES fix, **verifies the OpenMP
linkage actually matches what was requested**, builds, and solves the bundled example checking the
objective. It prints `DONE` only after everything asked of it succeeded, and every `FATAL` message
names the specific trap it hit. Add `--prefix ~/.local` to install, or `--serial` for a no-OpenMP
build (legitimate and CI-tested, not a failure mode).

If you would rather an agent do it, that same directory is packaged as a Claude Code skill and is
discovered automatically in a clone.

## Doing it by hand

**On Linux**, where `gcc` really is GCC:

```bash
autoreconf -fi                          # upstream ships no ./configure
./configure --enable-openmp=yes
make -j$(nproc)
```

**On macOS, do not run that** — `/usr/bin/gcc` and `/usr/bin/cc` are both Apple clang, which has
no OpenMP. Use the block below, which names the compiler explicitly. Since 2026-08-20 `configure`
refuses this case outright rather than building a serial solver and reporting success, so you will
get an error telling you what to do; before that it succeeded silently.

GMP and SPOOLES are bundled and built in-tree. The SPOOLES compiler/flags fix that upstream's
`.POSIX:` `Make.inc` needs is applied from `external/spooles/patches/patch-Make.inc`, and CI
asserts it reached the real compile lines — so plain `make` must work. **Do not hand-edit
`Make.inc`:** the next `make` re-extracts SPOOLES and discards the change.

### Prerequisites

- **A real GCC.** On macOS `/usr/bin/gcc` is Apple clang, which has no OpenMP —
  `brew install gcc`. Both the install script and `configure` refuse that state now; it used to
  produce a **silent serial build**, because autoconf's `AC_OPENMP` records "unsupported" and
  carries on rather than failing. Never bypass either check.
- autoconf, automake, libtool, make, a C/C++ toolchain.
- macOS: the bundled GMP 6.2.1 predates arm64, so build against Homebrew's (`--with-system-gmp`).

### macOS (Apple Silicon), verified on an M1 Max

```bash
brew install gcc gmp autoconf automake libtool
GXX=$(ls $(brew --prefix gcc)/bin/g++-[0-9]* | head -1)   # brew may have just upgraded it
GCC=${GXX/g++/gcc}
autoreconf -fi
./configure --enable-openmp=yes CC="$GCC" CXX="$GXX" --with-system-gmp="$(brew --prefix gmp)"
make -j$(sysctl -n hw.perflevel0.physicalcpu)
```

## Verifying the build

Two checks. The first is instant and proves the solver is correct; the second is the one that
proves **OpenMP actually works**, which the first cannot.

### 1. It solves, and gets the right answer

```bash
./sdpa_gmp -ds example1.dat-s -o out.result -p param.sdpa
grep objValPrimal out.result          # expect -4.1900000000000000e+01 at the default 256-bit
nm sdpa_gmp | grep -i GOMP | head -1  # must print something -- a silent serial build is the classic failure
```

`example1` runs in milliseconds. That makes it a good correctness check and a **useless** threading
check: it finishes before threading can matter.

### 2. Threading works, and does not change the answer

`Canonical_example.dat-s.xz` ships for this purpose — a real m=74 problem with 12 blocks, big
enough that the threaded regions dominate — with a matched parameter file:

```bash
xz -dkc Canonical_example.dat-s.xz > Canonical_example.dat-s

time OMP_NUM_THREADS=1 ./sdpa_gmp -ds Canonical_example.dat-s -o t1.result -p Canonical_example.param.sdpa
time OMP_NUM_THREADS=8 ./sdpa_gmp -ds Canonical_example.dat-s -o t8.result -p Canonical_example.param.sdpa

# the answer must not depend on the thread count
diff <(grep -E 'objValPrimal|objValDual|phase.value|relative gap|Iteration =' t1.result) \
     <(grep -E 'objValPrimal|objValDual|phase.value|relative gap|Iteration =' t8.result)
# .result is gitignored, so this check leaves the working tree clean
```

**What to expect.** The `diff` must print nothing, and both runs must report:

```
phase.value  = pdOPT
   Iteration = 57
relative gap = 2.4491149344443876e-07
objValPrimal = -4.7925951579965138e-01
objValDual   = -4.7925927088815794e-01
```

Measured on an idle, exclusively allocated Intel Xeon Gold 6148 node (2×20 physical cores),
building this exact tip from a fresh clone:

| `OMP_NUM_THREADS` | wall | speedup |
|---:|---:|---:|
| 1 | 21.36 s | 1.00× |
| 2 | 13.56 s | 1.57× |
| 4 | 9.50 s | 2.25× |
| 8 | 7.70 s | 2.77× |
| **20** (one socket) | **7.08 s** | **3.02×** |
| 40 (both sockets) | 7.59 s | 2.81× |

All six runs were bit-identical — same objectives, same iteration 57, same relative gap — and they
match what the same tip produces on an Apple M1 Max, so the *result* does not depend on the
machine even though the timing does. Absolute times will differ on your hardware; what matters is
that the threaded run is clearly faster and the `diff` is empty.

If the times are the same, threading is not working: check the `GOMP` linkage above, and that you
are not oversubscribing or pinned to a single core.

**Note the last row.** On this small example 40 threads is *slower* than 20 — the peak is at one
socket. That is not a contradiction of the guidance below, which says filling both sockets pays on
this hardware: it pays on the large problems this solver is used for, and this example is m=74.
The smaller the problem, the earlier it saturates and the more the socket crossing costs. Treat
the table above as a threading smoke test, not as a scaling curve.

**Why a separate parameter file.** The shipped `param.sdpa` sets `epsilonStar = 1.0E-30`, which
this problem never satisfies — it stalls at iteration 114 and reports `pdFEAS`, which is a usable
timing check but an unclear reference. The production parameters used on our clusters set
`1.0E-7`, and this problem lands just outside that (`gap = 2.449e-07`), so it runs to the
iteration limit instead. `Canonical_example.param.sdpa` is those production settings — 256-bit
precision, `maxIteration = 300` — with `epsilonStar = 1.0E-6`, which converges it at iteration 57.
A converged solve is a cleaner thing to compare against than a stalled one.

*Two variables are deliberately exempt from the no-change guarantee and will make the `diff`
non-empty if you set them:* `SDPA_BMAT_MODE` *(a different factorisation) and*
`SDPA_SOLVE_BACKWARD` *(a reordered sum). Leave both unset for this check.*

## Choosing a thread count

This is the one setting worth thinking about; everything else can be left alone.

**Start with one socket's worth of physical cores.** Never count SMT threads. On Linux, pin:
`taskset -c <cores>` with `OMP_PROC_BIND=true OMP_PLACES=cores`.

"Match `OMP_NUM_THREADS` to physical cores" is the usual advice and it is **wrong on a large
two-socket node**. Measured on this solver:

| node | crossing the socket boundary | effect |
|---|---|---|
| 2×64-core EPYC 7742 | 64 → 128 threads | **2.65× slower** |
| 2×20-core Xeon | 20 → 40 threads | 1.04–1.30× faster |

So it is a property of the machine, not a number to copy. One socket: use all its physical cores.
Two sockets and a long job ahead: spend ten minutes comparing **one socket's physical-core count
against the whole machine's** — that is 64 vs 128 on the 2×64 EPYC and 20 vs 40 on the 2×20 Xeon
above — with each run pinned to exactly the cores it is meant to use (`taskset -c 0-63` against
`taskset -c 0-127`). The answer is also **problem-dependent**, not only machine-dependent: on the
same EPYC node, the 1→128 ladder in BENCHMARKS.md has the large sparse problems 1.27–1.42× slower
at 128 threads than at 64 while a dense m=330 problem is fastest at 128. Small problems saturate
early — on a 2×20 Xeon an m=2439 problem gains only 4% past one socket, so packing two 20-thread
jobs per node beats one 40-thread job.

`SDPA_SOLVE_MAX_TEAM` needs no action from you; it protects the triangular solve by itself.
*Why the socket boundary costs what it does, and why the solve needs a cap at all:*
[`doc/technical.pdf`](doc/technical.pdf), "What was threaded" and "Thread count is a property of
the machine".

*Threshold caveat: the OpenMP work thresholds were calibrated on double-double at roughly 256-bit
cost. Other precisions should re-run the calibration sweep.*

## Running it — where the reference lives

Everything about *running* the solver rather than building it is in
[`doc/technical.pdf`](doc/technical.pdf):

- **all 30 `SDPA_*` environment variables**, what each does, and which of them can change a
  computed value — "Environment variables";
- **the exit-status contract** every code from 0 to 3, including `solveStatus = PARTIAL` and
  `failureIteration` — "Exit status";
- why the chooser exists and what its gates decide — "The factorisation choice";
- what `SDPA_BMAT_MAX_GB` actually bounds, and why `m ≥ 46341` is refused — "What the memory cap
  bounds".

In normal use none of it is needed: set `OMP_NUM_THREADS` as above and leave every `SDPA_*`
variable alone. The two you might reach for are `SDPA_BMAT_MODE=legacy`, to reproduce a result
from before 2026-08-18 bit-for-bit, and `SDPA_BMAT_LOG=1`, to see which factorisation route was
chosen and why.


## Troubleshooting

| symptom | cause | action |
|---|---|---|
| macOS: `configure: error: … does not support OpenMP` | `/usr/bin/gcc` is Apple clang, which has none | `brew install gcc` and pass `CC`/`CXX` explicitly, as in the macOS block above. This was a **silent serial build** until 2026-08-20; the error replaced it |
| `c99: illegal option -- f` inside SPOOLES | upstream's `.POSIX:` `Make.inc` defaults `CC` to `c99` | the script's SPOOLES rescue; manually, also `mkdir -p external/i/SPOOLES/lib` before the `cp` |
| `storage size of 'TZ' isn't known` | strict toolchain hides `struct timezone` | `-D_GNU_SOURCE` in SPOOLES CFLAGS |
| link: `Graph_free` / `IV_entries` undefined | incomplete `spooles.a` from an earlier failed pass | delete SPOOLES `.o`s and rebuild it |
| macOS: GMP test-suite `Bus error: 10` | bundled GMP 6.2.1 predates arm64 | `--with-system-gmp` against `brew install gmp` |
| threaded run no faster | binary is serial, or threads unpinned/oversubscribed | check `nm \| grep GOMP`; pin cores; see thread count above |
| `invalid GMP format` from `param.sdpa` | entries 12–15 are printf formats needing the `F` conversion | use e.g. `%+8.3Fe`, not `%e` |

**Expected working-tree changes after a successful build: none.** `autoreconf` generates
`configure`, `Makefile.in`, `aclocal.m4` and the usual auxiliary scripts (`compile`,
`config.guess`, `config.sub`, `install-sh`, `missing`, `depcomp`); all are gitignored, so a
successful build leaves `git status` clean. Confirmed by cloning and building: clean afterwards. Until 2026-08-20 `INSTALL`
and `aclocal.m4` were tracked, which made every build produce a diff and required a careful
restore procedure; untracking them removed both problems.
