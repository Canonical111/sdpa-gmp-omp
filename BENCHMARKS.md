<!-- ============================================================================
     DIVERGENCE WARNING, 2026-08-20 -- READ BEFORE REGENERATING ANYTHING

     This file is NO LONGER what patches/make_benchmarks_md.py produces. The generator
     emits BENCHMARKS-gmp.md in the recipe repo (142 lines, last generated 2026-08-06);
     this file is 400+ lines because sections below were WRITTEN BY HAND since then:

       - the whole "Expanse - full 1->128 thread ladder, five-host campaign (2026-08-22)"
         section, including the per-thread RSS table, the dE3/dE4 sparse-path measurements
         and the SDPB comparison
       - the whole "Expanse - AMD EPYC 7742, 32 threads - vs PRISTINE upstream" comparison
       - "The honest losses" and "One behavioural difference"
       - "Limitation - these numbers are a floor"
       - "These figures survive the 2026-08-18 chooser change"

     Copying a freshly generated BENCHMARKS-gmp.md over this file DESTROYS all of that,
     silently, with no conflict and no error. dd and qd are still byte-identical to their
     generated copies; gmp alone has diverged.

     How to reconcile this has NOT been decided (options: make this file hand-maintained
     and stop generating a gmp copy; or teach the generator to write only between explicit
     markers and leave prose alone). Until it is decided, do not regenerate-and-copy for
     gmp. See the recipe repo's BENCHMARKS-gmp.md, which carries the same warning.
     ============================================================================ -->

# Benchmarks — sdpa-gmp-omp

Fork base: upstream `ca110db`. **Requested 200-bit precision; GMP rounds up to 256 bits
(5 limbs), which is the actual working precision.** thanos: AMD EPYC 7232P, 8 physical
cores, Ubuntu.

## Host key

The `machine` ids recorded in the raw TSVs are internal hostnames; this is the hardware behind
them. Every row in every raw file carries one of these ids.

| TSV id | hardware | physical cores | notes |
|---|---|---:|---|
| `expanse-epyc7742` | SDSC Expanse compute node, 2×AMD EPYC 7742 | 128 | exclusive node, `taskset`-pinned |
| `symmetry-xeon6148` | cluster node, 2×Intel Xeon Gold 6148 | 40 | 2-way SMT, only physical cores used |
| `pi-i9-13900k` | workstation, Intel i9-13900K (hybrid) | 24 = 8P + 16E | P-cores 5.5 GHz, E-cores 4.3 GHz |
| `thanos-epyc7232p` | workstation, AMD EPYC 7232P | 8 | |
| `mac-m1max` | laptop, Apple M1 Max (hybrid) | 8P + 2E | no per-process affinity on macOS |

## Which table should I use?

| you want | read |
|---|---|
| the current reference numbers (1→128 ladder, per-family parameters, sparse problems) | [Expanse — full 1→128 thread ladder](#expanse--full-1128-thread-ladder-five-host-campaign-2026-08-22) |
| the best-replicated fork-vs-upstream comparison (3 repeats, 70 paired runs) | [Expanse — 32 threads vs pristine upstream](#expanse--amd-epyc-7742-32-threads--vs-pristine-upstream-2026-08-16) |
| sdpa-gmp against SDPB | [Versus SDPB](#versus-sdpb-the-standard-multiprecision-sdp-solver) |
| the threaded sparse-Cholesky path (dE3/dE4) | [The large sparse problems](#the-large-sparse-problems--the-path-the-tables-below-cannot-reach) |
| behaviour on an 8–24-core workstation | the three small-machine tables at the end |

Rule of thumb: quote totals and memory from the **ladder** (widest scope, but single runs on the
heavy tier), and quote fork-vs-upstream *ratios* from the **32-thread campaign** where replication
is strongest — the two agree where they overlap.

## Methodology

External wall-clock seconds, **median of 3 repeats**, spread reported where it exceeds
rounding. Runs are pinned to physical cores on Linux (`taskset` + `OMP_PROC_BIND=true
OMP_PLACES=cores`, CPU set recorded per row); macOS exposes no per-process affinity, so those
runs use OpenMP binding plus one unrecorded warmup. The parameter file is passed explicitly
and its SHA-256 recorded. Every repeat is a row in raw TSVs published in this repository: the
three small-machine campaigns in [`bench/gmp_v2_*.tsv`](bench/) and the 2026-08-22 five-host
campaign — including the Expanse ladder — in [`bench/fivehost-2026-08-22/`](bench/fivehost-2026-08-22/),
which carries its own integrity validator. The SDPB side's raw per-cell output lives in the
non-public companion repository and the Expanse share, as the SDPB section states.
A run is counted only if it exited cleanly and every field parsed. Iteration counts
and objectives are checked across repeats and across configurations -- a wall-time ratio
between builds with different iteration counts measures path length, not speed, and is
flagged.

Solver-internal timers are elapsed time in sdpa-dd/sdpa-gmp; upstream sdpa-qd's timer
reports process CPU time (summed over threads), which this fork fixes -- all qd numbers here
use external wall time and the corrected clock.

## Expanse — full 1→128 thread ladder, five-host campaign (2026-08-22)

**This is the current reference measurement.** It supersedes the 32-thread table below in scope:
the whole ladder rather than a 32-thread cap, per-family parameter files rather than one file for
everything, and ten problems including two large sparse ones.

**Provenance, stated as strongly as the record supports and no more.** The build tree that produced
these rows had a clean worktree whose code content is what this repository publishes (internal
build id `0803895`, source tree `9d6e0d88…` — the repository's public history was squashed to a
single release commit on 2026-08-22, so that id resolves only in the internal archive and in the
deployed module's PROVENANCE.txt). The binary hashes `bb64c6d5…`, checked on the machine. But the archived campaign material is per-repeat TSV
rows, and those rows do **not** each carry the binary hash they were produced with: the host
manifest in the archive predates the 2026-08-20 rebuild and names a different binary and tip. So
the internal build id is the **recorded deployment attribution** for this campaign, not per-row
repository-verifiable provenance — **the binary sha256 is the load-bearing identity** — and the same limit applies to the statement that both arms shared
compiler, flags and bundled GMP source — that is how the trees were built, not something the rows
prove. Closing this properly needs an in-allocation manifest captured at run time; the newer
`verify/verify_12_min.sb` does exactly that for anyone re-running a cell.

One exclusive Expanse node, 2×64-core AMD EPYC 7742. Upstream is `nakatamaho/sdpa-gmp` at
`ca110db` from a pristine clone, built `--enable-openmp=yes` and swept over the *same* ladder, so
it is compared at its own best. Both arms use the same compiler, flags and bundled GMP source, and
**each problem family gets the parameter file it is meant to run under** — SDPLIB at 200-bit
(`param.sdpa`), the min series at 512-bit (`paramgmplow.sdpa`), `dE3`/`dE4` at 256-bit
(`param_gmp256_d15.sdpa`). Iteration counts and objectives are identical between arms on every
paired problem.

**Repeat counts, stated rather than implied.** 3 repeats on SDPLIB, `8_min` and `10_min`; **1
repeat** on the heavy tier (`12_min`, `dE3`, `dE4`), where a single upstream `12_min` run costs
over half an hour; 2 on the one `12_min`/upstream/64-thread cell that two campaign invocations both
produced. Heavy-tier ratios are therefore **single-run observations**, and the thread count with
the lowest total is the lowest *observed*, not a located optimum.

### Ratio (upstream ÷ fork), median wall clock

| problem | m | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control1 | 21 | 1.00× | 0.86× | 0.82× | 0.82× | 0.82× | 0.86× | 1.14× | 1.27× |
| theta1 | 104 | 1.13× | 1.24× | 1.33× | 1.38× | 1.41× | 1.40× | 1.39× | 1.36× |
| gpp100 | 101 | 1.00× | 1.11× | 1.26× | 1.39× | 1.51× | 1.60× | **1.65×** | 1.62× |
| truss5 | 208 | 1.04× | 1.29× | 1.47× | 1.58× | 1.66× | 1.70× | 1.73× | **1.78×** |
| arch0 | 174 | 1.04× | 1.51× | 2.26× | 3.24× | 4.27× | 5.10× | **5.57×** | 5.54× |
| 8_min | 19 | 1.04× | 0.89× | 0.82× | 0.83× | 0.83× | 0.90× | 1.00× | 1.57× |
| 10_min | 74 | 1.05× | 1.29× | 1.55× | 1.83× | 2.04× | 2.18× | 2.24× | **2.53×** |
| 12_min | 330 | 1.06× | 1.52× | 2.23× | 3.68× | 5.72× | 8.07× | 9.85× | **11.84×** |

### Lowest observed total for each arm, and where

| problem | fork | upstream | ratio |
|---|---:|---:|---:|
| control1 | 0.22 s @1 | 0.18 s @4 | 0.82× |
| 8_min | 0.71 s @32 | 0.59 s @4 | 0.83× |
| theta1 | 1.58 s @32 | 2.20 s @64 | 1.39× |
| gpp100 | 6.89 s @64 | 11.37 s @64 | 1.65× |
| truss5 | 7.01 s @64 | 11.86 s @4 | 1.69× |
| 10_min | 9.13 s @64 | 20.01 s @32 | 2.19× |
| arch0 | 15.78 s @64 | 87.95 s @64 | **5.57×** |
| 12_min | 165.2 s @128 | 1,917.5 s @64 | **11.61×** |

`truss5` is the clean illustration of what is being fixed: upstream's lowest total is at **4
threads** and threading past that buys it nothing (11.86 s at 4, 12.70 s at 128 — it gets
*slower*), while the fork keeps improving to 64.

### Peak RSS — the direction depends on the problem, and the old single-point table was misleading

Peak RSS in MB, median over repeats, at 1 thread and at 128:

| problem | upstream @1 | upstream @128 | fork @1 | fork @128 | fork/upstream @128 |
|---|---:|---:|---:|---:|---:|
| control1 | 4.0 | 6.4 | 4.1 | 4.1 | **0.64×** |
| 8_min | 4.6 | 7.1 | 4.7 | 4.7 | **0.66×** |
| arch0 | 34.9 | 39.8 | 33.3 | 35.5 | **0.89×** |
| gpp100 | 16.6 | 17.6 | 15.8 | 16.8 | **0.95×** |
| truss5 | 11.7 | 13.6 | 11.7 | 15.3 | 1.12× |
| 10_min | 11.7 | 14.1 | 11.5 | 30.7 | 2.18× |
| theta1 | 7.5 | 10.0 | 7.5 | 24.4 | 2.44× |
| 12_min | 126.6 | 130.1 | 124.5 | 337.8 | 2.60× |

**On four of the eight problems the fork is lighter at full thread count.** That is the opposite of
what an earlier version of this file reported, which quoted only `10_min` and `12_min` at 32
threads and generalised from them to "the fork uses more memory once threaded".

Being precise about the *shape*, since "flat" is only true of two of the four:

| growth 1 → 128 threads | upstream | fork |
|---|---:|---:|
| `control1` | +60.0% | **+0.0%** |
| `8_min` | +54.3% | **+0.0%** |
| `arch0` | +14.0% | +6.6% |
| `gpp100` | +6.0% | +6.3% |
| `truss5` | +16.2% | +30.8% |
| `10_min` | +20.5% | +167% |
| `theta1` | +33.3% | +225% |
| `12_min` | +2.8% | +171% |

So: **flat** on `control1` and `8_min` (genuinely 0%, while upstream grows 54–60%); **modestly
growing but still lighter at the endpoint** on `arch0` and `gpp100` — on `gpp100` the fork actually
grows *slightly faster* than upstream (6.3% against 6.0%) and wins only because it starts lower;
**modestly heavier** on `truss5` (1.12×); and **materially heavier** on `theta1`, `10_min` and
`12_min` (2.2–2.6×), which is where threaded work needs real per-thread scratch. `12_min` is the
worst case at 337.8 MB against 130.1 — and also where the fork is 11.6× faster.

### The large sparse problems — the path the tables below cannot reach

The section "Limitation — these numbers are a floor" further down is correct about the SDPLIB set:
every one of those problems takes a **dense** `bMat`, so none of them exercises the threaded sparse
Cholesky or the threaded sparse `bMat` assembly. `dE3` (m=6067) and `dE4` (m=7401) do, and they are
now measured directly rather than left to a census.

Fork scaling on Expanse, wall clock, one repeat per cell:

| threads | 1 | 2 | 4 | 8 | 16 | 32 | **64** | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dE3` | 5,332 s | 2,732 s | 1,417 s | 758 s | 495 s | 324 s | **231 s** | 293 s |
| `dE4` | 8,322 s | 4,251 s | 2,195 s | 1,164 s | 709 s | 438 s | **356 s** | 504 s |

**Both regress past 64 threads** — 128 threads is 1.27× slower on `dE3` and 1.42× slower on `dE4`.
This is **consistent with** crossing the socket boundary of the 2×64 EPYC, but a full-solve ladder
does not isolate the cause: no phase-level NUMA or binding measurement was taken here, so treat
the mechanism as unattributed. `12_min` does not regress and peaks at 128. So the best thread count is problem-dependent on this machine, and the top of the ladder is
the wrong default for the large sparse problems.

Against upstream, measured on **pi** (i9-13900K, 24 physical cores) at 24 threads, one repeat:

| | fork | upstream | |
|---|---:|---:|---|
| `dE3` wall | **229.9 s** | 18,282 s | **79.5×** |
| `dE3` peak RSS | **636.5 MB** | 3,121.0 MB | **4.90× lighter** |
| `dE4` wall | **350.8 s** | 5,233 s | **14.9×** |
| `dE4` peak RSS | 877.9 MB | 864.3 MB | 1.02× (parity) |

`dE3` is the one case measured here where the fork is both **dramatically faster and substantially
lighter at the same time** — which is what the gate-3 census predicted for this problem class and
what the dense-`bMat` SDPLIB set could never show. Upstream took 18,282 s at 24 *requested* threads. (An
earlier draft added that it occupied "1.00 core" of those 24; the retained pi rows carry no CPU-time
or utilisation column, and the 1.00-core trace on record is from a different run on a different
machine, so that figure is withdrawn as unsupported for this cell.)

Two honest limits on this pair: one repeat each, and upstream `dE3`/`dE4` were not completed on
Expanse (a single upstream `dE3` run exceeded the per-run cap there), so the fork-vs-upstream
comparison for them is a pi measurement while the scaling ladder is an Expanse one.

### Versus SDPB, the standard multiprecision SDP solver

Measured separately on the same Expanse node class against SDPB 3.1.0 (`sdpa-solver`, sha256
`effc67c0…`) at matched nominal thresholds — 512-bit both sides, `epsilonStar 1e-7` against
`--dualityGapThreshold 1e-7`. The two agree on `12_min`'s primal objective to **nine leading
significant digits** (1.76×10⁻⁹ relative).

| `12_min`, each at its lowest observed total | fork | SDPB | |
|---|---:|---:|---|
| total | **165.2 s** @128 thr | 843 s @64 ranks | **5.10×** |
| per iteration | 2.016 s | 4.318 s | 2.14× |
| iterations | **78** | 188 | 2.41× fewer |
| peak memory | **337.8 MB** (RSS) | 8,610 MB (summed PSS) | **25.5×** |

The measured decomposition is **2.41× fewer reported iterations and 2.14× lower observed time per
reported iteration**. That is all it is: an iteration is not a common unit of work between two
different algorithms, so this does not establish a solver-independent causal split of the total
difference between "trajectory length" and "arithmetic throughput".

Caveats that bound this: one timed run per cell; `--maxSharedMemory` left at default, which an
older tuning sweep found worth up to 2.29×, so these are **SDPB-as-invoked** rather than SDPB at
its best; and one of three available `sdpa-solver` variants. **Evidence.** The raw per-cell output, the assembled data and the rebuild scripts live in the
companion recipe repository, which is **not public**, under
`review/artifacts/sdpb-comparison-2026-08/`; the method and caveats are in
`review/MIN-SERIES-BENCHMARK-2026-08-22.md` and a step-by-step Expanse verification guide with a
runnable job script is in `review/artifacts/sdpb-comparison-2026-08/VERIFY-ON-EXPANSE.md`. Since
those paths are not reachable from this repository, the load-bearing identities are reproduced here
so a reader is not sent to a path they cannot open:

| | sha256 |
|---|---|
| `12_min.dat-s` | `6b32a44162269c95d621349c84cef4089dd2411a208f9b6345461d3a982f0093` |
| fork binary (this source; internal build id `0803895`) | `bb64c6d5115dae7921f84bb704abf73a1c594346bfb4d2d7bb7566745b16a37f` |
| upstream binary (`ca110db`) | `4582adf7a0ee6d1518a8ec978eec94ac4ae0bf039ca111751b10f618fcdf961c` |
| SDPB 3.1.0-94-g16b1a86f | `effc67c0e516069508c7137dfef099b890485ae2ee6b49458b725a11c53fcf8e` |
| `paramgmplow.sdpa` | `e3e82d8dfcd9b7c37c389e7939809dacb102b822990a8cff98ecc61313ca1c80` |

The derived TSVs rebuild byte-for-byte from the raw; the figures rebuild pixel-identical but not
byte-identical, since a PNG records its Matplotlib version. Ask for repository access to audit
the rest.

## Expanse — AMD EPYC 7742, 32 threads — vs PRISTINE upstream (2026-08-16)

**Superseded in scope by the 1→128 ladder above, but retained**, because it is the
best-*replicated* comparison on this page: 3 repeats throughout, 70 paired SDPLIB/min runs with
zero iteration-count mismatches, and it is the run behind the README headline. It differs from the
campaign above in three ways that matter when comparing numbers: a **32-thread cap**, **upstream's
own `param.sdpa` for every problem** rather than per-family files, and fork commit `ae6d266`
rather than the released code (internal build id `0803895`). Where the two disagree, prefer the ladder for scope and this
one for replication. Expanse job 53568046, `exp-1-20`, 4h32m, exit 0.

**Both arms.** Upstream is `nakatamaho/sdpa-gmp` at `ca110db` from a fresh clone, verified
pristine (zero files containing `SDPA_SOLVE_` or `SDPA_BMAT_`), built **`--enable-openmp=yes`**
and swept over the same thread counts — it is compared at *its* best, not against a serial build
chosen for it. Fork is `ae6d266`. Same `gcc 10.2.0`, same `-O2`, same bundled GMP source, and
**upstream's own `param.sdpa`** (`caa2f86d…`) for both, so the settings cannot be called
fork-tuned. The SPOOLES `Make.inc` rescue was applied to upstream: a *build* fix without which it
does not compile here at all, identical to the one this repo's installer applies.

Capped at 32 threads, which is also this fork's `SDPA_SOLVE_MAX_TEAM` default — the forward
triangular solve degrades once a team spans two sockets.

**Iteration counts are identical in all 70 paired SDPLIB/min runs** (zero mismatches), so every
ratio here is like-for-like and equals a per-iteration ratio.

### Ratio (upstream ÷ fork), median wall clock

| problem | m | 1 thr | 4 | 8 | 16 | **32** |
|---|---:|---:|---:|---:|---:|---:|
| control1 | 21 | 1.04× | 0.82× | 0.96× | 0.96× | 0.87× |
| theta1 | 104 | 1.21× | 1.40× | 1.48× | 1.47× | **1.49×** |
| gpp100 | 101 | 1.10× | 1.36× | 1.51× | 1.62× | **1.70×** |
| truss5 | 208 | 1.11× | 1.55× | 1.70× | 1.78× | **1.80×** |
| arch0 | 174 | 1.12× | 2.43× | 3.47× | 4.58× | **5.37×** |
| 8_min | | 1.06× | 0.91× | 0.88× | 0.90× | 1.03× |
| 10_min | | 1.07× | 1.63× | 1.91× | 2.15× | **2.34×** |
| 12_min | | 1.05× | — | 3.83× | — | **8.32×** |

### Five-problem totals

| | upstream | fork | ratio |
|---|---:|---:|---:|
| 1 thread | 196.27 s | 175.77 s | 1.12× |
| 32 threads | 115.10 s | **32.13 s** | **3.58×** |
| fork @32 vs upstream **serial** | 196.27 s | **32.13 s** | **6.11×** |

### What 32 cores buy each side

| | 1 → 32 threads | gain |
|---|---|---|
| upstream, five problems | 196.27 → 115.10 s | **1.71×** |
| fork, five problems | 175.77 → 32.13 s | **6.1×** |
| upstream, `12_min` | 3951 → 2046 s | **1.93×** |
| fork, `12_min` | 3765 → 246 s | **15.3×** |

At one thread the fork is only 1.05–1.21× faster: almost the whole advantage is scaling, not
kernel micro-optimisation. `truss5` is the clean illustration — upstream takes 12.29 s at one
thread and 12.13 s at thirty-two, so threading buys it nothing measurable.

### The honest losses

**The fork is nominally *slower* on the problems too small to time.** `control1` (0.23–0.30 s) and
`8_min` (0.68–0.82 s) give ratios wandering to 0.82× and 0.88× and back above 1.0 with no pattern
— scheduler jitter, not the solver. `control1` is one of the five problems the pre-2026-08-16
headline averaged into its 2.36×.

**The fork uses more memory once threaded — on *some* problems.** At 32 threads on the two below
it does. ⚠️ **Do not generalise from this pair**, which an earlier version of this file did: the
full ladder above measures eight problems and the fork is *lighter* at 128 threads on four of them
— genuinely flat on two of those. It is also heavier on `truss5`, which this pair does not mention.
See "Peak RSS — the direction depends on the problem".

| peak RSS | upstream | fork @32t |
|---|---:|---:|
| 10_min | 12 MB | 20 MB |
| 12_min | 127 MB | **177 MB** |

**The fork's parser is slower.** On `12_min`'s 333 MB input, file read takes **11.6 s against
upstream's 6.5 s** — the price of record-bounded parsing. It is a fixed serial tax that does not
shrink with threads, so it dilutes the whole-run ratio:

| 12_min @32t | upstream | fork | ratio |
|---|---:|---:|---:|
| whole run | 2046.0 s | 245.8 s | 8.32× |
| main loop | 2039.1 s | 233.9 s | **8.72×** |

The whole-run figure is the one quoted, because it is what a user waits for.

### One behavioural difference

On `gpp100` the fork exits **3** where upstream exits **0**. Not a failure, and not unfair to
either arm: both run 61 iterations, both reach `pFEAS`, and both print the identical objective
`-4.4943550775891146e+01`. The fork reports `solveStatus = PARTIAL, failureIteration = 61` — a
recoverable numerical failure at the last iteration, rolled back and labelled. Upstream exits 0 on
every path, so it does not tell you this happened.

### Limitation — these numbers are a floor

Every problem in *this* section takes a **dense** `bMat`, so none of it exercises the threaded
sparse Cholesky or the threaded sparse bMat assembly — the work with the 4.2×-at-5×-less-memory
result on an m=6067 problem.

**As of 2026-08-22 that path is no longer unrepresented:** `dE3` (m=6067) and `dE4` (m=7401) are
measured directly in "The large sparse problems" above, including against upstream — `dE3` at
**79.5× faster and 4.90× lighter simultaneously**. The limitation stands for the SDPLIB tables on
this page; it no longer stands for the document.

### These figures survive the 2026-08-18 chooser change, and that is checked rather than assumed

Every number on this page was measured before the `bMat` chooser's default moved. It would be
reasonable to ask whether the promotion invalidates them. It does not, and the reason is
verifiable: a census of all 92 SDPLIB problems records which gate decides each one and its
aggregate/ordered-fill densities, and **not one of the 92 changes route** under the new policy.

- 84 are decided by gate 1 (`m ≤ 100` or `nBlock ≤ 5`) and 6 by gate 2 — both gates are
  bit-identical under either policy, since gate 2's cutoff was frozen at exactly the `0.5·m` that
  `m·√0.25` already evaluated to;
- of the two that reach gates 3–4, `truss5` has ordered fill **1.0** and stays DENSE, and
  `truss6` has **0.329** and stays SPARSE.

So the dense-`bMat` limitation above is not an artefact of the old chooser: these problems take a
dense factorisation because their factors *are* dense, and the new rule agrees. Census and
verifier: `review/artifacts/gate3/sdplib_census.tsv` and `census_verify.sh` in the recipe repo.

## AMD EPYC 7232P workstation, 8 cores — gmp-200bit — `wall_s`  (TSV id `thanos-epyc7232p`)

Median of 3 repeats, seconds (`wall_s`). Spread = (max-min)/median.

| problem | m | pristine1 | pristine8 | hardened1 | hardened4 | hardened8 |
|---|---|---|---|---|---|---|
| control1 | 21 | 0.430 ±2% | 0.340 ±24% | 0.240 ±8% | 0.240 ±4% | 0.240 |
| gpp100 | 101 | 44.760 ±1% | 21.240 ±1% | 44.080 ±1% | 27.800 ±1% | 20.870 ±1% |
| theta1 | 104 | 3.920 ±1% | 2.970 ±2% | 3.570 ±1% | 2.890 ±1% | 2.650 ±4% |
| truss5 | 208 | 13.870 ±1% | 14.290 ±1% | 13.640 ±1% | 11.870 ±2% | 11.150 |
| arch0 | 174 | 160.200 ±1% | 116.000 ±1% | 149.190 ±1% | 82.300 ±1% | 59.620 ±1% |
| **total** | | **223.2** | **154.8** | **210.7** | **125.1** | **94.5** |

**hardened8 vs pristine1: 2.36x**  (totals 223.2 s -> 94.5 s)

Iteration counts are identical between these two configurations on all 5 problems, so this is a like-for-like speed ratio.

### Integrity

- all repeats `ok`; iteration count and objective identical across repeats and across configs for every problem

### Peak RSS (MB, max over repeats)

| problem | pristine1 | pristine8 | hardened1 | hardened4 | hardened8 |
|---|---|---|---|---|---|
| control1 | 11.8 | 11.8 | 11.8 | 11.6 | 11.8 |
| gpp100 | 17.6 | 17.6 | 17.6 | 17.6 | 17.6 |
| theta1 | 11.8 | 11.7 | 11.6 | 11.7 | 11.8 |
| truss5 | 12.9 | 13.2 | 12.8 | 12.9 | 13.2 |
| arch0 | 36.6 | 36.3 | 36.5 | 36.5 | 36.2 |


## Intel i9-13900K workstation, 24 cores (8P+16E)  (TSV id `pi-i9-13900k`)

The `fork*` binary here is the one produced by the README build instructions, from a fresh
clone of this repository — the benchmark validates the installation guide's output, not a
hand-configured tree. `fork8P` is pinned to the 8 P-cores only.


Median of 3 repeats, seconds (`wall_s`). Spread = (max-min)/median.

| problem | m | pristine1 | pristine24 | fork1 | fork8P | fork24 |
|---|---|---|---|---|---|---|
| control1 | 21 | 0.100 | 0.110 | 0.100 | 0.100 | 0.100 |
| gpp100 | 101 | 18.230 | 5.820 | 18.220 | 6.340 | 5.700 |
| theta1 | 104 | 1.580 ±1% | 1.060 | 1.470 ±1% | 0.970 | 0.990 ±1% |
| truss5 | 208 | 5.590 | 5.780 | 5.590 | 4.280 | 4.270 |
| arch0 | 174 | 67.220 | 42.500 | 62.200 | 19.330 | 16.530 |
| **total** | | **92.7** | **55.3** | **87.6** | **31.0** | **27.6** |

**fork24 vs pristine1: 3.36x**  (totals 92.7 s -> 27.6 s)

Iteration counts are identical between these two configurations on all 5 problems, so this is a like-for-like speed ratio.

### Integrity

- all repeats `ok`; iteration count and objective identical across repeats and across configs for every problem

### Peak RSS (MB, max over repeats)

| problem | pristine1 | pristine24 | fork1 | fork8P | fork24 |
|---|---|---|---|---|---|
| control1 | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 |
| gpp100 | 17.8 | 17.5 | 17.7 | 17.5 | 17.5 |
| theta1 | 9.3 | 9.0 | 9.4 | 9.7 | 9.9 |
| truss5 | 13.3 | 13.3 | 13.4 | 13.0 | 13.2 |
| arch0 | 36.4 | 36.4 | 36.4 | 36.2 | 36.4 |


## Apple M1 Max laptop, 8P+2E  (TSV id `mac-m1max`)

The fork binary is again the one built by this README's macOS instructions.


Median of 3 repeats, seconds (`wall_s`). Spread = (max-min)/median.

| problem | m | pristine1 | pristine8 | fork1 | fork8 |
|---|---|---|---|---|---|
| control1 | 21 | 0.200 ±5% | 0.330 | 0.200 | 0.210 ±10% |
| gpp100 | 101 | 34.250 ±1% | 13.300 ±1% | 33.980 | 13.170 ±1% |
| theta1 | 104 | 2.940 ±1% | 2.110 ±2% | 2.840 ±3% | 2.000 ±2% |
| truss5 | 208 | 10.980 | 11.890 ±1% | 11.010 ±1% | 8.490 ±2% |
| arch0 | 174 | 123.610 ±1% | 87.820 ±1% | 118.450 ±1% | 39.430 ±1% |
| **total** | | **172.0** | **115.4** | **166.5** | **63.3** |

**fork8 vs pristine1: 2.72x**  (totals 172.0 s -> 63.3 s)

Iteration counts are identical between these two configurations on all 5 problems, so this is a like-for-like speed ratio.

### Integrity

- all repeats `ok`; iteration count and objective identical across repeats and across configs for every problem

### Peak RSS (MB, max over repeats)

| problem | pristine1 | pristine8 | fork1 | fork8 |
|---|---|---|---|---|
| control1 | 8.6 | 8.6 | 8.6 | 8.6 |
| gpp100 | 14.3 | 14.4 | 14.3 | 14.5 |
| theta1 | 8.6 | 8.6 | 8.6 | 8.6 |
| truss5 | 12.0 | 12.1 | 12.0 | 12.2 |
| arch0 | 29.5 | 29.6 | 29.6 | 29.7 |


Iteration counts and objectives are identical across every repeat and every configuration,
on all three machines. Upstream threading is *negative* on `truss5` on every one of them
(e.g. Mac: 10.98 s serial vs 11.89 s threaded; this fork: 8.49 s).
Two upstream *regressions* become gains: `control1` (upstream threading 1.10x slower than
serial; this fork 1.79x faster) and `truss5` (upstream threading net negative: 13.87 s
serial vs 14.29 s threaded; this fork 11.15 s).

Caveat carried honestly: the OpenMP work thresholds were calibrated on double-double
arithmetic; other precisions should re-run the calibration sweep (see the companion
repository).

Raw data: [`bench/gmp_v2_thanos.tsv`](bench/gmp_v2_thanos.tsv), [`bench/gmp_v2_pi.tsv`](bench/gmp_v2_pi.tsv), [`bench/gmp_v2_mac.tsv`](bench/gmp_v2_mac.tsv); five-host campaign rows in [`bench/fivehost-2026-08-22/`](bench/fivehost-2026-08-22/); the non-SDPLIB inputs themselves in [`bench/problems/`](bench/problems/).
