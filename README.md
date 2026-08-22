# sdpa-gmp-omp

An OpenMP-threaded, reproducible fork of **sdpa-gmp**, a semidefinite-programming solver that
works in arbitrary-precision arithmetic.

Upstream builds with OpenMP but barely benefits from it: across five SDPLIB problems, thirty-two
cores buy upstream **1.71×**. This fork threads the regions that actually dominate the runtime,
makes threaded results reproducible, and re-derives the heuristic that decides how the Schur
complement is factored.

Fork of [nakatamaho/sdpa-gmp](https://github.com/nakatamaho/sdpa-gmp) at `ca110db` — see that
repository for upstream's own documentation. Reported upstream; not adopted there.

| | |
|---|---|
| **Build and run it** | [INSTALL.md](INSTALL.md) — build, verify, choose a thread count, troubleshoot |
| **Why it works this way** | [doc/technical.pdf](doc/technical.pdf) — mechanisms, the derivation behind the factorisation rule, every environment variable, the exit-status contract, and what is *not* established |
| **Full benchmark tables** | [BENCHMARKS.md](BENCHMARKS.md), harness in [`bench/`](bench/) |

## The benchmarks: best-replicated first, then the current reference

Two campaigns are summarised here. The table directly below is the **best-replicated** one
(3 repeats, 70 paired runs, zero iteration-count mismatches; 32-thread cap). The subsection after
it is the **current reference** (full 1→128 ladder, per-family parameters, large sparse problems;
mostly single runs). [BENCHMARKS.md](BENCHMARKS.md) is the full dossier and says which table to
use for what.

One AMD EPYC 7742 node at 32 threads, against upstream `ca110db` from a pristine clone built
**`--enable-openmp=yes`** and swept over the same thread counts — upstream compared at *its* best,
not against a serial build chosen for it. Same compiler, flags, bundled GMP source and upstream's
own `param.sdpa` (requested 200-bit, actual 256-bit). External wall clock, median of 3. Iteration
counts are identical in all 70 paired runs, so every ratio is like-for-like.

| | upstream | this fork | |
|---|---:|---:|---|
| 5 SDPLIB problems, both at 32 threads | 115.1 s | **32.1 s** | **3.58×** |
| 5 SDPLIB problems, fork vs upstream **serial** | 196.3 s | **32.1 s** | **6.11×** |
| a large problem (`12_min`), both at 32 threads | 2046 s | **246 s** | **8.32×** |

**Almost the whole gain is scaling, not faster kernels — and that is the point.** At one thread
this fork is only 1.05–1.21× faster. What differs is what the 2nd through 32nd core buy: over
those five problems upstream gains 1.71× and this fork **6.1×**; on `12_min`, 1.93× against
**15.3×**. On `truss5` upstream threading buys nothing measurable at all — 12.29 s at one thread,
12.13 s at thirty-two.

**Where it loses:** on problems too small to time — `control1` at 0.23 s, and `8_min` — this fork
is nominally slower.

### The full 1→128 ladder, and the large sparse problems (2026-08-22)

The table above is capped at 32 threads and uses one parameter file for every problem. A later
campaign swept the whole 1→128 ladder on the same node class, gave each problem family the
parameter file it is meant to run under, and added two large sparse problems that the SDPLIB set
cannot reach. Each arm at its own lowest observed total:

| | fork | upstream | |
|---|---:|---:|---|
| `arch0` (m=174, dense) | 15.78 s @64 thr | 87.95 s @64 thr | **5.57×** |
| `12_min` (m=330) | 165.2 s @128 thr | 1,917.5 s @64 thr | **11.61×** |
| `dE3` (m=6067, **sparse**) — on pi @24 thr | **229.9 s** | 18,282 s | **79.5×** |
| `dE3` peak RSS | **636.5 MB** | 3,121.0 MB | **4.90× lighter** |

`dE3` is the case where the fork is **both dramatically faster and substantially lighter**, and it
is the one that exercises the threaded sparse Cholesky — the path every SDPLIB problem above misses,
because they all take a dense `bMat`. Upstream took 18,282 s at 24 *requested* threads.

**Read the heavy rows as single observations.** SDPLIB and the small min problems have 3 repeats;
`12_min`, `dE3` and `dE4` have **one**, and the quoted thread count is the lowest *observed*, not a
located optimum. Two of the large problems also **regress past 64 threads** on a 2×64-core node
(`dE3` 1.27× slower at 128, `dE4` 1.42×) — consistent with crossing the socket boundary, though a
full-solve ladder does not isolate the cause — so the top of the ladder is the wrong default there.

**Memory is problem-dependent, in both directions.** Across the eight problems on the full ladder
the fork is *lighter* at 128 threads on four: **flat** on `control1` and `8_min` (0% growth from 1
to 128 threads, where upstream grows 54–60%), and **modestly growing but still lighter** on `arch0`
and `gpp100`. It is modestly heavier on `truss5` (1.12×) and materially heavier on `theta1`,
`10_min` and `12_min` (2.2–2.6×), where threaded work needs real per-thread scratch.
[BENCHMARKS.md](BENCHMARKS.md) has the per-thread table and the growth figures.

<details><summary>Earlier small-machine measurements (8–24 cores), for reference</summary>

| | EPYC 7232P (8 cores) | i9-13900K (24 cores) | M1 Max (8P+2E) |
|---|---|---|---|
| upstream, serial | 223.2 s | 92.7 s | 172.0 s |
| **this fork** | **94.5 s (2.36×)** | **27.6 s (3.36×)** | **63.3 s (2.72×)** |

These were the headline until 2026-08-16 and understate the fork: the five SDPLIB problems are
small (m ≤ 208), 8 cores is where a threaded fork and a barely-threaded upstream look most alike,
and the numbers predate the threaded sparse Cholesky, the threaded `bMat` assembly and the
threaded triangular solve.

</details>

## What was improved

**Threading the regions that dominate.** The `bMat` build, the sparse Cholesky, the `bMat`
assembly and the forward triangular solve are threaded. This is where the table above comes from,
and the table's own decomposition is the honest summary: the gain is scaling, not faster kernels.

The forward solve is **bit-identical** by construction — within a row the writes go to distinct
destinations. The backward pass reassociates a sum, so it is opt-in behind `SDPA_SOLVE_BACKWARD`
and off by default.

**Reproducibility, measured rather than asserted.** Four problems at 1, 8, 20 and 40 threads, two
repetitions each — 32 runs — produced **exactly one distinct objective and one iteration count per
problem**. CI asserts the 1-thread-versus-N-thread case on every push. Upstream threaded is not
reproducible: on the sibling `sdpa-dd` fork, where this was first diagnosed, one problem ran 50,
51, 56, 65 and 80 iterations across five identical invocations, and three returned a different
objective run to run. The cause was a reduction combined under `omp critical`, so the summation
order followed thread scheduling.

**A re-derived factorisation choice (2026-08-18).** Four gates decide dense vs sparse for the
Schur complement, and gates 2 and 3 were both driven by a single constant — so neither could be
retuned without moving the other. Splitting it frees gate 3, which then stops being a tuned
threshold at all: symbolic factorisation only adds entries, so an aggregate density above F
already proves the ordered fill exceeds F, making gate 3 the provable early exit from gate 4.
**One tunable (F = 0.40, unchanged) replaces two.**

On a 221-instance census of bootstrap problems, **167 instances across seven structures** stop
taking a route that cost **2.6–9.8×** in time and **3.5–4.9×** in peak memory, with no reversal at
any of nine labelled thread points on two architectures. Not "sparse always wins": SDPLIB
`truss5` has an ordered fill of 1.0 and dense is 1.4× faster there — gate 4 sends it to dense,
which is the point of testing fill rather than assuming an answer.

**Two inherited kernel defects, fixed.** `mplapack` 2.0.1 had dropped netlib `dgemm`'s zero-skip
guard, so multiplying by zero still paid full multiprecision cost; it is restored in
`mplapack/Rgemm_NN_omp.cpp` and `Rgemm_NT_omp.cpp`. And OpenMP regions were entered regardless of
problem size — upstream has a threshold mechanism but disables it behind `if (0)`, and it cannot
simply be switched on because the `_ref` bodies it calls are absent and would not link. Both were
quantified on `sdpa-dd` rather than here, so the figures live in that fork's write-up; at 256-bit
they would need re-measuring, which is also why the OpenMP thresholds carry a precision caveat.

**Reporting that does not lie.** A recoverable numerical failure is reported as
`solveStatus = PARTIAL` naming the failing iteration. Upstream exits 0 on every path.
