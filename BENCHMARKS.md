# Benchmarks — sdpa-gmp-omp

Fork base: upstream `ca110db`. **Requested 200-bit precision; GMP rounds up to 256 bits
(5 limbs), which is the actual working precision.** thanos: AMD EPYC 7232P, 8 physical
cores, Ubuntu.

## Methodology

External wall-clock seconds, **median of 3 repeats**, spread reported where it exceeds
rounding. Runs are pinned to physical cores on Linux (`taskset` + `OMP_PROC_BIND=true
OMP_PLACES=cores`, CPU set recorded per row); macOS exposes no per-process affinity, so those
runs use OpenMP binding plus one unrecorded warmup. The parameter file is passed explicitly
and its SHA-256 recorded. Every repeat is a row in the raw TSVs published alongside this
document; a run is counted only if it exited cleanly and every field parsed. Iteration counts
and objectives are checked across repeats and across configurations -- a wall-time ratio
between builds with different iteration counts measures path length, not speed, and is
flagged.

Solver-internal timers are elapsed time in sdpa-dd/sdpa-gmp; upstream sdpa-qd's timer
reports process CPU time (summed over threads), which this fork fixes -- all qd numbers here
use external wall time and the corrected clock.

## Results

## thanos-epyc7232p — gmp-200bit — `wall_s`

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


Iteration counts and objectives are identical across every repeat and every configuration.
Two upstream *regressions* become gains: `control1` (upstream threading 1.10x slower than
serial; this fork 1.79x faster) and `truss5` (upstream threading net negative: 13.87 s
serial vs 14.29 s threaded; this fork 11.15 s).

Caveat carried honestly: the OpenMP work thresholds were calibrated on double-double
arithmetic; other precisions should re-run the calibration sweep (see the companion
repository).

Raw data: [`bench/gmp_v2_thanos.tsv`](bench/gmp_v2_thanos.tsv).
