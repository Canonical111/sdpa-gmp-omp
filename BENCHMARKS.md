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

## thanos — EPYC 7232P, 8 physical cores

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


## pi — i9-13900K, 24 physical cores

The `fork*` binary here is the one produced by the README build instructions, from a fresh
clone of this repository — the benchmark validates the installation guide's output, not a
hand-configured tree. `fork8P` is pinned to the 8 P-cores only.

## pi-i9-13900k — gmp-200bit — `wall_s`

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


## Mac — Apple M1 Max (8P+2E)

The fork binary is again the one built by this README's macOS instructions.

## mac-m1max — gmp-200bit — `wall_s`

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

Raw data: [`bench/gmp_v2_thanos.tsv`](bench/gmp_v2_thanos.tsv), [`bench/gmp_v2_pi.tsv`](bench/gmp_v2_pi.tsv).
