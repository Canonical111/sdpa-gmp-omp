# Five-host campaign, 2026-08-22 — per-repeat rows

The raw rows behind the "Expanse — full 1→128 thread ladder" section of ../../BENCHMARKS.md,
for all five hosts. Directory names are internal hostnames; the hardware is:

| directory | TSV `machine` id | hardware | physical cores |
|---|---|---|---:|
| `expanse/` | `expanse-epyc7742` | SDSC Expanse node, 2×AMD EPYC 7742 | 128 |
| `symmetry/` | `symmetry-xeon6148` | cluster node, 2×Intel Xeon Gold 6148 | 40 |
| `pi/` | `pi-i9-13900k` | workstation, Intel i9-13900K (8P+16E hybrid) | 24 |
| `thanos/` | `thanos-epyc7232p` | workstation, AMD EPYC 7232P | 8 |
| `mac/` | `mac-m1max` | laptop, Apple M1 Max (8P+2E) | — | One TSV per measured cell, **one row per repeat** —
never an aggregate; the `repeat` column identifies each run and `cpuset` records the exact pinning.

Check the integrity yourself (repeats agreeing on status/iterations/objective, statuses,
parameter families, cross-arm objective identity, repeat-id uniqueness and contiguity):

    python3 validate_integrity.py            # report
    python3 validate_integrity.py --fail-on-integrity   # exit 1 on any failure

Expected output ends: 1,320 rows, 507 cells, **ONE** integrity failure — the known duplicate
`12_min`/upstream/64-thread cell on expanse, where two campaign invocations each produced a
`repeat=1` row (1,913.53 s and 1,921.50 s, 0.42% apart). BENCHMARKS.md publishes their median and
declares `n_obs=2` for that cell; the validator flags it by design rather than hiding it.
