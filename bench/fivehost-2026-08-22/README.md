# Five-host campaign, 2026-08-22 — per-repeat rows

The raw rows behind the "Expanse — full 1→128 thread ladder" section of ../../BENCHMARKS.md:
**one TSV per machine, one row per repeat** — never an aggregate. File names carry internal
hostnames; the hardware is:

| file | TSV `machine` id | hardware | physical cores |
|---|---|---|---:|
| `gmp_fivehost_expanse.tsv` | `expanse-epyc7742` | SDSC Expanse node, 2×AMD EPYC 7742 | 128 |
| `gmp_fivehost_symmetry.tsv` | `symmetry-xeon6148` | cluster node, 2×Intel Xeon Gold 6148 | 40 |
| `gmp_fivehost_pi.tsv` | `pi-i9-13900k` | workstation, Intel i9-13900K (8P+16E hybrid) | 24 |
| `gmp_fivehost_thanos.tsv` | `thanos-epyc7232p` | workstation, AMD EPYC 7232P | 8 |
| `gmp_fivehost_mac.tsv` | `mac-m1max` | laptop, Apple M1 Max (8P+2E) | — |

The `repeat` column identifies each run and `cpuset` records the exact pinning. The final `src`
column names the per-cell file each row came from in the original campaign output — provenance
that matters for exactly one cell: the known duplicate below, whose two rows came from separate
campaign invocations (`out_up/…` and `out_up12/…`) and would otherwise be indistinguishable.
Rows with `wall_s = -` are declared-but-unexecuted dispositions, not measurements.

Check the integrity yourself (repeats agreeing on status/iterations/objective, statuses,
parameter families, cross-arm objective identity, repeat-id uniqueness and contiguity):

    python3 validate_integrity.py                       # report
    python3 validate_integrity.py --fail-on-integrity   # strict: exit 0 iff archive matches its documentation

Expected: 1,320 rows, 507 cells, zero **undocumented** failures, and exactly one **documented
anomaly present as declared** — the known duplicate `12_min`/upstream/64-thread cell on expanse,
where two campaign invocations each produced a `repeat=1` row (1,913.53 s and 1,921.50 s, 0.42%
apart; `src` = `out_up/…` and `out_up12/…`). BENCHMARKS.md publishes their median and declares
`n_obs=2` for that cell. The rows are kept **verbatim** — relabelling a measured row would be the
kind of quiet edit this validator exists to catch — so strict mode instead knows the anomaly's
exact signature and fails in *both* directions: if anything undocumented appears, **or** if the
documented duplicate ever goes missing or changes shape. CI runs the strict check on every push.
