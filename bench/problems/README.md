# Benchmark problems that are not in SDPLIB

The tables in `../../BENCHMARKS.md` mix classic SDPLIB problems (`control1`, `theta1`, `gpp100`,
`truss5`, `arch0` — get those from the [SDPLIB collection](https://github.com/vsdp/SDPLIB)) with
five problems from conformal-bootstrap work that SDPLIB does not carry. **This directory ships
those five**, so every problem named in the tables is obtainable.

| problem | m | size (raw) | family | parameter file |
|---|---:|---:|---|---|
| `8_min` | 19 | 41 KB | min series, 512-bit | `paramgmplow.sdpa` |
| `10_min` | 74 | 3.4 MB | min series, 512-bit | `paramgmplow.sdpa` |
| `12_min` | 330 | 230 MB | min series, 512-bit | `paramgmplow.sdpa` |
| `dE3` | 6067 | 7.5 MB | large sparse, 256-bit | `param_gmp256_d15.sdpa` |
| `dE4` | 7401 | 7.5 MB | large sparse, 256-bit | `param_gmp256_d15.sdpa` |

The `.dat-s` files are xz-compressed (GitHub rejects files over 100 MB; `12_min` is 230 MB raw
and 6 MB compressed). Decompress and verify before use — the hashes are the very ones the
benchmark campaign recorded per row:

    xz -dk *.dat-s.xz
    shasum -a 256 -c SHA256SUMS.orig      # (sha256sum -c on Linux)

Run exactly as benchmarked (`-p` is mandatory: without it sdpa-gmp silently falls back to
compiled-in defaults at a different precision):

    OMP_PROC_BIND=true OMP_PLACES=cores OMP_NUM_THREADS=<n> \
      ./sdpa_gmp -ds 10_min.dat-s -o out.result -p paramgmplow.sdpa

The min-series inputs are digit-stripped copies verified **numerically lossless** against the
originals, entry by entry as 400-digit decimals (542,300 entries compared on `12_min`, zero
changed). The SDPLIB problems use upstream's own `param.sdpa` from the repository root.
