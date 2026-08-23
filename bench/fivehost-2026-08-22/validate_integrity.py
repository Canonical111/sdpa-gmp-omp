#!/usr/bin/env python3
"""Integrity validation over the archived five-host campaign rows.

Run:  python3 validate_integrity.py [--root DIR] [--fail-on-integrity]

This is the validator behind the "N rows / M cells / zero integrity failures" claim in
the benchmark documentation. The rows it reads sit beside it in this directory, ONE TSV PER
MACHINE (gmp_fivehost_<host>.tsv), one row per repeat; each row's `src` column names the
per-cell file it came from in the original campaign output.

CHECKS. The first five were in the original. The last two were added because an independent review
observed that the original could pass a cell whose repeat ids were duplicated or incomplete -- and
that is not hypothetical: the campaign's own history includes a heavy-tier bug where several
problems wrote to one filename, and a case where two campaign invocations each produced a
repeat=1 row for the same cell. A validator that cannot see either of those is not checking
completeness, only consistency.

  1. within a cell, repeats agree on status
  2. within a cell, repeats agree on iteration count
  3. within a cell, repeats agree on the objective
  4. status is one of ok / partial
  5. cpuset recorded (non-macOS, which has no taskset)
  6. parameter file matches the problem's family
  7. fork and upstream reach an identical objective on every shared problem
  8. NEW: repeat ids within a cell are unique
  9. NEW: repeat ids within a cell form the contiguous set 1..n

A cell legitimately holding one repeat (the heavy tier) passes 8 and 9 with {1}. A cell holding
{1, 1} fails 8 -- which is exactly the out_up / out_up12 collision -- and a cell holding {1, 3}
fails 9.

DOCUMENTED ANOMALIES AND STRICT MODE. This archive contains exactly one known anomaly, declared
in KNOWN_ANOMALIES below and in README.md: the expanse 12_min/upstream/64-thread cell holds two
repeat=1 rows, from two separate campaign invocations (src out_up/... and out_up12/...). The rows
are kept VERBATIM -- relabelling a measured row's repeat id would be the kind of quiet data edit
the rest of this file exists to catch. Instead, --fail-on-integrity passes when the archive
matches its documentation EXACTLY: the known anomaly present with precisely the declared src
pair, and nothing else wrong. It exits 1 both for any UNDOCUMENTED failure and for a documented
anomaly that has gone MISSING or changed shape -- either direction means the archive no longer
matches its documentation, and a strict gate that cannot detect the second is not a tamper check.
"""
import csv, glob, collections, os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = HERE   # the rows live beside this script in bench/fivehost-2026-08-22/
HOSTS = ["mac", "thanos", "pi", "symmetry", "expanse"]
EXPECT_PARAM = {
    "control1": "33f968827a10dca7", "theta1": "33f968827a10dca7", "gpp100": "33f968827a10dca7",
    "truss5": "33f968827a10dca7", "arch0": "33f968827a10dca7",
    "8_min": "e3e82d8dfcd9b7c3", "10_min": "e3e82d8dfcd9b7c3", "12_min": "e3e82d8dfcd9b7c3",
    "dE3": "3859004693f5546f", "dE4": "3859004693f5546f"}

# (host, problem, arm, threads) -> the exact src set that constitutes the documented anomaly.
KNOWN_ANOMALIES = {
    ("expanse", "12_min", "up", "64"): {
        "out_up/rows_H_12_min_t64.tsv",
        "out_up12/rows_H_12_min_t64.tsv",
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=DEFAULT_ROOT)
ap.add_argument("--fail-on-integrity", action="store_true")
a = ap.parse_args()
ROOT = os.path.abspath(a.root)

total_fail = 0
grand_rows = grand_cells = grand_skipped = 0
documented_seen = []
for h in HOSTS:
    rows, skipped = [], 0
    # One TSV per machine. Each row carries a `src` column naming the per-cell file it came from
    # in the original campaign output -- that column is what keeps the known duplicate cell
    # (12_min/upstream/64: one repeat=1 row from out_up/, one from out_up12/) distinguishable
    # after concatenation, instead of looking like an inexplicable pair of identical labels.
    for f in sorted(glob.glob(os.path.join(ROOT, "gmp_fivehost_%s.tsv" % h))):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if not r.get("wall_s") or r["wall_s"] == "-":
                skipped += 1          # a declared-but-unexecuted cell, not a measurement
                continue
            rows.append(r)
    if not rows:
        print("%s: NO ROWS" % h)
        continue
    cells = collections.defaultdict(list)
    for r in rows:
        arm = "fork" if r["config"].startswith("fork") else "up"
        cells[(r["problem"], arm, r["threads"])].append(r)

    fails = []
    for k, rs in cells.items():
        if len({x["status"] for x in rs}) > 1: fails.append("mixed status %s" % (k,))
        if len({x["iters"] for x in rs}) > 1: fails.append("iters vary %s" % (k,))
        if len({x["obj"] for x in rs}) > 1: fails.append("obj varies %s" % (k,))
        ids = [x["repeat"] for x in rs]
        if len(set(ids)) != len(ids):
            srcs = {x.get("src", "") for x in rs}
            if KNOWN_ANOMALIES.get((h,) + k) == srcs:
                documented_seen.append((h,) + k)
            else:
                fails.append("duplicate repeat ids %s: %s (srcs %s)"
                             % (k, sorted(ids), sorted(srcs)))
        elif sorted(int(i) for i in ids) != list(range(1, len(ids) + 1)):
            fails.append("non-contiguous repeat ids %s: %s" % (k, sorted(ids)))
    badst = [r for r in rows if r["status"] not in ("ok", "partial")]
    nocpu = [r for r in rows if h != "mac" and r.get("cpuset") in (None, "", "-")]
    pmis = {(r["problem"], r["param_sha"][:16]) for r in rows
            if EXPECT_PARAM.get(r["problem"]) and r["param_sha"][:16] != EXPECT_PARAM[r["problem"]]}
    byp = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        arm = "fork" if r["config"].startswith("fork") else "up"
        byp[r["problem"]][arm].add(r["obj"])
    xarm, xok = [], 0
    for p, d in byp.items():
        f_, u_ = d.get("fork", set()), d.get("up", set())
        if f_ and u_:
            if len(f_) == 1 and len(u_) == 1 and f_ == u_: xok += 1
            else: xarm.append(p)
    fails += ["param mismatch %s" % (x,) for x in pmis]
    fails += ["cross-arm obj mismatch %s" % p for p in xarm]
    total_fail += len(fails)
    grand_rows += len(rows); grand_cells += len(cells); grand_skipped += skipped

    print("\n=== %s ===" % h)
    print("  rows %d   cells %d   (declared-but-unexecuted rows skipped: %d)"
          % (len(rows), len(cells), skipped))
    print("  status not ok/partial : %d" % len(badst))
    print("  missing cpuset        : %d" % len(nocpu))
    print("  param-family mismatch : %d" % len(pmis))
    print("  repeat-id anomalies   : %d undocumented, %d documented-and-expected"
          % (len([x for x in fails if "repeat ids" in x]),
             len([a for a in documented_seen if a[0] == h])))
    print("  cross-arm objective   : %d problems identical, %d mismatched" % (xok, len(xarm)))
    print("  INTEGRITY FAILURES    : %d" % len(fails))
    for x in fails[:8]:
        print("      %s" % x)

# A documented anomaly that is MISSING is itself a failure: the archive would no longer match
# its documentation, which is what a tamper-check must refuse to bless.
for key in KNOWN_ANOMALIES:
    if key not in documented_seen:
        total_fail += 1
        print("FAIL: documented anomaly %s NOT FOUND -- archive does not match its documentation"
              % (key,), file=sys.stderr)

print("\n%s\nTOTALS: %d rows, %d cells, %d skipped declared-but-unexecuted"
      % ("=" * 70, grand_rows, grand_cells, grand_skipped))
print("documented anomalies present as declared: %d of %d"
      % (len(documented_seen), len(KNOWN_ANOMALIES)))
print("UNDOCUMENTED INTEGRITY FAILURES ACROSS ALL HOSTS: %d\n%s" % (total_fail, "=" * 70))
if a.fail_on_integrity and total_fail:
    sys.exit(1)
