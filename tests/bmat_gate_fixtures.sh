#!/bin/sh
# MODIFIED from upstream (GPLv2 2a notice), 2026-08-18: NEW FILE, hardened the same day after
# independent review.
#
# Gate-boundary fixture test for the bMat chooser: the released `auto` policy and the opt-in
# `fill` policy, asserted against hand-computable expectations on both sides of every gate,
# at the exact 0.25 and 0.40 boundaries, at gate 1's m/nBlock edges, through gate 2's SDP and
# LP loops (above and exactly at the cutoff), and through the fill policy's gate-4 DENSE
# branch -- every branch REACHABLE from dat-s input (SOCP blocks do not exist in SDPA 7
# input, so the SOCP loop is unreachable and not claimed) (the nonchordal ring, whose
# fill > F for EVERY ordering by the cycle-chord bound). See tests/gen_bmat_gate_fixtures.py.
#
# FAIL-CLOSED, per the 2026-08-18 review:
#   * every solver invocation runs under a watchdog and must exit 0 -- a run that prints the
#     expected line and then fails still fails the cell;
#   * exactly ONE decision record per run, compared as an EXACT string, not a substring;
#   * cells whose route implies the ordering ran must contain exactly one gate-3 and one
#     gate-4 record, and the fill/aggregate invariant is then asserted -- as EQUALITY on the
#     block-diagonal fixtures (chordal: fill cannot exceed aggregate) and as STRICT
#     dominance on the ring;
#   * the cell count is pinned; any shortfall exits nonzero.
#
# COLUMN MEANINGS AFTER THE 2026-08-18 PROMOTION:
#   auto   = the CURRENT default (the derived chooser). These rows are the promotion's contract.
#   legacy = the PRE-PROMOTION released chooser, retained as a mode. These rows are the same
#            expectations the auto column carried before promotion, so the released routing
#            stays a tested contract instead of a memory -- if `legacy` ever stops reproducing
#            it, this suite fails.
#   fill   = explicit synonym for auto; one row proves the synonym, no more.
#
# Usage: tests/bmat_gate_fixtures.sh <path-to-sdpa_gmp> [workdir]
set -u

BIN=${1:?usage: bmat_gate_fixtures.sh <sdpa_gmp> [workdir]}
W=${2:-$(mktemp -d)}
mkdir -p "$W" || { echo "FATAL: cannot create workdir $W"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
FAIL=0
CELLS=0
EXPECTED_CELLS=33
TIMEOUT_S=120

python3 "$HERE/gen_bmat_gate_fixtures.py" "$W" || { echo "FATAL: generator failed"; exit 1; }

cat > "$W/param_it1.sdpa" <<'EOF'
1	unsigned int maxIteration;
1.0E-7	double 0.0 < epsilonStar;
1.0E2   double 0.0 < lambdaStar;
2.0   	double 1.0 < omegaStar;
-1.0E5  double lowerBound;
1.0E5   double upperBound;
0.1     double 0.0 <= betaStar <  1.0;
0.3     double 0.0 <= betaBar  <  1.0, betaStar <= betaBar;
0.9     double 0.0 < gammaStar  <  1.0;
1.0E-15	double 0.0 < epsilonDash;
256     precision
NOPRINT     char*  xPrint
NOPRINT     char*  XPrint
NOPRINT     char*  YPrint
NOPRINT     char*  infPrint
EOF

# Portable watchdog: run "$@", kill after $TIMEOUT_S, return 124 on timeout, else the rc.
run_to() {
    "$@" &
    _pid=$!
    _n=0
    while kill -0 "$_pid" 2>/dev/null; do
        _n=$((_n + 1))
        if [ "$_n" -gt "$TIMEOUT_S" ]; then
            kill -9 "$_pid" 2>/dev/null
            wait "$_pid" 2>/dev/null
            return 124
        fi
        sleep 1
    done
    wait "$_pid"
}

# one <fixture> <mode> <expected-exact-decision> <invariant: none|equal|strict> [exp_agg] [exp_fill]
# exp_agg/exp_fill pin the COUNTS to the generator's arithmetic -- a malformed record whose
# extraction yields a stray token can no longer satisfy the invariant by accident.
one() {
    fx=$1; md=$2; want=$3; inv=$4; exp_agg=${5:-}; exp_fill=${6:-}
    CELLS=$((CELLS + 1))
    log=$W/${fx}_${md}.log
    # md=UNSET runs with SDPA_BMAT_MODE genuinely ABSENT rather than set to "auto". Added
    # 2026-08-19 after review: every cell here, and every corpus and CI check, previously
    # supplied a mode explicitly, and the only unset-mode runs anywhere used inputs (example1,
    # truss6) that route identically under both policies. So nothing in the suite would have
    # failed if an edit made the unset default silently revert to `legacy` -- a blind spot on
    # precisely the change the 2026-08-18 promotion makes.
    if [ "$md" = UNSET ]; then
        set -- -u SDPA_BMAT_MODE
    else
        set -- SDPA_BMAT_MODE="$md"
    fi
    run_to env -u SDPA_BMAT_MAX_GB -u SDPA_BMAT_TEST_BREAK_INVARIANT -u SDPA_BMAT_LOG \
        "$@" SDPA_BMAT_LOG=1 OMP_NUM_THREADS=2 \
        "$BIN" -ds "$W/bmat_gate_${fx}.dat-s" -o "$W/${fx}_${md}.out" -p "$W/param_it1.sdpa" \
        > "$log" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
        echo "FAIL $fx/$md: solver rc=$rc (want 0; 124 = watchdog timeout)"
        FAIL=$((FAIL + 1)); return
    fi
    ndec=$(grep -c "^bMat decision  *:" "$log")
    if [ "$ndec" != 1 ]; then
        echo "FAIL $fx/$md: $ndec decision records, want exactly 1"; FAIL=$((FAIL + 1)); return
    fi
    # First-colon strip, not greedy: a decision tag may itself contain punctuation.
    dec=$(grep "^bMat decision  *:" "$log" | sed 's/^[^:]*: *//')
    if [ "$dec" = "$want" ]; then
        echo "ok   $fx/$md: $dec"
    else
        echo "FAIL $fx/$md: got '$dec', wanted exactly '$want'"; FAIL=$((FAIL + 1)); return
    fi
    if [ "$inv" != none ]; then
        # The route implies the ordering ran: both count records are REQUIRED, exactly once.
        n3=$(grep -c "^bMat gate3 aggregate" "$log"); n4=$(grep -c "^bMat gate4 fill" "$log")
        if [ "$n3" != 1 ] || [ "$n4" != 1 ]; then
            echo "FAIL $fx/$md: gate3/gate4 records $n3/$n4, want exactly 1/1"
            FAIL=$((FAIL + 1)); return
        fi
        # Anchored numeric grammar: the records must parse as their full printed format, and
        # the extracted fields must be integers -- anything else is a malformed record, not a
        # count.
        agg=$(sed -n 's/^bMat gate3 aggregate : \([0-9][0-9]*\) elems \/ m^2 .*$/\1/p' "$log")
        fil=$(sed -n 's/^bMat gate4 fill      : method [0-9][0-9]*, \([0-9][0-9]*\) elems \/ m^2 .*$/\1/p' "$log")
        case "$agg" in *[!0-9]*|"") echo "FAIL $fx/$md: gate3 record malformed"; FAIL=$((FAIL+1)); return;; esac
        case "$fil" in *[!0-9]*|"") echo "FAIL $fx/$md: gate4 record malformed"; FAIL=$((FAIL+1)); return;; esac
        if [ -n "$exp_agg" ] && [ "$agg" != "$exp_agg" ]; then
            echo "FAIL $fx/$md: aggregate $agg != generator arithmetic $exp_agg"; FAIL=$((FAIL+1)); return
        fi
        if [ -n "$exp_fill" ] && [ "$fil" != "$exp_fill" ]; then
            echo "FAIL $fx/$md: fill $fil != expected $exp_fill"; FAIL=$((FAIL+1)); return
        fi
        case "$inv" in
        equal)
            if [ "$fil" = "$agg" ]; then echo "     invariant: fill $fil == aggregate $agg (chordal)"
            else echo "FAIL $fx/$md: chordal fixture must have fill==aggregate, got $fil vs $agg"
                 FAIL=$((FAIL + 1)); fi ;;
        strict)
            if [ "$fil" -gt "$agg" ] 2>/dev/null; then echo "     invariant: fill $fil > aggregate $agg (nonchordal)"
            else echo "FAIL $fx/$md: ring must have fill>aggregate, got $fil vs $agg"
                 FAIL=$((FAIL + 1)); fi ;;
        esac
    fi
}

#   fixture         mode    exact expected decision                                              invariant
# gate 1 -- policy-independent
one g1_smallblocks  auto    "DENSE (gate1 size)"                                                 none
one g1_smallblocks  legacy  "DENSE (gate1 size)"                                                 none
one g1_m100         auto    "DENSE (gate1 size)"                                                 none
one g1_m100         legacy  "DENSE (gate1 size)"                                                 none
one g1_m101         auto    "SPARSE (method 1)"                                                  equal 1721 1721
one g1_m101         legacy  "SPARSE (method 1) [legacy chooser]"                                 equal 1721 1721
one g1_nb5          auto    "DENSE (gate1 size)"                                                 none
one g1_nb5          legacy  "DENSE (gate1 size)"                                                 none
# gate 2 -- frozen at 0.5m under BOTH policies (the whole point of the constant split)
one g2_bigblock     auto    "DENSE (gate2 SDP block 0)"                                          none
one g2_bigblock     legacy  "DENSE (gate2 SDP block 0)"                                          none
one g2_lp           auto    "DENSE (gate2 LP block 0)"                                           none
one g2_lp           legacy  "DENSE (gate2 LP block 0)"                                           none
# gate 2 exactly AT its cutoff (60 > 60 is false) -- and a policy divergence point
one g2_lp_low       auto    "SPARSE (method 1)"                                                  equal 4200 4200
one g2_lp_low       legacy  "DENSE (gate3 aggregate pattern) [legacy chooser]"                    none
# below every threshold -- both agree
one low_sparse      auto    "SPARSE (method 1)"                                                  equal 2700 2700
one low_sparse      legacy  "SPARSE (method 1) [legacy chooser]"                                 equal 2700 2700
# exactly 0.25: legacy's strict inequality holds, so both still route sparse
one boundary_025    auto    "SPARSE (method 1)"                                                  equal 3600 3600
one boundary_025    legacy  "SPARSE (method 1) [legacy chooser]"                                 equal 3600 3600
# THE PROMOTION, on the switch population: default now reaches gate 4 and routes sparse
one switch          auto    "SPARSE (method 1)"                                                  equal 4428 4428
one switch          legacy  "DENSE (gate3 aggregate pattern) [legacy chooser]"                    none
one switch          fill    "SPARSE (method 1)"                                                  equal 4428 4428
# exactly 0.40: BOTH of the new policy's strict inequalities pinned at F
one boundary_040    auto    "SPARSE (method 1)"                                                  equal 5760 5760
one boundary_040    legacy  "DENSE (gate3 aggregate pattern) [legacy chooser]"                    none
# above F: the provable gate-3 skip, and legacy's gate 3 agreeing for its own reason
one high_dense      auto    "DENSE (gate3 aggregate pattern) [gate3 skip: aggregate > F implies fill > F]" none
one high_dense      legacy  "DENSE (gate3 aggregate pattern) [legacy chooser]"                    none
# nonchordal: fill > F for EVERY ordering, so the default reaches gate 4 and says DENSE there
one ring            auto    "DENSE (gate4 ordered fill)"                                         strict 5400 7650
one ring            legacy  "DENSE (gate3 aggregate pattern) [legacy chooser]"                    none

# THE UNSET DEFAULT, on POLICY-DIVERGENT fixtures only. A cell where both policies agree cannot
# detect a reverted default, which is exactly why the pre-2026-08-19 suite could not: run these
# three under `legacy` and every one of them changes its answer.
one switch          UNSET   "SPARSE (method 1)"                                                  equal 4428 4428
one g2_lp_low       UNSET   "SPARSE (method 1)"                                                  equal 4200 4200
one high_dense      UNSET   "DENSE (gate3 aggregate pattern) [gate3 skip: aggregate > F implies fill > F]" none

# ...and unset must agree with an explicit `auto` on everything it COMPUTES. Compared over the
# numerical section only -- the raw files also carry a start timestamp, the -o path, and per-phase
# timings, none of which are results. Same selection CI applies to dense-vs-sparse initial points.
#
# WHAT THESE THREE CELLS DO AND DO NOT CATCH, measured rather than assumed. Sabotaging ONLY the
# unset path to return BMAT_LEGACY makes the three route-string cells above fail and leaves these
# three PASSING -- because the two routes agree numerically to printed precision, which is the
# acceptance round's central finding, not a defect. So the route-string cells are the
# reverted-default detector; these are a weaker complement that would catch unset acquiring some
# THIRD behaviour that changes results. Both are kept, and the distinction is written down here
# so nobody later mistakes a green identity cell for proof that the default is intact.
sel() { grep -aE 'objValPrimal|objValDual|phase\.value|Iteration =' "$1"
        awk '/^xVec *=/{f=1} f && !/time *=/' "$1"; }
for fx in switch g2_lp_low high_dense; do
    CELLS=$((CELLS + 1))
    sel "$W/${fx}_UNSET.out" > "$W/${fx}_UNSET.sel"
    sel "$W/${fx}_auto.out"  > "$W/${fx}_auto.sel"
    # Non-vacuity: an empty selection would make any two cells compare equal.
    if ! grep -q 'objValPrimal' "$W/${fx}_UNSET.sel"; then
        echo "FAIL $fx: unset/auto comparison selected nothing to compare"
        FAIL=$((FAIL + 1))
    elif diff -q "$W/${fx}_UNSET.sel" "$W/${fx}_auto.sel" >/dev/null 2>&1; then
        echo "ok   $fx/unset==auto: $(wc -l < "$W/${fx}_UNSET.sel" | tr -d ' ') result lines identical"
    else
        echo "FAIL $fx: unset-mode results differ from explicit auto"
        diff "$W/${fx}_UNSET.sel" "$W/${fx}_auto.sel" | head -5
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "cells: $CELLS / $EXPECTED_CELLS   failures: $FAIL"
[ "$CELLS" = "$EXPECTED_CELLS" ] || { echo "FAIL: cell count mismatch"; FAIL=$((FAIL + 1)); }
if [ "$FAIL" = 0 ]; then echo "BMAT_GATE_FIXTURES_PASS"; else echo "BMAT_GATE_FIXTURES_FAIL"; fi
exit "$FAIL"
