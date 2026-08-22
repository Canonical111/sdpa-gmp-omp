/* -------------------------------------------------------------

This file is a component of SDPA-C
Copyright (C) 2004 SDPA Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

------------------------------------------------------------- */

#define UseMETIS 0
#define PrintSparsity 0
#define OrderOnlyByMDO 1

#include <sdpa_chordal.h>
// getenv/atof, strcmp, va_list for the bMat mode switch and its decision log below.
// climits for INT_MAX, which bounds the dense bMat allocation (see ordering_bMat);
// cerrno/cmath for the strict SDPA_BMAT_MAX_GB parse.
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

namespace sdpa {

Chordal::Chordal() { initialize(); }

Chordal::~Chordal() {
    //
}

void Chordal::initialize() {
    // condition of sparse computation
    // m_threshold < mDim,
    // b_threshold < nBlock,
    // aggregate_threshold >= aggrigated sparsity ratio
    // extend_threshold    >= extended sparsity ratio
    m_threshold = 100;
    b_threshold = 5;
    aggregate_threshold = 0.25;
    extend_threshold = 0.4;

#if 0 // DENSE computation for debugging
  m_threshold = 10000000;
  b_threshold = 1000000; 
  aggregate_threshold = 0.0; 
  extend_threshold = 0.0;
#endif
#if 0 // SPARSE computation for debugging
  m_threshold = 0;
  b_threshold = 0; 
  aggregate_threshold = 2.0; 
  extend_threshold = 2.0;
#endif

    /* indicates the used ordering method */
    /* 0: METIS 4.0.1 - nested dissection <--- not support */
    /* 1: Spooles 2.2 - mininum degree */
    /* 2: Spooles 2.2 - generalized nested dissection */
    /* 3: Spooles 2.2 - multisection */
    /* 4: Spooles 2.2 - better of 2 and 3 */
#if OrderOnlyByMDO
    Method[0] = 0;
    Method[1] = 1;
    Method[2] = 0;
    Method[3] = 0;
    Method[4] = 0;
#else
    Method[0] = 0;
    Method[1] = 1;
    Method[2] = 1;
    Method[3] = 1;
    Method[4] = 1;
#endif
    best = -1;
}

void Chordal::terminate() {
    if (Method[0]) {
        rError("no support for METIS");
    }
    if (Method[1] > 1) {
        IV_free(newToOldIV_MMD);
        IVL_free(symbfacIVL_MMD);
    }
    if (Method[2] > 1) {
        IV_free(newToOldIV_ND);
        IVL_free(symbfacIVL_ND);
    }
    if (Method[3] > 1) {
        IV_free(newToOldIV_MS);
        IVL_free(symbfacIVL_MS);
    }
    if (Method[4] > 1) {
        IV_free(newToOldIV_NDMS);
        IVL_free(symbfacIVL_NDMS);
    }
}

// marge array1 to array2
void Chordal::margeArray(int na1, int *array1, int na2, int *array2) {

    int ptr = na1 + na2 - 1;
    int ptr1 = na1 - 1;
    int ptr2 = na2 - 1;
    int idx1, idx2;

    while ((ptr1 >= 0) || (ptr2 >= 0)) {

        if (ptr1 >= 0) {
            idx1 = array1[ptr1];
        } else {
            idx1 = -1;
        }
        if (ptr2 >= 0) {
            idx2 = array2[ptr2];
        } else {
            idx2 = -1;
        }
        if (idx1 > idx2) {
            array2[ptr] = idx1;
            ptr1--;
        } else {
            array2[ptr] = idx2;
            ptr2--;
        }
        ptr--;
    }

    // error check
    if (ptr != -1) {
        rMessage("Chordal::margeArray:: program bug");
    }
}

// make aggrigate sparsity pattern
void Chordal::makeGraph(InputData &inputData, int m) {

    int i, j, k, l;
    int SDP_nBlock = inputData.SDP_nBlock;
    int SOCP_nBlock = inputData.SOCP_nBlock;
    int LP_nBlock = inputData.LP_nBlock;

    int *counter;
    counter = new int[m];
    for (int i = 0; i < m; i++) {
        counter[i] = 0;
    }

    // count maximum mumber of index
    for (l = 0; l < SDP_nBlock; l++) {
        int SDP_nConstraint = inputData.SDP_nConstraint[l];
        for (k = 0; k < SDP_nConstraint; k++) {
            i = inputData.SDP_constraint[l][k];
            counter[i] += SDP_nConstraint;
        }
    }
    for (l = 0; l < SOCP_nBlock; l++) {
        int SOCP_nConstraint = inputData.SOCP_nConstraint[l];
        for (k = 0; k < SOCP_nConstraint; k++) {
            i = inputData.SOCP_constraint[l][k];
            counter[i] += SOCP_nConstraint;
        }
    }
    for (l = 0; l < LP_nBlock; l++) {
        int LP_nConstraint = inputData.LP_nConstraint[l];
        for (k = 0; k < LP_nConstraint; k++) {
            i = inputData.LP_constraint[l][k];
            counter[i] += LP_nConstraint;
        }
    }

    // allocate temporaly workspace
    int **tmp;
    tmp = new int *[m];
    for (i = 0; i < m; i++) {
        tmp[i] = new int[counter[i]];
    }

    // merge index
    for (int i = 0; i < m; i++) {
        counter[i] = 0;
    }
    // marge index of for SDP
    for (l = 0; l < SDP_nBlock; l++) {
        for (k = 0; k < inputData.SDP_nConstraint[l]; k++) {
            i = inputData.SDP_constraint[l][k];
            margeArray(inputData.SDP_nConstraint[l], inputData.SDP_constraint[l], counter[i], tmp[i]);
            counter[i] += inputData.SDP_nConstraint[l];
        }
    }
    // marge index of for SOCP
    for (l = 0; l < SOCP_nBlock; l++) {
        for (k = 0; k < inputData.SOCP_nConstraint[l]; k++) {
            i = inputData.SOCP_constraint[l][k];
            margeArray(inputData.SOCP_nConstraint[l], inputData.SOCP_constraint[l], counter[i], tmp[i]);
            counter[i] += inputData.SOCP_nConstraint[l];
        }
    }
    // marge index of for LP
    for (l = 0; l < LP_nBlock; l++) {
        for (k = 0; k < inputData.LP_nConstraint[l]; k++) {
            i = inputData.LP_constraint[l][k];
            margeArray(inputData.LP_nConstraint[l], inputData.LP_constraint[l], counter[i], tmp[i]);
            counter[i] += inputData.LP_nConstraint[l];
        }
    }

    // construct adjacency list of SPOOLES
    IVL_init1(adjIVL, IVL_CHUNKED, m);
    int isize, previous;
    int *ivec;
    ivec = new int[m];
    for (i = 0; i < m; i++) {
        isize = 0;
        previous = -1;
        for (j = 0; j < counter[i]; j++) {
            if (tmp[i][j] != previous) {
                ivec[isize] = tmp[i][j];
                previous = ivec[isize];
                isize++;
            }
        }
        IVL_setList(adjIVL, i, isize, ivec);
    }

    // constract graph of SPOOLES
    Graph_init2(graph, 0, m, 0, IVL_tsize(adjIVL), m, IVL_tsize(adjIVL), adjIVL, NULL, NULL);

    delete[] counter;
    for (int i = 0; i < m; i++) {
        delete[] tmp[i];
    }
    delete[] tmp;
    delete[] ivec;
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-12: the running total is an int, and
   every caller doubles it (`return nonzeros * 2 - m`) before comparing it against
   thresholds. For a large, densely-filled problem that overflows silently and the gate
   decisions are then taken on a negative number.

   The count now RUNS in 64 bits and is validated before it is narrowed. A first attempt
   kept the int accumulator and carried a double alongside to check afterwards; that was no
   fix at all, because the overflowing `nonzeros += ...` executes first and the undefined
   behaviour has already happened by the time the check runs. Widen the arithmetic, then
   validate, then cast -- in that order.

   Unreachable for anything upstream could allocate: the dense limit is m = 46340, and
   reaching this one needs a symbolic factor with more than 2^31 stored entries. */
int Chordal::countNonZero(int m, IVL *symbfacIVL) {
    long long nonzeros = 0;
    bool *bnode;

    // count non-zero element
    rNewCheck();
    bnode = new bool[m];
    if (bnode == NULL) {
        rError("Newton::initialize_sparse_bMat memory exhausted ");
    }
    for (int i = 0; i < m; i++) {
        bnode[i] = false;
    }

    int nClique = IVL_nlist(symbfacIVL);
    int psize;
    int *pivec;
    for (int l = nClique - 1; l >= 0; l--) {
        IVL_listAndSize(symbfacIVL, l, &psize, &pivec);
        for (int i = 0; i < psize; i++) {
            int ii = pivec[i];
            if (bnode[ii] == false) {
                nonzeros += (long long)(psize - i);   /* 64-bit: cannot wrap here */
                bnode[ii] = true;
            }
        }
    }

    delete[] bnode;
    // Validate in 64 bits, THEN narrow. Every caller returns `nonzeros * 2 - m`, so the
    // doubled value is what has to fit, not the count itself.
    if (nonzeros * 2LL - (long long)m > (long long)INT_MAX || nonzeros > (long long)INT_MAX) {
        rError("Chordal::countNonZero: the symbolic factor has " << nonzeros
               << " stored entries, so 2*nnz-m exceeds INT_MAX=" << INT_MAX
               << ". This problem is too large for either bMat representation in this build.");
    }
    return (int)nonzeros;
}

int Chordal::Spooles_MMD(int m) {
    int seed = 0, msglvl = 0;
    FILE *fp = NULL;

    //  rMessage("orderViaMMD:start");
    etree = orderViaMMD(graph, seed, msglvl, fp);
    //  rMessage("orderViaMMD:end");
    newToOldIV_MMD = ETree_newToOldVtxPerm(etree);
    symbfacIVL_MMD = SymbFac_initFromGraph(etree, graph);
    //  IVL_writeForHumanEye(symbfacIVL_MMD,stdout);

    int nonzeros = countNonZero(m, symbfacIVL_MMD);

    return nonzeros * 2 - m;
}

int Chordal::Spooles_NDMS(int m) {
    int seed = 0, msglvl = 0;
    FILE *fp = NULL;

    int maxdomainsize = m / 16 + 1;
    int maxzeros = m / 10 + 1;
    int maxsize = 64;

    //  rMessage("orderViaMMD:start");
    etree = orderViaBestOfNDandMS(graph, maxdomainsize, maxzeros, maxsize, seed, msglvl, fp);
    //  rMessage("orderViaMMD:end");
    newToOldIV_NDMS = ETree_newToOldVtxPerm(etree);
    symbfacIVL_NDMS = SymbFac_initFromGraph(etree, graph);
    //  IVL_writeForHumanEye(symbfacIVL_NDMS,stdout);

    int nonzeros = countNonZero(m, symbfacIVL_NDMS);

    return nonzeros * 2 - m;
}

int Chordal::Spooles_ND(int m) {
    int seed = 0, msglvl = 0;
    FILE *fp = NULL;
    bool *bnode;

    int maxdomainsize = m / 16 + 1;

    //  rMessage("orderViaMMD:start");
    etree = orderViaND(graph, maxdomainsize, seed, msglvl, fp);
    //  rMessage("orderViaMMD:end");
    newToOldIV_ND = ETree_newToOldVtxPerm(etree);
    symbfacIVL_ND = SymbFac_initFromGraph(etree, graph);
    //  IVL_writeForHumanEye(symbfacIVL_ND,stdout);

    int nonzeros = countNonZero(m, symbfacIVL_ND);

    return nonzeros * 2 - m;
}

int Chordal::Spooles_MS(int m) {
    int seed = 0, msglvl = 0;
    FILE *fp = NULL;
    bool *bnode;

    int maxdomainsize = m / 16 + 1;

    //  rMessage("orderViaMMD:start");
    etree = orderViaMS(graph, maxdomainsize, seed, msglvl, fp);
    //  rMessage("orderViaMMD:end");
    newToOldIV_MS = ETree_newToOldVtxPerm(etree);
    symbfacIVL_MS = SymbFac_initFromGraph(etree, graph);
    //  IVL_writeForHumanEye(symbfacIVL_MS,stdout);

    int nonzeros = countNonZero(m, symbfacIVL_MS);

    return nonzeros * 2 - m;
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-11: the sparse/dense choice for
   bMat is observable and overridable. Rationale: this switch decides whether the Schur
   complement is factored by the dense Rpotrf or by Lal::getCholesky, which on the measured
   problems is 92-98% of each iteration -- yet the solver printed only its verdict, never the
   numbers behind it, and the thresholds were compile-time only.

   AMENDED 2026-08-19. Two claims in the original notice have since become false and are
   corrected here rather than left to be discovered:
     - "the SERIAL scalar Lal::getCholesky" -- the sparse Cholesky is threaded now, and that
       fact is the entire reason the thresholds needed recalibrating;
     - "Default behaviour is unchanged and, with SDPA_BMAT_LOG unset, so is the output, byte
       for byte" -- true until the 2026-08-18 promotion, false after it. Unset now selects the
       derived policy; SDPA_BMAT_MODE=legacy is what preserves the released behaviour exactly. */

namespace {

// Bytes per stored mpf_class at the current default precision. Limbs are allocated
// separately, so sizeof() alone undercounts several-fold at 256 bits.
double bmat_bytes_per_elem() {
    const double limbs =
        (double)((mpf_get_default_prec() + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) + 1.0;
    return (double)sizeof(mpf_class) + limbs * (double)sizeof(mp_limb_t) + 16.0;
}

// MemAvailable in GiB, or -1 when it cannot be read (non-Linux, restricted /proc).
// -1 means "unknown". The guard below then proceeds but says so, rather than either
// refusing (which would make forced dense unusable off Linux) or staying quiet about
// having checked nothing. SDPA_BMAT_MAX_GB supplies a cap explicitly.
double bmat_available_gb() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) {
        return -1.0;
    }
    char key[64], unit[16];
    unsigned long val;
    double gb = -1.0;
    while (fscanf(f, "%63s %lu %15s", key, &val, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) {
            gb = (double)val / 1048576.0;
            break;
        }
    }
    fclose(f);
    return gb;
}

/* No BMAT_AUTO enumerator: since the 2026-08-18 promotion the string "auto" SELECTS
   BMAT_FILL, so an enumerator by that name would be a trap -- `mode == BMAT_AUTO` reads as
   "is this the default?" and would be false for every default run. The mode NAME still exists
   and is still the default; it simply is not its own policy any more. */
enum BMatMode { BMAT_DENSE, BMAT_SPARSE, BMAT_FILL, BMAT_LEGACY };

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-18: the DEFAULT chooser is now the
   fill-derived policy. Under it, gate 2 keeps a frozen cutoff of 0.5*m through a constant of
   its own; gate 3 is demoted from an independent 0.25 policy to the early implication of gate
   4's own threshold (aggregate density > extend_threshold proves ordered fill >
   extend_threshold, since symbolic factorisation only adds entries, so the skip cannot change
   the verdict); and gate 4 decides the rest. Derivation, evidence and five review rounds:
   review/GATE3-DECISION-RULE-FINAL-REPORT.md in the recipe repo.

   WHY THE DEFAULT MOVED. On all SEVEN affected structures of the bootstrap switch population --
   those the released rule sent to dense -- sparse won: converged cases 5.0-9.8x faster,
   wall-capped cases bounded at >=2.0-4.4x, 3.5-4.9x less peak memory on the pairs that captured
   it, on two architectures and at 256 and 512 bits, at every one of the nine labelled thread
   points, with no reversal anywhere. 167 of 221 census INSTANCES (221 input files over 10
   structures, not 221 distinct problems) paid that penalty silently.

   AND NOT BECAUSE OF THREADING, which is what an earlier version of this comment said. It read
   "the released 0.25 was calibrated when the sparse Cholesky was serial. It no longer is" --
   i.e. that threading is what invalidated the threshold. Then T=1 was measured on both
   architectures, and sparse still won all four structures single-threaded: 2.60x and 2.62x where
   the dense arm converged, >=2.85-3.06x where it hit its cap, with the dense arm witnessed at
   98-99% of one core. The threshold was therefore already mis-routing these structures in the
   SERIAL regime it was calibrated for. Threading only amplifies the penalty -- the margin grows
   to ~8x by 32-64 threads, mostly because the dense arm's parallel efficiency collapses (99% at
   T=1 to 15% at T=64 on r2439) while sparse keeps scaling.

   WHY the original constant is wrong even serially is NOT established by this campaign, and no
   mechanism is asserted here. What is established is that it is wrong, in every regime measured.

   NOT "sparse always wins", and the counterexample is one of ours: SDPLIB truss5 has an ordered
   fill of 1.0 -- a completely dense factor -- and the released rule sent it to dense, where dense
   IS 1.4x faster. It is the boundary control for F=0.40 against truss6 at fill 0.33, and it is
   why the rule tests fill rather than assuming an answer. An earlier draft of this comment said
   "on every measured problem it still sent to dense, sparse won", which truss5 disproves.

   WHAT THE CHANGE COSTS. Selecting a different factorisation changes results in the last
   digits: measured across four structures at full convergence, printed objectives were
   bit-identical, phases and iteration counts equal, and the solution vectors agreed to
   max|dx|/|x|inf <= 6.3e-38 against a declared 1e-8 tolerance. It is a numerical-behaviour
   change nonetheless, so:

     SDPA_BMAT_MODE=legacy  restores the released chooser EXACTLY -- the pre-promotion gate
                            expressions, including gate 2 derived from
                            m*sqrt(aggregate_threshold) and gate 3 at aggregate_threshold.
                            Anyone reproducing a pre-promotion result should set it.
     SDPA_BMAT_MODE=fill    is retained as an explicit synonym for the new default, so scripts
                            written during the opt-in phase keep working.

   The gate-boundary fixture suite (tests/bmat_gate_fixtures.sh) pins BOTH policies, so the
   released routing remains a tested contract rather than a memory. */

// Gate 2's cutoff under the fill policy. The released chooser derives gate 2 from
// m*sqrt(aggregate_threshold) -- one constant secretly serving two gates, which is exactly
// how a gate-3 retune would have silently moved gate 2. The fill policy freezes gate 2 at
// the same released value through a constant of its own; the two can now move independently.
static const double BMAT_FILL_BLOCK_FRACTION = 0.5;

/* TEST-ONLY, same compile gate as the spchol fault hooks: pretend the ordered-fill count came
   back BELOW the aggregate count, so the fill-policy invariant's error branch is executable in
   CI rather than merely source-audited. Absent unless -DSDPA_SPCHOL_TEST_HOOKS; a build
   without the gate REFUSES the variable rather than ignoring it, because a knob that silently
   does nothing misleads whoever set it. */
static bool bmat_test_break_invariant() {
    const char *e = getenv("SDPA_BMAT_TEST_BREAK_INVARIANT");
    if (e == NULL || e[0] == '\0') {
        return false;
    }
#ifndef SDPA_SPCHOL_TEST_HOOKS
    rError("SDPA_BMAT_TEST_BREAK_INVARIANT is a test hook and this binary was not built"
           " with SDPA_SPCHOL_TEST_HOOKS");
#endif
    return strcmp(e, "0") != 0;
}

BMatMode bmat_mode() {
    const char *e = getenv("SDPA_BMAT_MODE");
    // Unset and "auto" both mean the current default policy, which is now the fill-derived
    // one. "fill" is its explicit synonym, kept so opt-in-era scripts keep working.
    if (e == NULL || e[0] == '\0' || strcmp(e, "auto") == 0 || strcmp(e, "fill") == 0) {
        return BMAT_FILL;
    }
    if (strcmp(e, "dense") == 0) {
        return BMAT_DENSE;
    }
    if (strcmp(e, "sparse") == 0) {
        return BMAT_SPARSE;
    }
    // The pre-promotion chooser, exactly.
    if (strcmp(e, "legacy") == 0) {
        return BMAT_LEGACY;
    }
    // Reject rather than fall back: silently ignoring a typo here would measure the
    // wrong configuration and look like a result.
    rError("SDPA_BMAT_MODE must be auto, dense, sparse, fill or legacy (got \"" << e
           << "\"). auto is the current policy; legacy is the pre-2026-08 chooser;"
           << " fill is an explicit synonym for auto.");
    return BMAT_FILL;
}

bool bmat_log() {
    const char *e = getenv("SDPA_BMAT_LOG");
    return e != NULL && e[0] != '\0' && strcmp(e, "0") != 0;
}

// SDPA_BMAT_MAX_GB in GiB, or -1.0 when unset. A cap is a safety limit, so every way of
// writing one that we cannot honour is an error rather than a silent no-cap: atof() would
// return 0.0 for "64GB", "sixty", "" and "-1" alike, and the caller who asked to be
// protected would run unprotected without a word. Same contract as SDPA_BMAT_MODE, which
// rejects a typo rather than quietly meaning "auto".
double bmat_max_gb() {
    const char *e = getenv("SDPA_BMAT_MAX_GB");
    if (e == NULL) {
        return -1.0;                       /* absent: no cap requested */
    }
    // Present but empty is an ERROR, not "unset". `SDPA_BMAT_MAX_GB="$SOME_TYPO"` is the
    // ordinary way an empty value arrives in a job script, and treating it as no-cap fails
    // OPEN on a safety limit -- the caller believes a ceiling is in force and none is. Only
    // an absent variable means "no ceiling wanted".
    if (e[0] == '\0') {
        rError("SDPA_BMAT_MAX_GB is set but empty. Unset it to run without a cap; an empty"
               " value is refused rather than treated as no cap, because it usually means an"
               " unset shell variable was expanded into it.");
    }
    errno = 0;
    char *end = NULL;
    const double v = strtod(e, &end);
    while (end != NULL && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        end++;
    }
    if (end == e || end == NULL || *end != '\0' || errno == ERANGE) {
        rError("SDPA_BMAT_MAX_GB must be a number of GiB (got \"" << e << "\")");
    }
    // Rejects NaN (every comparison false, so it would disable the cap) and infinity (a cap
    // nothing can exceed, i.e. also no cap). Zero and negative are refused because a cap of
    // "nothing may be allocated" is far more likely a typo or an arithmetic accident in a
    // job script than a request, and treating it as no-cap is the dangerous reading.
    if (!(v > 0.0) || !(v < HUGE_VAL)) {
        rError("SDPA_BMAT_MAX_GB must be a finite, positive number of GiB (got \"" << e << "\")");
    }
    return v;
}

void bmat_say(FILE *fpOut, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    printf("%s", buf);
    if (fpOut != NULL) {
        fprintf(fpOut, "%s", buf);
    }
}

} // namespace

void Chordal::ordering_bMat(int m, int nBlock, InputData &inputData, FILE *fpOut) {
    const BMatMode mode = bmat_mode();
    const bool blog = bmat_log();

    /* The derived policy's two departures from the RELEASED chooser, expressed as per-mode
       cutoffs so that the released expressions survive verbatim on the legacy path (the
       ternaries select between the two; under `legacy` every arithmetic expression below is
       the released one, bit for bit):
         gate 2:  the derived policy uses its own frozen 0.5*m;
                  legacy keeps m*sqrt(aggregate_threshold).
         gate 3:  the derived policy uses extend_threshold (gate 4's F -- the
                  early-implication skip); legacy keeps the released aggregate_threshold.
       Before the promotion this comment said the same thing with `auto` and `fill` swapped
       into these roles, which is how it read while auto WAS the released chooser. Naming the
       policies rather than the mode strings is what keeps it true through a default change. */
    // Every mode except the explicit legacy request uses the derived cutoffs. The forced
    // dense/sparse modes never consult them, so this is exact rather than merely harmless.
    const bool fill_policy = (mode != BMAT_LEGACY);
    // Evaluated unconditionally so a build without the test gate refuses the hook variable in
    // EVERY mode -- a knob that is silently inert in auto would mislead whoever set it.
    const bool bmat_break_inv = bmat_test_break_invariant();
    const double gate2_cut = fill_policy ? (BMAT_FILL_BLOCK_FRACTION * m)
                                         : (m * sqrt(aggregate_threshold));
    const double gate3_cut = fill_policy ? extend_threshold : aggregate_threshold;

    // Largest per-block constraint count, for gate 2. Computed unconditionally only
    // when logging; the gates themselves keep their original short-circuit order.
    if (blog) {
        int maxc = 0;
        for (int b = 0; b < inputData.SDP_nBlock; b++)
            if (inputData.SDP_nConstraint[b] > maxc) maxc = inputData.SDP_nConstraint[b];
        for (int b = 0; b < inputData.SOCP_nBlock; b++)
            if (inputData.SOCP_nConstraint[b] > maxc) maxc = inputData.SOCP_nConstraint[b];
        for (int b = 0; b < inputData.LP_nBlock; b++)
            if (inputData.LP_nConstraint[b] > maxc) maxc = inputData.LP_nConstraint[b];
        bmat_say(fpOut, "bMat decision: mode=%s m=%d nBlock=%d\n",
                 mode == BMAT_DENSE    ? "dense"
                 : mode == BMAT_SPARSE ? "sparse"
                 : mode == BMAT_LEGACY ? "legacy"
                                       : "auto",
                 m, nBlock);
        bmat_say(fpOut, "bMat gate1 size      : m<=%d or nBlock<=%d -> %s\n",
                 m_threshold, b_threshold,
                 ((m <= m_threshold) || (nBlock <= b_threshold)) ? "DENSE" : "pass");
        bmat_say(fpOut, "bMat gate2 per-block : max nConstraint %d vs %.1f -> %s\n",
                 maxc, gate2_cut, (maxc > gate2_cut) ? "DENSE" : "pass");
    }

    // A dense bMat has to clear TWO independent feasibility tests, and only the second is
    // about memory:
    //   (1) can m*m be REPRESENTED?  DenseMatrix::initialize computes
    //       `int length = nRow * nCol` (sdpa_struct.cpp), so m >= 46341 overflows signed
    //       int and reaches `new mpf_class[negative]`. No amount of RAM helps -- and the
    //       memory test will not catch it first, because at 80 B/element m=46341 is only
    //       172 GiB, which fits on a 256 GB node.
    //   (2) does it FIT?  -- the cap test inside the forced-dense branch below.
    // (1) is checked here for BOTH paths: the automatic gates select dense too, so
    // refusing only in the forced branch would just route an unrepresentable request
    // through gate 1 or gate 2 into the same overflow.
    const double bmat_elems = (double)m * (double)m;
    const bool bmat_representable = (bmat_elems <= (double)INT_MAX);

    // The cap binds on EVERY path that can select a dense bMat, not just the forced one.
    // It used to bind only inside the forced-dense branch, which made it advertising
    // rather than a limit: the branch printed REFUSED, fell through to the automatic
    // policy, and gate 1 selected dense again on the next line -- allocating exactly the
    // memory the caller had just been told would not be allocated. Anything that reports a
    // ceiling has to be enforced against the decision, not against one route to it.
    const double bmat_need_gb = bmat_elems * bmat_bytes_per_elem() / 1073741824.0;
    const double bmat_cap_gb = bmat_max_gb();
    const bool bmat_over_cap = (bmat_cap_gb > 0.0 && bmat_need_gb > bmat_cap_gb);

    if (mode == BMAT_DENSE) {
        // Asking for dense is the caller's decision, and running out of memory is the
        // caller's risk to take: an OOM kill is a defined, diagnosable outcome, and
        // second-guessing the request would hand back the very algorithm the caller set
        // this variable to avoid. So the memory figures are REPORTED, not enforced,
        // unless SDPA_BMAT_MAX_GB opts in to a cap.
        //
        // Representability is not in that category. m*m overflowing signed int is
        // undefined behaviour, not exhaustion -- there is no outcome to consent to. It is
        // refused unconditionally, and cannot be waived by SDPA_BMAT_MAX_GB.
        if (!bmat_representable) {
            rError("SDPA_BMAT_MODE=dense: m=" << m << " needs m*m=" << bmat_elems
                   << " elements, past the INT_MAX=" << INT_MAX << " limit of the dense bMat"
                   << " allocation (DenseMatrix::initialize computes int nRow*nCol). This is"
                   << " not a memory limit and raising SDPA_BMAT_MAX_GB will not help."
                   << " Unset SDPA_BMAT_MODE to let the automatic policy try the sparse path,"
                   << " which is representable only if the factor's own non-zero count fits in"
                   << " an int -- it is checked, not assumed.");
        }
        const double need_gb = bmat_need_gb;
        const double avail_gb = bmat_available_gb();
        const double cap_gb = bmat_cap_gb;
        if (blog) {
            bmat_say(fpOut, "bMat forced dense    : need %.2f GiB (m*m=%.0f elems, %.0f B each)"
                            ", available %.2f GiB, cap %s\n",
                     need_gb, bmat_elems, bmat_bytes_per_elem(), avail_gb,
                     cap_gb > 0.0 ? "set" : "none (SDPA_BMAT_MAX_GB unset)");
        }
        if (bmat_over_cap) {
            // Two instructions that contradict each other -- "use dense" and "never
            // allocate more than N GiB" -- so honour the safety limit over the preference,
            // say so loudly, and let the fall-through below select sparse. Dense stays
            // DISABLED for the automatic gates; that is the whole point, and its absence
            // is what made this cap bypassable. Always printed, not gated on
            // SDPA_BMAT_LOG.
            bmat_say(fpOut, "bMat forced dense REFUSED: %.2f GiB needed exceeds the"
                            " SDPA_BMAT_MAX_GB cap of %.2f GiB; dense is now DISABLED for"
                            " this solve and the automatic policy will select sparse\n",
                     need_gb, cap_gb);
        } else {
            if (cap_gb <= 0.0 && avail_gb > 0.0 && need_gb > avail_gb) {
                // Proceeding as instructed, but say so plainly: if the job is killed a few
                // seconds from now, this line is the explanation. Always printed.
                bmat_say(fpOut, "bMat forced dense    : WARNING need %.2f GiB but only %.2f GiB"
                                " available; proceeding as requested (set SDPA_BMAT_MAX_GB to"
                                " refuse instead)\n", need_gb, avail_gb);
            }
            if (blog) {
                bmat_say(fpOut, "bMat decision        : DENSE (forced by SDPA_BMAT_MODE)\n");
            }
            best = -1;
            return;
        }
    }

    // The automatic gates below select dense too, so BOTH limits have to apply to them --
    // otherwise each is merely a limit on one route to the allocation rather than a limit
    // on the allocation. An unrepresentable m would reach the same `new mpf_class[negative]`
    // through gate 1 or gate 2, and an over-cap request would allocate through them the
    // memory the forced branch had just refused. Where the caller asked for nothing in
    // particular the response is to take the algorithm that is allowed rather than to fail,
    // so a false `dense_possible` falls through to sparse.
    const bool force_sparse = (mode == BMAT_SPARSE);
    const bool dense_possible = bmat_representable && !bmat_over_cap;

    if (!bmat_representable) {
        // Says only that dense is unavailable. It deliberately does NOT promise the sparse
        // path will work: the factor's own counts are int too (Chordal::countNonZero,
        // Newton::initialize_sparse_bMat's diagonalIndex, SparseMatrix::NonZeroNumber), so a
        // problem this large is representable only if its factor is genuinely sparse. Both
        // counts are now checked at their accumulation points and fail with a diagnostic
        // rather than overflowing, so "attempted" is honest where "guaranteed" would not be.
        bmat_say(fpOut, "bMat note            : m=%d makes m*m=%.0f exceed INT_MAX=%d, so a dense"
                        " bMat cannot be allocated; the sparse path will be attempted and will"
                        " report a clear failure if its own non-zero count also exceeds"
                        " INT_MAX\n", m, bmat_elems, INT_MAX);
    } else if (bmat_over_cap && mode != BMAT_DENSE) {
        // The forced-dense branch has already printed its own REFUSED line; this is the
        // case where the caller set only a cap and let the policy choose.
        bmat_say(fpOut, "bMat note            : a dense bMat would need %.2f GiB, over the"
                        " SDPA_BMAT_MAX_GB cap of %.2f GiB; dense is DISABLED and the"
                        " automatic gates will select sparse\n", bmat_need_gb, bmat_cap_gb);
    }

    if (!force_sparse && dense_possible && ((m <= m_threshold) || (nBlock <= b_threshold))) {
        if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate1 size)\n");
        best = -1;
        return;
    }
    if (!force_sparse && dense_possible) {
        for (int b = 0; b < inputData.SDP_nBlock; b++) {
            if (inputData.SDP_nConstraint[b] > gate2_cut) {
                if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate2 SDP block %d)\n", b);
                best = -1;
                return;
            }
        }
        for (int b = 0; b < inputData.SOCP_nBlock; b++) {
            if (inputData.SOCP_nConstraint[b] > gate2_cut) {
                if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate2 SOCP block %d)\n", b);
                best = -1;
                return;
            }
        }
        for (int b = 0; b < inputData.LP_nBlock; b++) {
            if (inputData.LP_nConstraint[b] > gate2_cut) {
                if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate2 LP block %d)\n", b);
                best = -1;
                return;
            }
        }
    }

    adjIVL = IVL_new();
    graph = Graph_new();

    makeGraph(inputData, m);

    if (blog) {
        bmat_say(fpOut, "bMat gate3 aggregate : %d elems / m^2 %.0f = %.6f vs %.2f -> %s\n",
                 IVL_tsize(adjIVL), (double)m * (double)m,
                 (double)IVL_tsize(adjIVL) / ((double)m * (double)m), gate3_cut,
                 (IVL_tsize(adjIVL) > gate3_cut * m * m) ? "DENSE" : "pass");
    }

    // Kept for the fill policy's gate-4 invariant check: adjIVL's total is the aggregate
    // count in the same symmetric-with-diagonal convention as Method[best] = 2*nnz(L)-m.
    const int bmat_agg_tsize = IVL_tsize(adjIVL);

    if (!force_sparse && dense_possible && IVL_tsize(adjIVL) > gate3_cut * m * m) {
        if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate3 aggregate pattern)%s\n",
                           fill_policy ? " [gate3 skip: aggregate > F implies fill > F]" : " [legacy chooser]");
        best = -1;
        Graph_free(graph);
        return;
    }
#if PrintSparsity
    /* print sparsity information */
    printf("dense matrix               :\t\t\t%14d elements\n", m * m);
    fprintf(fpOut, "dense matrix               :\t\t\t%14d elements\n", m * m);
    printf("aggregate sparsity pattern :\t\t\t%14d elements\n", IVL_tsize(adjIVL));
    fprintf(fpOut, "aggregate sparsity pattern :\t\t\t%14d elements\n", IVL_tsize(adjIVL));
#endif

    /* Uses METIS */
    if (Method[0]) {
        rError("no support for METIS");
    }

    /* Uses Spooles */
    if (Method[1]) { /* Spooles 2.2 - minimum degree */
        Method[1] = Spooles_MMD(m);
        ETree_free(etree);
#if PrintSparsity
        printf("\tSpooles2.2 (minimum degree)\t\t%14d elements\n", Method[1]);
        fprintf(fpOut, "\tSpooles2.2 (minimum degree)\t\t%14d elements\n", Method[1]);
#endif
    }
    if (Method[2]) { /* Spooles 2.2 - generalized nested dissection */
        Method[2] = Spooles_ND(m);
        ETree_free(etree);
#if PrintSparsity
        printf("\tSpooles2.2 (generalized nested dissection)%12d elements\n", Method[2]);
        fprintf(fpOut, "\tSpooles2.2 (generalized nested dissection)%12d elements\n", Method[2]);
#endif
    }
    if (Method[3]) { /* Spooles 2.2 - multisection */
        Method[3] = Spooles_MS(m);
        ETree_free(etree);
#if PrintSparsity
        printf("\tSpooles2.2 (multisection)\t\t%14d elements\n", Method[3]);
        fprintf(fpOut, "\tSpooles2.2 (multisection)\t\t%14d elements\n", Method[3]);
#endif
    }
    if (Method[4]) { /* Spooles 2.2 - best between nested
                        dissection and multisection */
        Method[4] = Spooles_NDMS(m);
        ETree_free(etree);
#if PrintSparsity
        printf("\tSpooles2.2 (best of ND and MS)\t\t%14d elements\n", Method[4]);
        fprintf(fpOut, "\tSpooles2.2 (best of ND and MS)\t\t%14d elements\n", Method[4]);
#endif
    }
    /* Select the best ordering */

    Graph_free(graph);

    best = Best_Ordering(Method);

    if (blog) {
        bmat_say(fpOut, "bMat gate4 fill      : method %d, %d elems / m^2 = %.6f vs %.2f -> %s\n",
                 best, Method[best], (double)Method[best] / ((double)m * (double)m),
                 extend_threshold,
                 (Method[best] > extend_threshold * m * m) ? "DENSE" : "pass");
    }

    /* FILL-POLICY INVARIANT (every mode that uses the derived policy -- since the 2026-08-18
       promotion that includes `auto`, the default; `legacy` never reaches it). The gate-3' skip
       above is sound because symbolic factorisation
       only ADDS entries: Method[best] = 2*nnz(L)-m must dominate the aggregate count in the
       same symmetric-with-diagonal convention. If it ever does not, the two counts are not in
       the units the theorem needs and the skip could mis-route -- that is a build defect, not
       a data condition, so it is a hard error rather than a warning that scrolls away. */
    const int bmat_fill_for_invariant =
        bmat_break_inv ? (bmat_agg_tsize - 1) : Method[best];
    if (fill_policy && bmat_fill_for_invariant < bmat_agg_tsize) {
        // The diagnostic prints the value that was COMPARED. The first version printed
        // Method[best] here, so under the test injection it reported a real fill count as
        // "below" an aggregate it exceeded -- a lying error message, caught in review.
        rError("bMat chooser: ordered fill count " << bmat_fill_for_invariant
               << (bmat_break_inv ? " (TEST INJECTION ACTIVE: real count was " : " (real count ")
               << Method[best] << ")"
               << " is below the aggregate count " << bmat_agg_tsize
               << " -- the fill>=aggregate invariant that justifies the gate-3 skip is"
               << " violated, which means the two counts are no longer in the same units."
               << " This is a solver build defect; run with SDPA_BMAT_MODE=legacy"
               << " (the pre-2026-08 chooser, which does not consult this count) and report it.");
    }

    if (!force_sparse && dense_possible && Method[best] > extend_threshold * m * m) {
        if (blog) bmat_say(fpOut, "bMat decision        : DENSE (gate4 ordered fill)%s\n",
                           fill_policy ? "" : " [legacy chooser]");
        best = -1;
    } else if (blog) {
        bmat_say(fpOut, "bMat decision        : SPARSE (method %d)%s\n", best,
                 force_sparse      ? " [forced by SDPA_BMAT_MODE]"
                 : bmat_over_cap   ? " [dense disabled by SDPA_BMAT_MAX_GB]"
                 : !bmat_representable ? " [dense not representable]"
                 : !fill_policy    ? " [legacy chooser]"
                                   : "");
    }
}

int Chordal::Best_Ordering(int *Method)
/************************************************************************
        Determine the best ordering.
************************************************************************/
{
    int i, best;

    for (i = 0; Method[i] == 0; i++)
        ;
    best = i++;
    while (i < 5) {
        for (; i < 5; i++) {
            if (Method[i] != 0)
                break;
        }
        if (i < 5) {
            if (Method[i] < Method[best])
                best = i;
            i++;
        }
    }
    return best;
}

} // namespace sdpa
