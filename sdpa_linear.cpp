/* -------------------------------------------------------------

This file is a component of SDPA
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

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-03: fatal eigensolver failure exits non-zero; rdpotf2_ returns on all paths. See git log. */
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: the sparse Schur Cholesky reports a non-positive pivot as FAILURE instead of silently zeroing it and returning success. See git log. */
#include <sdpa_linear.h>
#include <sdpa_dataset.h>
/* Threading of the sparse Schur-complement Cholesky (see review/ in the recipe repo).
   cstdint for the 64-bit work counters -- the dE4 update count is ~3e9 and overflows
   signed 32-bit; cstdlib/cstring for the mode switch; omp.h only under _OPENMP. */
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdpa {

mpf_class Lal::getMinEigen(DenseMatrix &lMat, DenseMatrix &xMat, DenseMatrix &Q, Vector &out, Vector &b, Vector &r, Vector &q, Vector &qold, Vector &w, Vector &tmp, Vector &diagVec, Vector &diagVec2, Vector &workVec) {
    mpf_class alpha, beta, value;
    mpf_class min = 1.0e+51, min_old = 1.0e+52, min_min = 1.0e+50;
    mpf_class error = 1.0e+10;
    mpf_class MONE = 1.0;

    int nDim = xMat.nRow;
    int k = 0, kk = 0;

    diagVec.initialize(min_min);
    diagVec2.setZero();
    q.setZero();
    r.initialize(MONE);
    beta = sqrt((mpf_class)nDim); // norm of "r"

    // nakata 2004/12/12
    while (k < nDim && k < sqrt((mpf_class)nDim) + 10 && beta > 1.0e-16 &&
           (abs(min - min_old) > (1.0e-5) * abs(min) + (1.0e-8)
            // && (fabs(min-min_old) > (1.0e-3)*fabs(min)+(1.0e-6)
            || abs(error * beta) > (1.0e-2) * abs(min) + (1.0e-4))) {
        // rMessage("k = " << k);
        qold.copyFrom(q);
        value = MONE / beta;
        Lal::let(q, '=', r, '*', &value);

        // w = (lMat^T)*q
        w.copyFrom(q);
        Rtrmv("Lower", "Transpose", "NotUnit", nDim, lMat.de_ele, nDim, w.ele, 1);

        Lal::let(tmp, '=', xMat, '*', w);
        w.copyFrom(tmp);
        Rtrmv("Lower", "NoTranspose", "NotUnit", nDim, lMat.de_ele, nDim, w.ele, 1);
        // w = lMat*xMat*(lMat^T)*q
        // rMessage("w = ");
        // w.display();
        Lal::let(alpha, '=', q, '.', w);
        diagVec.ele[k] = alpha;
        Lal::let(r, '=', w, '-', q, &alpha);
        Lal::let(r, '=', r, '-', qold, &beta);
        // rMessage("r = ");
        // r.display();

        if (kk >= sqrt((mpf_class)k) || k == nDim - 1 || k > sqrt((mpf_class)nDim + 9)) {
            kk = 0;
            out.copyFrom(diagVec);
            b.copyFrom(diagVec2);
            out.ele[nDim - 1] = diagVec.ele[k];
            b.ele[nDim - 1] = 0.0;

            // rMessage("out = ");
            // out.display();
            // rMessage("b = ");
            // b.display();

            mplapackint info;
            int kp1 = k + 1;
            Rsteqr("I_withEigenvalues", kp1, out.ele, b.ele, Q.de_ele, Q.nRow, workVec.ele, info);

            if (info < 0) {
                rError(" rLanczos :: bad argument " << -info << " Q.nRow = " << Q.nRow << ": nDim = " << nDim << ": kp1 = " << kp1);
            } else if (info > 0) {
                rMessage(" rLanczos :: cannot converge " << info);
                break;
            }

            // rMessage("out = ");
            // out.display();
            // rMessage("Q = ");
            // Q.display();

            min_old = min;
#if 0
      min = 1.0e+50;
      error = 1.0e+10;
      for (int i=0; i<k+1; ++i) {
	if (min>out.ele[i]){
	  min = out.ele[i];
	  error = Q.de_ele[k+Q.nCol*i];
	}
      }
#else
            // out have eigen values with ascending order.
            min = out.ele[0];
            error = Q.de_ele[k];
#endif

        } // end of 'if ( kk>=sqrt(k) ...)'
        // printf("\n");

        Lal::let(value, '=', r, '.', r);
        beta = sqrt(value);
        diagVec2.ele[k] = beta;
        ++k;
        ++kk;
    } // end of while
    // rMessage("k = " << k);
    return min - abs(error * beta);
}

mpf_class Lal::getMinEigenValue(DenseMatrix &aMat, Vector &eigenVec, Vector &workVec) {
    // aMat is rewritten.
    // aMat must be symmetric.
    // eigenVec is the space of eigen values
    // and needs memory of length aMat.nRow
    // workVec is temporary space and needs
    // 3*aMat.nRow-1 length memory.
    mplapackint N = aMat.nRow;
    mplapackint LWORK, info;
    switch (aMat.type) {
    case DenseMatrix::DENSE:
        LWORK = 3 * N - 1;
        // "N" means that we need not eigen vectors
        // "L" means that we refer only lower triangular.
        Rsyev("NonVectors", "Lower", N, aMat.de_ele, N, eigenVec.ele, workVec.ele, LWORK, info);
        if (info != 0) {
            if (info < 0) {
                rMessage("getMinEigenValue:: info is mistaken " << info);
            } else {
                rMessage("getMinEigenValue:: cannot decomposition");
            }
            exit(EXIT_FAILURE);
            return 0.0;
        }
        return eigenVec.ele[0];
        // Eigen values are sorted by ascending order.
        break;
    case DenseMatrix::COMPLETION:
        rError("DenseMatrix:: no support for COMPLETION");
        break;
    }
    return 0.0;
}

bool Lal::getInnerProduct(mpf_class &ret, Vector &aVec, Vector &bVec) {
    int N = aVec.nDim;
    if (N != bVec.nDim) {
        rError("getInnerProduct:: different memory size");
    }
    ret = Rdot(N, aVec.ele, 1, bVec.ele, 1);

    return _SUCCESS;
}

bool Lal::getInnerProduct(mpf_class &ret, BlockVector &aVec, BlockVector &bVec) {
    if (aVec.nBlock != bVec.nBlock) {
        rError("getInnerProduct:: different memory size");
    }
    bool total_judge = _SUCCESS;
    ret = 0.0;
    mpf_class tmp_ret;
    for (int l = 0; l < aVec.nBlock; ++l) {
        bool judge = getInnerProduct(tmp_ret, aVec.ele[l], bVec.ele[l]);
        ret += tmp_ret;
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }
    return total_judge;
}

bool Lal::getInnerProduct(mpf_class &ret, DenseMatrix &aMat, DenseMatrix &bMat) {
    if (aMat.nRow != bMat.nRow || aMat.nCol != bMat.nCol) {
        rError("getInnerProduct:: different memory size");
    }
    int length;
    switch (aMat.type) {
    case DenseMatrix::DENSE:
        length = aMat.nRow * aMat.nCol;
        ret = Rdot(length, aMat.de_ele, 1, bMat.de_ele, 1);
        break;
    case DenseMatrix::COMPLETION:
        rError("DenseMatrix:: no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::getInnerProduct(mpf_class &ret, SparseMatrix &aMat, DenseMatrix &bMat) {
    if (aMat.nRow != bMat.nRow || aMat.nCol != bMat.nCol) {
        rError("getInnerProduct:: different memory size");
    }
    int length;
    int amari, shou;
    mpf_class temp;
    mpf_class value1, value2, value3, value4;
    mpf_class ret1, ret2, ret3, ret4;

    switch (aMat.type) {
    case SparseMatrix::SPARSE:
        // Attension: in SPARSE case, only half elements
        // are stored. And bMat must be DENSE case.
        ret = 0.0;
// rMessage("aMat.NonZeroCount == " << aMat.NonZeroCount);
#if 0
    for (int index=0; index<aMat.NonZeroCount; ++index) {
      int        i = aMat.row_index   [index];
      int        j = aMat.column_index[index];
      mpf_class value = aMat.sp_ele      [index];
      // rMessage("i=" << i << "  j=" << j);
      if (i==j) {
	ret+= value*bMat.de_ele[i+bMat.nRow*j];
      } else {
	ret+= value*(bMat.de_ele[i+bMat.nRow*j]
		     + bMat.de_ele[j+bMat.nRow*i]);

      }
    }
#else
#ifdef _OPENMP
//#pragma omp parallel for
        for (int index = 0; index < aMat.NonZeroCount; ++index) {
            int i = aMat.row_index[index];
            int j = aMat.column_index[index];
            mpf_class value = aMat.sp_ele[index];
            mpf_class temp = 0.0;

            if (i == j) {
                temp += value;
                temp *= bMat.de_ele[i + bMat.nRow * j];
            } else {
                temp += bMat.de_ele[i + bMat.nRow * j];
                temp += bMat.de_ele[j + bMat.nRow * i];
                temp *= value;
            }

//#pragma omp critical
            ret += temp;
        }
#else
        amari = aMat.NonZeroCount % 4;
        shou = aMat.NonZeroCount / 4;
        for (int index = 0; index < amari; ++index) {
            int i = aMat.row_index[index];
            int j = aMat.column_index[index];
            mpf_class value = aMat.sp_ele[index];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                temp = value;
                temp *= bMat.de_ele[i + bMat.nRow * j];
                ret += temp;
            } else {
                temp = bMat.de_ele[i + bMat.nRow * j];
                temp += bMat.de_ele[j + bMat.nRow * i];
                temp *= value;
                ret += temp;
            }
        }
        for (int index = amari, counter = 0; counter < shou; ++counter, index += 4) {
            int i1 = aMat.row_index[index];
            int j1 = aMat.column_index[index];
            value1 = aMat.sp_ele[index];
            // rMessage("i=" << i << "  j=" << j);
            if (i1 == j1) {
                ret1 = value1;
                ret1 *= bMat.de_ele[i1 + bMat.nRow * j1];
            } else {
                ret1 = bMat.de_ele[i1 + bMat.nRow * j1];
                ret1 += bMat.de_ele[j1 + bMat.nRow * i1];
                ret1 *= value1;
            }
            int i2 = aMat.row_index[index + 1];
            int j2 = aMat.column_index[index + 1];
            value2 = aMat.sp_ele[index + 1];
            ret2 = 0.0;
            // rMessage("i=" << i << "  j=" << j);
            if (i2 == j2) {
                ret2 = value2;
                ret2 *= bMat.de_ele[i2 + bMat.nRow * j2];
            } else {
                ret2 = bMat.de_ele[i2 + bMat.nRow * j2];
                ret2 += bMat.de_ele[j2 + bMat.nRow * i2];
                ret2 *= value2;
            }
            int i3 = aMat.row_index[index + 2];
            int j3 = aMat.column_index[index + 2];
            value3 = aMat.sp_ele[index + 2];
            ret3 = 0.0;
            // rMessage("i=" << i << "  j=" << j);
            if (i3 == j3) {
                ret3 = value3;
                ret3 *= bMat.de_ele[i3 + bMat.nRow * j3];
            } else {
                ret3 = bMat.de_ele[i3 + bMat.nRow * j3];
                ret3 += bMat.de_ele[j3 + bMat.nRow * i3];
                ret3 *= value3;
            }
            int i4 = aMat.row_index[index + 3];
            int j4 = aMat.column_index[index + 3];
            value4 = aMat.sp_ele[index + 3];
            ret4 = 0.0;
            // rMessage("i=" << i << "  j=" << j);
            if (i4 == j4) {
                ret4 = value4;
                ret4 *= bMat.de_ele[i4 + bMat.nRow * j4];
            } else {
                ret4 = bMat.de_ele[i4 + bMat.nRow * j4];
                ret4 += bMat.de_ele[j4 + bMat.nRow * i4];
                ret4 *= value4;
            }
            ret += ret1;
            ret += ret2;
            ret += ret3;
            ret += ret4;
        }
#endif
#endif
        break;
    case SparseMatrix::DENSE:
        length = aMat.nRow * aMat.nCol;
        ret = Rdot(length, aMat.de_ele, 1, bMat.de_ele, 1);
        break;
    }
    return _SUCCESS;
}

bool Lal::getCholesky(DenseMatrix &retMat, DenseMatrix &aMat) {
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.type != aMat.type) {
        rError("getCholesky:: different memory size");
    }
    int length, shou, amari;
    mplapackint info;
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        length = retMat.nRow * retMat.nCol;
        Rcopy(length, aMat.de_ele, 1, retMat.de_ele, 1);
#if 1
        Rpotrf("Lower", retMat.nRow, retMat.de_ele, retMat.nRow, info);
#else
        info = choleskyFactorWithAdjust(retMat);
#endif
        if (info != 0) {
            rMessage("cannot cholesky decomposition");
            rMessage("Could you try with smaller gammaStar?");
            return FAILURE;
        }
// Make matrix as lower triangular matrix
#if 0
    for (int j=0; j<retMat.nCol; ++j) {
      for (int i=0; i<j; ++i) {
	retMat.de_ele[i+retMat.nCol*j] = 0.0;
      }
    }
#else
        for (int j = 0; j < retMat.nCol; ++j) {
            shou = j / 4;
            amari = j % 4;
            for (int i = 0; i < amari; ++i) {
                retMat.de_ele[i + retMat.nCol * j] = 0.0;
            }
            for (int i = amari, count = 0; count < shou; ++count, i += 4) {
                retMat.de_ele[i + retMat.nCol * j] = 0.0;
                retMat.de_ele[i + 1 + retMat.nCol * j] = 0.0;
                retMat.de_ele[i + 2 + retMat.nCol * j] = 0.0;
                retMat.de_ele[i + 3 + retMat.nCol * j] = 0.0;
            }
        }
#endif
        break;
    case DenseMatrix::COMPLETION:
        rError("DenseMatrix:: no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

// nakata 2004/12/01
// modified 2008/05/20    "aMat.sp_ele[indexA1] = 0.0;"
// aMat = L L^T
//
// 2026-08-05 ("C7"): report failure instead of silently patching a non-positive pivot.
//
// Upstream set a negative pivot to 0.0, kept going, and returned `true`
// unconditionally, so a Schur complement that is not positive definite was reported
// to the caller as a successful factorisation. The zeroed pivot then propagates: the
// scaling loop below multiplies that whole column by 0.0, so the returned "L" is not
// a factor of anything and the search direction computed from it is silently wrong.
// A pivot of exactly 0.0 was worse still -- it fell into the `else` arm and computed
// 1.0/sqrt(0.0), which poisons the rest of the factorisation.
// Both cases are now a reported FAILURE, which is what the DENSE twin
// choleskyFactorWithAdjust already does (`info > 0` -> rMessage + FAILURE, below),
// and what the CALLER already expects: Newton::compute_DyVec does
//     bool ret = Lal::getCholesky(sparse_bMat, diagonalIndex);
//     if (ret == FAILURE) { return FAILURE; }
// The plumbing was there; only this callee was incapable of ever using it.
//
// Numerically inert on any input whose Schur complement stays positive definite:
// neither branch is reachable unless a pivot is <= 0, and this function's arithmetic
// is otherwise untouched. Proven bit-identical with patches/regress.sh.
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-13: fixes B and C of
   review/SPARSE-CHOLESKY-THREADING-PLAN-V2.md.

   B removes the per-matched-update GMP temporary: `aMat.sp_ele[k3] -= tmp * tmp2` built a
   __gmp_expr temporary, i.e. an mpf_init2 + malloc + free around one multiply and one
   subtract. It is replaced by one reusable scratch per thread plus mpf_srcptr aliases for
   the operands. Bit-identical ONLY under the precision invariant asserted below.

   C threads the k1 loop. It is race-free without atomics because for a fixed pivot i, each
   k1 owns one target row j = column_index[k1], row indices within a segment are unique, so
   distinct k1 write disjoint index ranges; reads come from pivot row i, which this loop
   never writes. Each destination is touched by exactly one k1 per pivot and the k2 order
   inside it is unchanged, and a barrier per pivot preserves the i order -- so every
   destination sees the identical sequence of subtractions in identical order and the factor
   is bit-identical at ANY thread count. That is the property the harness tests.

   Default RESULTS are unchanged; the default EXECUTION PATH is not. With SDPA_SPCHOL_MODE
   unset the gates below decide, and a factorisation that clears them now runs threaded where
   it previously ran serial -- producing a bit-identical factor, but different CPU and wall
   time. Short factorisations stay on the serial routine. Saying "default behaviour is
   unchanged" here would be convenient and wrong. */

namespace {

// ---------------------------------------------------------------- tuning + mode switch
// UNCALIBRATED defaults, deliberately conservative. The fork's own measured constants say a
// fork/join is ~0.6-2.1 us and an mpf multiply-add is 74-76 ns at 256 bits ON THANOS (33-35
// on pi) -- so break-even is of order 10^2 updates there, and these sit well above it. They
// are a floor against pathology, not a tuned optimum: calibrate on the target machine at the
// target precision before claiming otherwise (see the plan, "gates are calibrated, not
// reasoned about").
#ifndef SDPA_OMP_MIN_SPCHOL_WORK
#define SDPA_OMP_MIN_SPCHOL_WORK 20000ULL /* matched updates in one pivot */
#endif
#ifndef SDPA_OMP_MIN_SPCHOL_WIDTH
#define SDPA_OMP_MIN_SPCHOL_WIDTH 8 /* target rows in one pivot */
#endif
#ifndef SDPA_OMP_MIN_SPCHOL_TOTAL
#define SDPA_OMP_MIN_SPCHOL_TOTAL 200000ULL /* matched updates over the whole factor */
#endif

// The three gate constants are compile-time defaults, overridable at runtime. Two reasons,
// and the second is the one that forced this:
//   1. CALIBRATION. The plan is explicit that a gate is calibrated, not reasoned about, and
//      the defaults below are conservative floors rather than tuned optima. Sweeping them
//      should not require a rebuild.
//   2. TESTABILITY. The gates were previously reachable in tests only by SDPA_SPCHOL_MODE=
//      parallel, which forces past BOTH the whole-factor gate and the per-pivot predicate --
//      so the production `auto` dispatch was never exercised at all. truss6 is the only
//      SDPLIB problem that selects SPARSE, its 79022 updates are below the total gate, and
//      its widest pivot yields work of 1326 against a 20000 per-pivot gate, so no corpus
//      fixture can reach the admitted path with the defaults. Lowering the gates lets CI run
//      the real auto path, including a MIXED factorisation where some pivots are workshared
//      and some fall to `single` -- which is the case most likely to expose a divergence
//      between the two branches.
// Strictly parsed: a gate that silently reverts to its default on a typo would quietly
// measure or test the wrong configuration.
uint64_t spchol_gate(const char *name, uint64_t dflt) {
    const char *e = getenv(name);
    if (e == NULL) {
        return dflt;
    }
    if (e[0] == '\0') {
        rError(name << " is set but empty; unset it to use the default");
    }
    // strtoull ACCEPTS a leading minus and wraps it: strtoull("-5") returns ULLONG_MAX-4
    // with errno unset, so a negative would silently become an astronomically large gate --
    // i.e. "never parallel" -- while looking like it was accepted. Reject the sign before
    // parsing. (Caught by this file's own CI control, which found -5 accepted.)
    const char *p = e;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p == '-' || *p == '+') {
        rError(name << " must be a non-negative integer without a sign (got \"" << e << "\")");
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(p, &end, 10);
    if (end == p || *end != '\0' || errno == ERANGE) {
        rError(name << " must be a non-negative integer (got \"" << e << "\")");
    }
    return (uint64_t)v;
}

enum SpcholMode {
    SPCHOL_AUTO,     // gates decide (default)
    SPCHOL_SERIAL,   // fix B, one thread, no team ever created
    SPCHOL_PARALLEL, // fix B + C, gates bypassed (test-only force-parallel route)
    SPCHOL_LEGACY    // the pre-B expression, byte for byte -- the independent oracle
};

SpcholMode spchol_mode() {
    const char *e = getenv("SDPA_SPCHOL_MODE");
    if (e == NULL || e[0] == '\0' || strcmp(e, "auto") == 0)
        return SPCHOL_AUTO;
    if (strcmp(e, "serial") == 0)
        return SPCHOL_SERIAL;
    if (strcmp(e, "parallel") == 0)
        return SPCHOL_PARALLEL;
    if (strcmp(e, "legacy") == 0)
        return SPCHOL_LEGACY;
    // Reject rather than fall back: silently ignoring a typo would measure the wrong
    // configuration and look like a result. Same contract as SDPA_BMAT_MODE.
    rError("SDPA_SPCHOL_MODE must be auto, serial, parallel or legacy (got \"" << e << "\")");
    return SPCHOL_AUTO;
}

bool spchol_log() {
    const char *e = getenv("SDPA_SPCHOL_LOG");
    return e != NULL && e[0] != '\0' && strcmp(e, "0") != 0;
}

// TEST-ONLY fault injection: force the non-positive-pivot branch at a chosen pivot, so the
// synchronised-termination path can be exercised deterministically. A failure at pivot 0
// exercises none of it -- what matters is failing AFTER genuinely parallel pivots have run.
// Previously this lived as an uncommitted scratch edit, which meant the one test covering
// deadlock-freedom could not be re-run by anyone else. Committing it fixed that and created a
// different problem: a knob that aborts a factorisation at a chosen pivot then existed in every
// production solver. It is now COMPILE-GATED -- absent unless -DSDPA_SPCHOL_TEST_HOOKS -- and a
// build without it REFUSES the variable rather than ignoring it, because a user who sets a knob
// that silently does nothing has been misled. CI builds one leg with hooks to run the fault
// test and asserts the refusal in the leg without them.
// Returns -1 when unset, i.e. never inject.
int spchol_fail_at() {
    const char *e = getenv("SDPA_SPCHOL_FAIL_AT");
    if (e == NULL || e[0] == '\0') {
        return -1;
    }
#ifndef SDPA_SPCHOL_TEST_HOOKS
    rError("SDPA_SPCHOL_FAIL_AT is a test hook and this binary was not built with"
           " -DSDPA_SPCHOL_TEST_HOOKS, so the hook does not exist here");
    return -1;
#else
    // errno as well as the range test: where long is 32 bits, strtol saturates an overflowing
    // input to LONG_MAX == INT_MAX, which the range test alone would accept as a valid index.
    errno = 0;
    char *end = NULL;
    const long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || v < 0 || v > INT_MAX) {
        rError("SDPA_SPCHOL_FAIL_AT must be a non-negative pivot index (got \"" << e << "\")");
    }
    return (int)v;
#endif
}

// Every tunable, parsed ONCE at the public entry point, before any mode-specific early return.
// The previous arrangement parsed _WORK and _WIDTH inside spchol_parallel() and _TOTAL behind a
// short-circuit, so a malformed value was silently ignored on the serial, legacy, nested,
// below-gate and no-OpenMP paths -- the identical "a validator on a conditional path is not a
// validator" defect that had just been fixed for SDPA_SPCHOL_FAIL_AT, reintroduced in the same
// commit that recorded the lesson. One struct, one parse site, no path that skips it.
struct SpcholCfg {
    uint64_t gate_work;
    uint64_t gate_width;
    uint64_t gate_total;
    int fail_at;
    int team_override; // 0 = none; test hook, see below
    int mutate;        // test hook: 0 none, 1 perturb a value, 2 perturb a precision
};

// TEST-ONLY, same compile gate as the fault hook. The one-thread branch inside the region
// fires whenever the team actually has one member -- which happens if the runtime hands back
// fewer threads than requested, something OMP_DYNAMIC permits but no setting compels. This
// forces the REQUEST down after the gates have admitted the factorisation, so the region asks
// for one and receives one. That covers the branch; it is not literally a request-many,
// receive-one contraction, and calling it a contraction test would overstate it.
int spchol_team_override() {
    const char *e = getenv("SDPA_SPCHOL_TEAM_OVERRIDE");
    if (e == NULL || e[0] == '\0') {
        return 0;
    }
#ifndef SDPA_SPCHOL_TEST_HOOKS
    rError("SDPA_SPCHOL_TEAM_OVERRIDE is a test hook and this binary was not built with"
           " -DSDPA_SPCHOL_TEST_HOOKS, so the hook does not exist here");
    return 0;
#else
    errno = 0;
    char *end = NULL;
    const long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || v < 1 || v > INT_MAX) {
        rError("SDPA_SPCHOL_TEAM_OVERRIDE must be a positive thread count (got \"" << e << "\")");
    }
    return (int)v;
#endif
}

// TEST-ONLY, same compile gate: perturb the middle diagonal of the FINISHED factor by about
// one unit in the last place. This is the negative control for the digest below -- without it,
// "every arm produced the same digest" is equally consistent with a digest that cannot tell
// anything apart. The middle diagonal is chosen because a successful factorisation guarantees
// it is nonzero, so the perturbation cannot be a no-op.
int spchol_mutate() {
    const char *e = getenv("SDPA_SPCHOL_MUTATE");
    if (e == NULL || e[0] == '\0' || strcmp(e, "0") == 0) {
        return 0;
    }
#ifndef SDPA_SPCHOL_TEST_HOOKS
    rError("SDPA_SPCHOL_MUTATE is a test hook and this binary was not built with"
           " -DSDPA_SPCHOL_TEST_HOOKS, so the hook does not exist here");
    return 0;
#else
    // 1 perturbs a VALUE -- the digest's negative control.
    // 2 perturbs a PRECISION -- the negative control for the uniform-precision invariant,
    //   which was implemented over every element but had no fixture proving it can reject.
    //   The change lands after this factorisation, so the NEXT one must refuse to start.
    if (strcmp(e, "1") == 0)
        return 1;
    if (strcmp(e, "2") == 0)
        return 2;
    rError("SDPA_SPCHOL_MUTATE must be 0, 1 or 2 (got \"" << e << "\")");
    return 0;
#endif
}

SpcholCfg spchol_cfg() {
    SpcholCfg c;
    c.gate_work = spchol_gate("SDPA_OMP_MIN_SPCHOL_WORK", SDPA_OMP_MIN_SPCHOL_WORK);
    c.gate_width = spchol_gate("SDPA_OMP_MIN_SPCHOL_WIDTH", SDPA_OMP_MIN_SPCHOL_WIDTH);
    c.gate_total = spchol_gate("SDPA_OMP_MIN_SPCHOL_TOTAL", SDPA_OMP_MIN_SPCHOL_TOTAL);
    c.fail_at = spchol_fail_at();
    c.team_override = spchol_team_override();
    c.mutate = spchol_mutate();
    return c;
}

// ------------------------------------------------------------------ the factor-level oracle
//
// Everything else in this file compares the PRINTED SOLUTION: objective, phase, iteration
// count, xVec. That is a real check, but it is not the claim being made. Two factorisations
// can differ deep in the mantissa and still print the same 17 digits, so "identical selected
// output" is consistent with a factor that changed -- and the whole argument for fixes B and C
// is that the factor does NOT change.
//
// This builds a CANONICAL, LENGTH-FRAMED byte stream over everything that defines the factor:
// matrix type and dimensions, the sparse counts, the row extents, and for every live element
// its row index, column index, precision, sign, exponent and full mantissa. Base 16 with 0
// digits requested makes mpf_get_str emit the exact value -- no rounding, no locale, no
// platform-dependent decimal formatting.
//
// Framing matters and its absence was a real weakness in the first version: without an explicit
// length before each variable-length field, the concatenations ("ab","c") and ("a","bc") are
// the same stream, so two different factors could in principle agree. Every field is now
// preceded by its length, and every record by a tag.
//
// Two outputs, because they answer different questions:
//   SDPA_SPCHOL_DIGEST=1        a 64-bit fingerprint plus the record and byte counts. Cheap,
//                               and enough to detect a change -- but a fingerprint, not a
//                               proof of identity, and it is described that way.
//   SDPA_SPCHOL_DIGEST_DUMP=f   the canonical stream itself, appended to file f, so two runs
//                               can be compared EXACTLY with cmp(1) rather than by hash
//                               equality. This is the proof-grade comparison.
//
// Off unless asked for: O(nnz) string conversions is fine for a fixture and not something to
// pay for in a solve.
// The stream type and writer now live in namespace sdpa (sdpa_linear.h) so that the bMat
// ASSEMBLY can emit the same bytes from the same code. See the header for why the assembly
// needs its own stream rather than inferring identity from the factor's.
typedef CanonicalStream SpcholDigest;

void spchol_dg_byte(SpcholDigest &d, unsigned char c) {
    // Correct 64-bit FNV-1a constants. The first version seeded with 1469598103934665603,
    // which is the published offset basis with a digit dropped -- a fine hash seed, but not
    // FNV-1a, while the comment said it was.
    d.fnv ^= (uint64_t)c;
    d.fnv *= 1099511628211ULL;
    d.bytes++;
    if (d.dump != NULL && !d.io_error) {
        // Checked, because the counter records bytes ATTEMPTED. A disk-full or quota error
        // would otherwise leave a truncated dump while the log reported a full-length stream,
        // and a comparison against a truncated file is not a comparison.
        if (fputc((int)c, d.dump) == EOF)
            d.io_error = true;
    }
}

void spchol_dg_u64(SpcholDigest &d, uint64_t v) {
    // Unsigned throughout: the previous version shifted a signed long long right, which is
    // implementation-defined for negative values.
    for (int b = 0; b < 8; ++b)
        spchol_dg_byte(d, (unsigned char)((v >> (8 * b)) & 0xffU));
}

void spchol_dg_i64(SpcholDigest &d, long long v) {
    spchol_dg_u64(d, (uint64_t)v); // two's complement, well defined as a conversion
}

void spchol_dg_bytes(SpcholDigest &d, const char *p, size_t n) {
    spchol_dg_u64(d, (uint64_t)n); // the length frame
    for (size_t i = 0; i < n; ++i)
        spchol_dg_byte(d, (unsigned char)p[i]);
}

SpcholDigest spchol_digest_impl(SparseMatrix &aMat, int *diagonalIndex, int nDIM,
                                const char *tag, FILE *dump) {
    SpcholDigest d;
    d.fnv = 14695981039346656037ULL; // the actual FNV-1a offset basis
    d.records = 0;
    d.bytes = 0;
    d.dump = dump;
    d.io_error = false;

    // Header record: structure first. A factor with the same values in a different sparsity
    // pattern is a different factor, and a digest over values alone would call them equal.
    spchol_dg_bytes(d, tag, strlen(tag));
    spchol_dg_i64(d, (long long)aMat.type);
    spchol_dg_i64(d, aMat.nRow);
    spchol_dg_i64(d, aMat.nCol);
    spchol_dg_i64(d, aMat.NonZeroNumber);
    spchol_dg_i64(d, aMat.NonZeroCount);
    spchol_dg_i64(d, aMat.NonZeroEffect);
    spchol_dg_i64(d, nDIM);
    spchol_dg_u64(d, (uint64_t)(nDIM + 1));
    for (int i = 0; i <= nDIM; ++i)
        spchol_dg_i64(d, diagonalIndex[i]);

    void (*freefunc)(void *, size_t) = NULL;
    mp_get_memory_functions(NULL, NULL, &freefunc);
    spchol_dg_u64(d, (uint64_t)aMat.NonZeroCount);
    for (int k = 0; k < aMat.NonZeroCount; ++k) {
        spchol_dg_byte(d, 'E'); // record tag
        spchol_dg_i64(d, k);
        spchol_dg_i64(d, aMat.row_index[k]);
        spchol_dg_i64(d, aMat.column_index[k]);
        mpf_srcptr x = aMat.sp_ele[k].get_mpf_t();
        spchol_dg_u64(d, (uint64_t)mpf_get_prec(x));
        spchol_dg_i64(d, mpf_sgn(x));
        mp_exp_t ex = 0;
        char *s = mpf_get_str(NULL, &ex, 16, 0, x);
        spchol_dg_i64(d, (long long)ex);
        spchol_dg_bytes(d, (s != NULL) ? s : "", (s != NULL) ? strlen(s) : 0);
        if (s != NULL && freefunc != NULL)
            freefunc(s, strlen(s) + 1);
        d.records++;
    }
    spchol_dg_byte(d, '.'); // terminator, so a truncated stream cannot look complete
    return d;
}


SpcholDigest spchol_digest(SparseMatrix &aMat, int *diagonalIndex, int nDIM, FILE *dump) {
    // The factor's tag. A different producer uses a different tag, so an assembly stream and
    // a factor stream can never compare equal by accident.
    return spchol_digest_impl(aMat, diagonalIndex, nDIM, "SPCHOLv2", dump);
}

bool spchol_want_digest() {
    const char *e = getenv("SDPA_SPCHOL_DIGEST");
    return e != NULL && e[0] != '\0' && strcmp(e, "0") != 0;
}

// Exact-comparison sink. Appended to, so one file holds every factorisation of a solve in
// order and two runs are compared with cmp(1) -- byte identity of the whole stream, not
// equality of a hash.
const char *spchol_dump_path() {
    const char *e = getenv("SDPA_SPCHOL_DIGEST_DUMP");
    if (e == NULL || e[0] == '\0')
        return NULL;
    return e;
}


// Fix B is exact only if EVERY live factor element shares one precision -- checking the first
// element and trusting the rest is not the invariant, it is a spot check. O(nnz) reads of an
// integer field: ~3-5 ms for the 6.7M-element factor that takes seconds to compute.
void spchol_assert_uniform_precision(SparseMatrix &aMat, mp_bitcnt_t factor_prec) {
    for (int k = 0; k < aMat.NonZeroCount; ++k) {
        if (mpf_get_prec(aMat.sp_ele[k].get_mpf_t()) != factor_prec) {
            rError("Lal::getCholesky: sp_ele[" << k << "] has precision "
                   << (unsigned long)mpf_get_prec(aMat.sp_ele[k].get_mpf_t())
                   << " but the factor's is " << (unsigned long)factor_prec
                   << "; the scratch-based update is bit-identical only under one precision");
        }
    }
}

// Counters. These exist so an identity test cannot pass vacuously: identical factors across
// thread counts prove nothing if the parallel branch never opened or if one worker did all
// the work. per_thread_matched is what distinguishes "a team of 8 existed" from "8 threads
// performed updates".
struct SpcholStats {
    uint64_t attempted;      // inner-search comparisons (the scan the ceiling model omits)
    uint64_t matched;        // structural updates actually performed
    uint64_t work_admitted;  // matched updates inside parallel pivots
    uint64_t pivots_total;
    uint64_t pivots_parallel;
    int team_requested;
    int team_actual;
    int max_width_admitted;
    int workers_used;        // threads with a nonzero matched count
    const char *path;        // "serial" | "parallel" | "legacy"
    const char *fallback;    // why serial, when it was not asked for
};

// Reported only under SDPA_SPCHOL_LOG. `workers` is the load-bearing number: a team of N
// existing is not evidence that N threads did work, and an identity test that cannot tell
// those apart is vacuous.
void spchol_report(const SpcholStats &st, int nDIM, const SpcholCfg &cfg) {
    // The thresholds in force, not the compiled-in defaults: a log that does not say which
    // policy produced it cannot be used to reproduce the run it describes.
    const char *req = getenv("SDPA_SPCHOL_MODE");
    const char *reqs = (req && req[0]) ? req : "auto";
    // "applied" is only true for auto. Forced parallel bypasses the gates; forced serial and
    // legacy never reach them at all, and reporting "applied" for those was simply wrong.
    const char *gates = (strcmp(reqs, "auto") == 0)       ? "applied"
                        : (strcmp(reqs, "parallel") == 0) ? "BYPASSED (forced parallel)"
                                                          : "not consulted (forced)";
    printf("spchol request   : mode=%s, gates %s\n", reqs, gates);
    printf("spchol gates     : total>=%llu, per-pivot work>=%llu and width>=%llu\n",
           (unsigned long long)cfg.gate_total, (unsigned long long)cfg.gate_work,
           (unsigned long long)cfg.gate_width);
    printf("spchol path      : %s", st.path);
    if (st.fallback[0])
        printf("   (serial because: %s)", st.fallback);
    printf("\n");
    printf("spchol team      : requested %d, actual %d, workers that updated %d\n",
           st.team_requested, st.team_actual, st.workers_used);
    printf("spchol pivots    : %llu total, %llu parallel, max admitted width %d\n",
           (unsigned long long)st.pivots_total, (unsigned long long)st.pivots_parallel,
           st.max_width_admitted);
    printf("spchol updates   : %llu matched, %llu attempted (search comparisons), "
           "%llu matched inside parallel pivots\n",
           (unsigned long long)st.matched, (unsigned long long)st.attempted,
           (unsigned long long)st.work_admitted);
    if (st.matched > 0)
        printf("spchol scan/match: %.3f comparisons per matched update  (m=%d)\n",
               (double)st.attempted / (double)st.matched, nDIM);
}

// One place where a completed factorisation is mutated (if asked), digested (if asked) and
// reported (if asked) -- so the three exit paths of getCholesky cannot drift apart.
bool spchol_finish(SparseMatrix &aMat, int *diagonalIndex, int nDIM, const SpcholCfg &cfg,
                   SpcholStats &st, bool ok) {
    if (ok && cfg.mutate == 1 && nDIM > 0) {
        mpf_ptr e = aMat.sp_ele[diagonalIndex[nDIM / 2]].get_mpf_t();
        mpf_t d;
        mpf_init2(d, mpf_get_prec(e));
        mpf_set(d, e);
        mpf_div_2exp(d, d, mpf_get_prec(e) - 2); // ~1 ulp, and never exactly zero
        mpf_add(e, e, d);
        mpf_clear(d);
    } else if (ok && cfg.mutate == 2 && nDIM > 0) {
        mpf_ptr e = aMat.sp_ele[diagonalIndex[nDIM / 2]].get_mpf_t();
        mpf_set_prec(e, mpf_get_prec(e) + 64); // the next factorisation must refuse to start
    }
    const char *dumpf = spchol_dump_path();
    if (ok && (spchol_want_digest() || dumpf != NULL)) {
        FILE *dump = (dumpf != NULL) ? fopen(dumpf, "ab") : NULL;
        if (dumpf != NULL && dump == NULL) {
            rError("SDPA_SPCHOL_DIGEST_DUMP: cannot open \"" << dumpf << "\" for append");
        }
        SpcholDigest d = spchol_digest(aMat, diagonalIndex, nDIM, dump);
        if (dump != NULL) {
            if (fflush(dump) != 0 || ferror(dump) != 0)
                d.io_error = true;
            if (fclose(dump) != 0)
                d.io_error = true;
        }
        if (d.io_error) {
            rError("SDPA_SPCHOL_DIGEST_DUMP: writing \"" << dumpf << "\" failed after "
                   << (unsigned long long)d.bytes << " bytes. The dump is truncated, so any"
                   << " comparison against it would be meaningless; failing rather than"
                   << " leaving a file that looks complete.");
        }
        // The counts are printed alongside the fingerprint deliberately: equality of a 64-bit
        // hash is strong evidence, not proof, and two streams of different length or record
        // count are not the same factor whatever their hashes do.
        if (spchol_want_digest())
            printf("spchol factor    : %d rows, %llu records, %llu stream bytes,"
                   " fingerprint %016llx\n",
                   nDIM, (unsigned long long)d.records, (unsigned long long)d.bytes,
                   (unsigned long long)d.fnv);
    }
    if (spchol_log())
        spchol_report(st, nDIM, cfg);
    return ok;
}

// ---------------------------------------------------------------- the one update kernel
// Shared by the serial routine, the threaded routine and both of its branches, so the four
// paths cannot drift apart. The oracle (SPCHOL_LEGACY) deliberately does NOT call it.
inline void spchol_axmy(mpf_ptr dst, mpf_srcptr a, mpf_srcptr b, mpf_ptr scratch) {
    mpf_mul(scratch, a, b);
    mpf_sub(dst, dst, scratch);
}

// One k1 task: the updates pivot row i contributes to target row column_index[k1].
// Every mutable per-iteration variable is declared here, inside the task, so the OpenMP
// `private(...)` clause is never needed -- and a raw mpf_t must never be privatised anyway,
// because the copy would not be GMP-initialised.
inline void spchol_k1(SparseMatrix &aMat, int *diagonalIndex, int k1, int indexA2,
                      mpf_ptr scratch, uint64_t *attempted, uint64_t *matched) {
    mpf_srcptr a = aMat.sp_ele[k1].get_mpf_t(); // alias, not a copy: row i is read-only here
    int k3 = diagonalIndex[aMat.column_index[k1]];
    const int indexB2 = diagonalIndex[aMat.column_index[k1] + 1];
    uint64_t att = 0, mat = 0;
    for (int k2 = k1; k2 < indexA2; ++k2) {
        mpf_srcptr b = aMat.sp_ele[k2].get_mpf_t();
        const int tmp3 = aMat.column_index[k2];
        // k3 is a MONOTONE cursor across k2 -- never reset. That is what keeps the search
        // amortised, and it is correct only because column indices ascend within a segment.
        for (; k3 < indexB2; ++k3) {
            ++att;
            if (aMat.column_index[k3] == tmp3) {
                spchol_axmy(aMat.sp_ele[k3].get_mpf_t(), a, b, scratch);
                ++mat;
                k3++;
                break;
            }
        }
    }
    *attempted += att;
    *matched += mat;
}

// Pivot bookkeeping every thread computes identically, so no shared array is needed and the
// per-pivot branch cannot diverge across the team.
inline int spchol_width(int a1, int a2) { return a2 - a1 - 1; }
inline uint64_t spchol_work(int a1, int a2) {
    const uint64_t w = (uint64_t)spchol_width(a1, a2);
    return w * (w + 1ULL) / 2ULL; // matched updates under suffix containment
}

// ---------------------------------------------------------------- serial routine (fix B)
bool spchol_serial(SparseMatrix &aMat, int *diagonalIndex, mp_bitcnt_t factor_prec,
                   SpcholStats &st) {
    const int nDIM = aMat.nRow;
    mpf_class scratch(0, factor_prec);
    for (int i = 0; i < nDIM; ++i) {
        const int a1 = diagonalIndex[i], a2 = diagonalIndex[i + 1];
        if (!(aMat.sp_ele[a1] > 0.0)) {
            rMessage("sparse cholesky miss condition :: not positive definite"
                     << " :: pivot " << i << " = " << aMat.sp_ele[a1].get_d());
            return FAILURE;
        }
        aMat.sp_ele[a1] = 1.0 / sqrt(aMat.sp_ele[a1]);
        for (int k1 = a1 + 1; k1 < a2; ++k1)
            aMat.sp_ele[k1] *= aMat.sp_ele[a1];
        for (int k1 = a1 + 1; k1 < a2; ++k1)
            spchol_k1(aMat, diagonalIndex, k1, a2, scratch.get_mpf_t(), &st.attempted,
                      &st.matched);
        st.pivots_total++;
    }
    return _SUCCESS;
}

// ---------------------------------------------------------------- oracle (pre-B, frozen)
// The exact upstream expression, kept so that fix B has an INDEPENDENT check. If B and the
// serial routine are compared only against each other, an identically wrong shared kernel
// agrees with itself; this arm does not call spchol_axmy at all.
bool spchol_legacy(SparseMatrix &aMat, int *diagonalIndex, SpcholStats &st) {
    const int nDIM = aMat.nRow;
    mpf_class tmp, tmp2;
    for (int i = 0; i < nDIM; ++i) {
        const int a1 = diagonalIndex[i], a2 = diagonalIndex[i + 1];
        if (!(aMat.sp_ele[a1] > 0.0)) {
            rMessage("sparse cholesky miss condition :: not positive definite"
                     << " :: pivot " << i << " = " << aMat.sp_ele[a1].get_d());
            return FAILURE;
        }
        aMat.sp_ele[a1] = 1.0 / sqrt(aMat.sp_ele[a1]);
        for (int k1 = a1 + 1; k1 < a2; ++k1)
            aMat.sp_ele[k1] *= aMat.sp_ele[a1];
        for (int k1 = a1 + 1; k1 < a2; ++k1) {
            tmp = aMat.sp_ele[k1];
            int k3 = diagonalIndex[aMat.column_index[k1]];
            const int indexB2 = diagonalIndex[aMat.column_index[k1] + 1];
            for (int k2 = k1; k2 < a2; ++k2) {
                tmp2 = aMat.sp_ele[k2];
                const int tmp3 = aMat.column_index[k2];
                for (; k3 < indexB2; ++k3) {
                    ++st.attempted;
                    if (aMat.column_index[k3] == tmp3) {
                        aMat.sp_ele[k3] -= tmp * tmp2;
                        ++st.matched;
                        k3++;
                        break;
                    }
                }
            }
        }
        st.pivots_total++;
    }
    return _SUCCESS;
}

#ifdef _OPENMP
// ---------------------------------------------------------------- threaded routine (C)
bool spchol_parallel(SparseMatrix &aMat, int *diagonalIndex, mp_bitcnt_t factor_prec,
                     int team, bool force, const SpcholCfg &cfg, SpcholStats &st) {
    // These four are const AND named in `firstprivate` rather than `shared`, which is the one
    // spelling that is both portable and honest about what they are.
    //
    // The portability half. The OpenMP revisions disagree about a const variable named in
    // `shared` under `default(none)`: through OpenMP 4.5 a const with no mutable member is
    // PREDETERMINED shared and naming it is an error, while OpenMP 5.0 deleted that rule so it
    // MUST be named. GCC 8 implements 4.5, GCC 9+ implement 5.0, and the released tip failed to
    // compile on Expanse's system GCC 8.5 with four of exactly this error. Five spellings were
    // COMPILED on four compilers rather than reasoned about, after the first answer here was
    // reasoned and turned out wrong:
    //
    //   const + shared()                 GCC 8 ERROR, others OK    <- what shipped
    //   const + not named                clang ERROR, others OK    <- satisfies BOTH GCCs, trap
    //   non-const + shared()             OK everywhere             <- the first fix
    //   const + firstprivate()           OK everywhere             <- this
    //   const + shared(), no default(none)  GCC 8 ERROR            <- removing the clause does
    //                                                                 not remove the conflict
    //
    // Two spellings work everywhere and this is the better of them, because the first one buys
    // portability by deleting an invariant: it drops const purely to satisfy a clause.
    //
    // The honesty half. All four are read-only scalars. `firstprivate` says each thread gets its
    // own copy of a value it only reads, which is what is happening; `shared` says the team
    // communicates through them, which is not. It also sidesteps the predetermined-shared
    // question altogether instead of working around it, so it stays correct if the rule moves
    // again in either direction.
    //
    // PRECONDITION: the region must never WRITE these. Verified -- the body contains no
    // assignment, increment or compound assignment to any of the four. If that ever changes,
    // `firstprivate` would quietly make the write thread-local instead of a race: safer, but
    // different, and not what the writer would have meant. `default(none)`, which is what
    // actually enforces the sharing contract, is kept.
    const int nDIM = aMat.nRow;
    const int fail_at = cfg.fail_at; // -1 unless the compiled-in test hook is set
    const uint64_t gate_work = cfg.gate_work;
    const uint64_t gate_width = cfg.gate_width;
    // Set by the sole thread when the team has one member; see the top of the region.
    bool one_thread = false;
    bool one_thread_r = false;
    bool fail = false;
    int failed_pivot = -1;
    double failed_value = 0.0;
    // Per-thread counters, summed after the region: no atomics on the hot path, and
    // per-thread matched counts are what prove more than one worker did real work.
    uint64_t *t_att = new uint64_t[team];
    uint64_t *t_mat = new uint64_t[team];
    uint64_t *t_adm = new uint64_t[team];
    for (int t = 0; t < team; ++t) {
        t_att[t] = t_mat[t] = t_adm[t] = 0;
    }
    int actual = 1;
    uint64_t pivots_par = 0;
    int max_width = 0;

#pragma omp parallel num_threads(team) default(none)                                        \
    shared(aMat, diagonalIndex, fail, failed_pivot, failed_value, actual, t_att,            \
           t_mat, t_adm, pivots_par, max_width, factor_prec, force, team,                   \
           one_thread, one_thread_r, st)                                                    \
    firstprivate(nDIM, fail_at, gate_work, gate_width)
    {
        // THE team, read inside the region that owns it -- not inferred from a probe.
        // num_threads() is a REQUEST: OMP_DYNAMIC or a runtime resource limit can return
        // fewer threads, including one, and OpenMP does NOT promise that two consecutive
        // regions receive the same team. The previous design probed with a throwaway region
        // and dispatched on its answer, which is precisely that unwarranted inference -- the
        // probe could be given two threads and this region one, and then every pivot would
        // run through `single`, which is the behaviour dispatching to the serial routine
        // was meant to eliminate. Reading it here cannot be wrong about it.
        //
        // The branch is COLLECTIVE: every thread evaluates the same team size, so no thread
        // takes a different path and the worksharing constructs below are still encountered
        // by all of them in the same order. With one thread, that thread runs the untouched
        // serial routine rather than an emulation of it inside a parallel region.
        const int here = omp_get_num_threads();
        if (here < 2) {
            one_thread = true;
            one_thread_r = spchol_serial(aMat, diagonalIndex, factor_prec, st);
        } else {
        const int tid = omp_get_thread_num();
#pragma omp single
        { actual = here; }

        // One scratch per thread for the whole factorisation, at the destination precision.
        mpf_class scratch(0, factor_prec);

        for (int i = 0; i < nDIM; ++i) {
            const int a1 = diagonalIndex[i], a2 = diagonalIndex[i + 1];
            // Only the pivot test and its inverse square root are inherently one-thread work.
#pragma omp single
            {
                if (i == fail_at) { // test hook; fail_at is -1 in every normal run
                    fail = true;
                    failed_pivot = i;
                    failed_value = -1.0;
                } else if (!(aMat.sp_ele[a1] > 0.0)) {
                    fail = true;
                    failed_pivot = i;
                    failed_value = aMat.sp_ele[a1].get_d();
                } else {
                    aMat.sp_ele[a1] = 1.0 / sqrt(aMat.sp_ele[a1]);
                }
            } // implicit barrier: publishes fail and the inverted diagonal
            if (fail)
                break; // identical in every thread, so the worksharing sequence stays identical

            // Row scaling, shared out rather than left in the `single` above. It is O(L_i) per
            // pivot and therefore O(nnz) over the factorisation -- 6.69M GMP multiplies on the
            // motivating problem -- so leaving it on one thread inside the parallel region puts
            // an Amdahl floor under the whole routine that only bites once the update loop is
            // fast. Elementwise and independent: each k1 writes its own element and reads only
            // the (already inverted, already published) diagonal, so this is bit-identical.
#pragma omp for
            for (int k1 = a1 + 1; k1 < a2; ++k1)
                aMat.sp_ele[k1] *= aMat.sp_ele[a1];
            // implicit barrier: the scaled row must be complete before any update reads it

            // Same predicate in every thread -- derived only from values all of them have.
            const bool par = (actual > 1) &&
                             (force || (spchol_work(a1, a2) >= gate_work &&
                                        (uint64_t)spchol_width(a1, a2) >= gate_width));
            if (par) {
#pragma omp single
                {
                    pivots_par++;
                    if (spchol_width(a1, a2) > max_width)
                        max_width = spchol_width(a1, a2);
                }
#pragma omp for schedule(dynamic)
                for (int k1 = a1 + 1; k1 < a2; ++k1) {
                    const uint64_t before = t_mat[tid];
                    spchol_k1(aMat, diagonalIndex, k1, a2, scratch.get_mpf_t(), &t_att[tid],
                              &t_mat[tid]);
                    t_adm[tid] += t_mat[tid] - before;
                } // implicit barrier: pivot i completes before i+1 begins. NEVER nowait.
            } else {
                // Too small to share out. One thread does it; the end barrier is retained so
                // the team stays in lockstep across pivots.
#pragma omp single
                {
                    for (int k1 = a1 + 1; k1 < a2; ++k1)
                        spchol_k1(aMat, diagonalIndex, k1, a2, scratch.get_mpf_t(),
                                  &t_att[tid], &t_mat[tid]);
                }
            }
        }
        } // else: the team really had more than one thread
    }

    // A one-member team already ran the whole factorisation through spchol_serial(), on its
    // sole thread, inside the region above. Report it as what it was: the serial path.
    if (one_thread) {
        delete[] t_att;
        delete[] t_mat;
        delete[] t_adm;
        st.path = "serial";
        st.fallback = "team has one thread; ran the serial routine";
        st.team_requested = team;
        st.team_actual = 1;
        return one_thread_r;
    }

    st.team_actual = actual;
    st.pivots_parallel = pivots_par;
    st.max_width_admitted = max_width;
    st.workers_used = 0;
    for (int t = 0; t < team; ++t) {
        st.attempted += t_att[t];
        st.matched += t_mat[t];
        st.work_admitted += t_adm[t];
        if (t_mat[t] > 0)
            st.workers_used++;
    }
    delete[] t_att;
    delete[] t_mat;
    delete[] t_adm;

    // Set BEFORE the failure check: reporting "0 total, 100 parallel" is nonsense, and a
    // counter that is only correct on the happy path is exactly the kind of thing that makes
    // a diagnostic untrustworthy when it matters. On failure, pivots_total is the number
    // that COMPLETED, i.e. the index of the pivot that failed.
    st.pivots_total = fail ? (uint64_t)(failed_pivot < 0 ? 0 : failed_pivot) : (uint64_t)nDIM;

    if (fail) {
        // Full diagnostic, preserved verbatim including the offending pivot's VALUE.
        rMessage("sparse cholesky miss condition :: not positive definite"
                 << " :: pivot " << failed_pivot << " = " << failed_value);
        return FAILURE;
    }
    return _SUCCESS;
}
#endif // _OPENMP

// ---------------------------------------------------------------------------
// Measurement-only profile of the sparse triangular solve.
//
// SOLE PURPOSE is measurement: it reads no solver decision and changes no computed value.
// It exists because review/SOLVE-CENSUS-PLAN.md §11.2 prices two phases of solve threading
// from an ESTIMATED solve-call count and an UNMEASURED barrier cost, and a projection built
// on those may not be reported as a result. ComputeTime::solve already covers the whole of
// Newton::compute_DyVec's solve block -- permuteVec, both passes, reverse_permuteVec -- as a
// single number, which can separate neither forward from backward nor the permutation
// overhead from the passes. This splits it; the permutation overhead is then com.solve minus
// the totals printed here.
//
// Two things it deliberately does NOT do, said here so nobody reads more into the numbers:
//   - The TUNEUP != 0 unrolled path is NOT instrumented. TUNEUP is a compile-time 0 in this
//     file and the record is emitted only from the live path, so a TUNEUP build would report
//     zero calls -- visibly nothing, rather than a plausible wrong number.
//   - It measures today's SERIAL passes. It is the baseline the phases get measured against;
//     it is not evidence about any threaded variant.
// ---------------------------------------------------------------------------
struct SolveProfile {
    long long calls;
    long long dense_calls; // solves that took the DENSE path, which this does not instrument
    long long par_calls;     // calls whose FORWARD pass actually ran on a team of 2 or more
    long long par_bwd_calls; // calls whose BACKWARD pass actually ran on a team of 2 or more
    long long shape_changes; // calls whose (n, nnz) differed from the first call's
    long long n_seen;
    long long nnz_seen;
    double idx_sum; // recovering the row extents and PROVING the pattern, per call
    double copy_sum;
    double fwd_sum, fwd_sumsq, fwd_min, fwd_max;
    double bwd_sum, bwd_sumsq, bwd_min, bwd_max;
};

// Namespace-scope, so zero-initialised before any call. calls == 0 is what marks the
// min/max fields as not yet meaningful; there is no sentinel to get wrong.
SolveProfile g_solve_prof;

double solve_profile_sd(double sum, double sumsq, long long n) {
    if (n < 2) {
        return 0.0;
    }
    const double mean = sum / (double)n;
    const double var = sumsq / (double)n - mean * mean;
    return var > 0.0 ? sqrt(var) : 0.0; // rounding can push a near-zero variance negative
}

// Registered with atexit. Must NOT use rError: rError calls exit(), and calling exit() from
// an atexit handler is undefined behaviour. An I/O failure here is reported to stderr and the
// process is left to finish normally -- the profile is a diagnostic, and losing it must not
// change the exit status of a solve that otherwise succeeded.
void solve_profile_dump() {
    const SolveProfile &p = g_solve_prof;
    FILE *fp = stderr;
    bool close_it = false;
    const char *path = getenv("SDPA_SOLVE_PROFILE_OUT");
    if (path != NULL && path[0] != '\0') {
        fp = fopen(path, "w");
        if (fp == NULL) {
            fprintf(stderr, "SOLVE_PROFILE could not open \"%s\" for writing;"
                            " falling back to stderr\n",
                    path);
            fp = stderr;
        } else {
            close_it = true;
        }
    }

    // Printed even when calls == 0. A profile that was asked for and saw no solve is a
    // result about the run, and silence would read as "the feature is off".
    fprintf(fp,
            "SOLVE_PROFILE calls=%lld par_fwd_calls=%lld par_bwd_calls=%lld dense_calls=%lld"
            " n=%lld nnz=%lld shape_changes=%lld\n",
            p.calls, p.par_calls, p.par_bwd_calls, p.dense_calls,
            p.calls > 0 ? p.n_seen : -1LL, p.calls > 0 ? p.nnz_seen : -1LL, p.shape_changes);
    if (p.calls == 0 && p.dense_calls > 0) {
        fprintf(fp, "SOLVE_PROFILE note: every solve took the DENSE path (Rtrsv), which the"
                    " threaded passes do not cover; nothing here was threadable.\n");
    }
    if (p.calls > 0) {
        const double c = (double)p.calls;
        // Reported separately and never folded into fwd_total: the index build is the price
        // of the parallel path's safety proof, and a speedup that quietly excluded its own
        // setup cost would not be a speedup.
        fprintf(fp, "SOLVE_PROFILE idx_total=%.9f idx_mean=%.9f\n", p.idx_sum, p.idx_sum / c);
        fprintf(fp, "SOLVE_PROFILE copy_total=%.9f copy_mean=%.9f\n", p.copy_sum, p.copy_sum / c);
        fprintf(fp,
                "SOLVE_PROFILE fwd_total=%.9f fwd_mean=%.9f fwd_min=%.9f fwd_max=%.9f fwd_sd=%.9f\n",
                p.fwd_sum, p.fwd_sum / c, p.fwd_min, p.fwd_max,
                solve_profile_sd(p.fwd_sum, p.fwd_sumsq, p.calls));
        fprintf(fp,
                "SOLVE_PROFILE bwd_total=%.9f bwd_mean=%.9f bwd_min=%.9f bwd_max=%.9f bwd_sd=%.9f\n",
                p.bwd_sum, p.bwd_sum / c, p.bwd_min, p.bwd_max,
                solve_profile_sd(p.bwd_sum, p.bwd_sumsq, p.calls));
        fprintf(fp, "SOLVE_PROFILE passes_total=%.9f\n",
                p.idx_sum + p.copy_sum + p.fwd_sum + p.bwd_sum);
    }

    if (fflush(fp) != 0 || ferror(fp) != 0) {
        fprintf(stderr, "SOLVE_PROFILE write failed; the profile above is incomplete\n");
    }
    if (close_it && fclose(fp) != 0) {
        fprintf(stderr, "SOLVE_PROFILE close failed; the profile may be truncated\n");
    }
}

bool solve_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("SDPA_SOLVE_PROFILE");
        cached = (e != NULL && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        if (cached != 0 && atexit(solve_profile_dump) != 0) {
            rError("SDPA_SOLVE_PROFILE: atexit registration failed, so the profile would be"
                   " silently discarded at exit");
        }
    }
    return cached != 0;
}

// Registers the dumper and counts a dense solve. The dense path uses Rtrsv and none of the
// threading below applies to it, but it must still be COUNTED: without this, asking for a
// profile on a problem whose bMat goes dense produced no output whatsoever, because the
// dumper is only registered on first use and the sparse solve was never called. Silence is the
// one thing this was documented not to do -- "a profile that was asked for and saw no solve is
// a result about the run" -- and the documented guarantee was not being met in exactly the case
// it was written for. Now such a run prints calls=0 alongside a non-zero dense_calls, which
// says what actually happened.
void solve_profile_note_dense() {
    if (solve_profile_enabled()) {
        g_solve_prof.dense_calls++;
    }
}

void solve_profile_record(double idx_s, double copy_s, double fwd_s, double bwd_s, bool par,
                          bool par_bwd, long long n, long long nnz) {
    SolveProfile &p = g_solve_prof;
    if (p.calls == 0) {
        p.n_seen = n;
        p.nnz_seen = nnz;
        p.fwd_min = p.fwd_max = fwd_s;
        p.bwd_min = p.bwd_max = bwd_s;
    } else {
        if (n != p.n_seen || nnz != p.nnz_seen) {
            p.shape_changes++;
        }
        if (fwd_s < p.fwd_min) {
            p.fwd_min = fwd_s;
        }
        if (fwd_s > p.fwd_max) {
            p.fwd_max = fwd_s;
        }
        if (bwd_s < p.bwd_min) {
            p.bwd_min = bwd_s;
        }
        if (bwd_s > p.bwd_max) {
            p.bwd_max = bwd_s;
        }
    }
    p.calls++;
    if (par) {
        p.par_calls++;
    }
    if (par_bwd) {
        p.par_bwd_calls++;
    }
    p.idx_sum += idx_s;
    p.copy_sum += copy_s;
    p.fwd_sum += fwd_s;
    p.fwd_sumsq += fwd_s * fwd_s;
    p.bwd_sum += bwd_s;
    p.bwd_sumsq += bwd_s * bwd_s;
}

// ---------------------------------------------------------------------------
// Phase 1 of the parallel solve: the FORWARD triangular substitution, threaded and
// bit-identical to the loop it replaces.
//
// review/SOLVE-CENSUS-PLAN.md §10 concluded that threading the solve costs bit-identity.
// That was right about the backward pass and WRONG about the forward one, and the difference
// is which way the data moves:
//
//   forward   x[col[k]] -= v[k] * x[i]   SCATTER. Within row i the columns strictly ascend,
//                                        so they are distinct, so the row's writes land in
//                                        distinct destinations. Rows still run in order, so
//                                        every x[j] receives its contributions from
//                                        i = 0, 1, 2, ... in exactly the legacy order.
//   backward  x[i] -= v[k] * x[j]        GATHER. Every entry of row i accumulates into the
//                                        SAME x[i]. Splitting that reorders a
//                                        finite-precision sum. That is Phase 2, and it needs
//                                        a licence this does not.
//
// The identity claim rests ENTIRELY on the columns of a row being distinct, so that is not
// assumed anywhere: solve_build_rows() proves strict ascent on every call and refuses the
// parallel path when it does not hold.
//
// Synchronisation, which is what actually decides whether this is worth doing. The obvious
// shape costs TWO barriers per row: one after scaling x[i] so every thread sees it, one after
// the scatter so the next row's scale does not race the current row's writes. The second is
// unavoidable. The first is not: each thread instead recomputes the scaled value into a
// PRIVATE copy -- one redundant multiply per row per thread, off the critical path -- and the
// master writes the shared x[i] back AFTER the scatter's implicit barrier, a point at which
// every read of x[i] has already happened and no later row can touch it, since row i' only
// writes columns j > i' > i. One barrier per row.
//
// That halves the dominant term. On this hardware the barrier is MEASURED, not assumed:
// review/artifacts/solve-census/omp_barrier_bench.cpp reports an `omp for` at 6.09 us on 32
// threads of an EPYC 7532 -- twice the 3 us §11.2 projected with. Against an average dE4 row
// of 6678348/7401 = 902 nonzeros, i.e. ~101 us of serial work, one barrier per row predicts
// ~10.9x on the pass where two would predict ~6.3x.
// ---------------------------------------------------------------------------

enum SolveMode {
    SOLVE_AUTO,    // the gate below decides (default)
    SOLVE_SERIAL,  // the legacy loop, no team ever created
    SOLVE_PARALLEL // forced past the gate; still refused if the pattern does not qualify
};

struct SolveCfg {
    SolveMode mode;     // the FORWARD pass; bit-identical, so auto may enable it
    SolveMode bwd_mode; // the BACKWARD pass; changes the last digits, so it defaults to SERIAL
    uint64_t min_mean_row; // mean off-diagonals per row below which auto stays serial
    uint64_t min_rows;     // row count below which auto stays serial
    uint64_t max_team;     // cap on the solve team; 0 disables the cap
    uint64_t chunks;       // FIXED chunks per row in the backward pass, independent of the team
    uint64_t min_chunk;    // rows shorter than this are computed exactly as the legacy loop
    int team_override;     // test hook, 0 = unset
    bool log;
};

SolveMode solve_mode() {
    const char *e = getenv("SDPA_SOLVE_MODE");
    if (e == NULL) {
        return SOLVE_AUTO;
    }
    if (e[0] == '\0') {
        rError("SDPA_SOLVE_MODE is set but empty; unset it to use the default");
    }
    if (strcmp(e, "auto") == 0) {
        return SOLVE_AUTO;
    }
    if (strcmp(e, "serial") == 0) {
        return SOLVE_SERIAL;
    }
    if (strcmp(e, "parallel") == 0) {
        return SOLVE_PARALLEL;
    }
    // Reject rather than fall back, the same contract as SDPA_SPCHOL_MODE and SDPA_BMAT_MODE:
    // a typo that silently reverts to the default measures the wrong configuration and then
    // looks like a result.
    rError("SDPA_SOLVE_MODE must be auto, serial or parallel (got \"" << e << "\")");
    return SOLVE_AUTO;
}

// The backward pass is a SEPARATE switch rather than a value of SDPA_SOLVE_MODE, because the
// two passes differ in the only way that matters to a user: the forward one returns the same
// bits and the backward one does not. Folding them into one knob would mean a single word
// silently decided whether the answer changed. Default serial, i.e. opt-in.
SolveMode solve_bwd_mode() {
    const char *e = getenv("SDPA_SOLVE_BACKWARD");
    if (e == NULL) {
        return SOLVE_SERIAL;
    }
    if (e[0] == '\0') {
        rError("SDPA_SOLVE_BACKWARD is set but empty; unset it to use the default");
    }
    if (strcmp(e, "serial") == 0) {
        return SOLVE_SERIAL;
    }
    if (strcmp(e, "auto") == 0) {
        return SOLVE_AUTO;
    }
    if (strcmp(e, "parallel") == 0) {
        return SOLVE_PARALLEL;
    }
    rError("SDPA_SOLVE_BACKWARD must be serial, auto or parallel (got \"" << e << "\")");
    return SOLVE_SERIAL;
}

// A gate that must survive a narrowing cast later. spchol_gate accepts any uint64_t, and three
// solve knobs were then cast: MIN_CHUNK to int, MIN_ROW to long long, the test hook to int. A
// value above the destination's range is implementation-defined and can land NEGATIVE, which
// for MIN_CHUNK makes every row chunkable and for MIN_ROW inverts the admission test. Rejecting
// out-of-range input is the fix; silently wrapping it is how a gate ends up meaning its
// opposite.
uint64_t solve_gate_bounded(const char *name, uint64_t dflt, uint64_t lo, uint64_t hi) {
    const uint64_t v = spchol_gate(name, dflt);
    if (v < lo || v > hi) {
        rError(name << " must be between " << lo << " and " << hi << " (got " << v << ")");
    }
    return v;
}

SolveCfg solve_cfg() {
    SolveCfg c;
    c.mode = solve_mode();
    c.bwd_mode = solve_bwd_mode();
    // Gate. Parallel wins when a row's work exceeds the barrier it costs: at 111.5 ns per
    // GMP operation and a 6.09 us barrier, break-even at 32 threads is a mean row of about
    // 56 off-diagonals. The default asks for 256 -- roughly 4.6x the break-even, so a problem
    // has to be clearly worth threading, not marginally. dE4's mean is 902 and dE3's 787;
    // truss6's is 26 and is correctly refused. Both numbers are settable because the
    // break-even moves with precision, and a gate calibrated once at one precision going
    // stale is exactly what happened to the bMat gate 3.
    c.min_mean_row = solve_gate_bounded("SDPA_SOLVE_MIN_ROW", 256, 0, 1000000000ULL);
    c.min_rows = solve_gate_bounded("SDPA_SOLVE_MIN_ROWS", 64, 0, 1000000000ULL);
    // See the collapse note at the team computation. 0 disables the cap entirely, for anyone
    // measuring the cliff on purpose.
    c.max_team = solve_gate_bounded("SDPA_SOLVE_MAX_TEAM", 32, 0, 65536);
    // Chunks per row in the backward pass. FIXED and independent of the team size: this is
    // what makes 1, 8, 32 and 64 threads produce the same bits. It is settable so the
    // accuracy/parallelism trade can be swept, and CHANGING IT CHANGES THE ANSWER -- which is
    // why the log prints it rather than leaving it implicit.
    c.chunks = solve_gate_bounded("SDPA_SOLVE_CHUNKS", 64, 2, 4096);
    // Below this many off-diagonals a row is not chunked at all and is computed exactly as
    // the legacy loop computes it. Default 4x the chunk count, so a chunked row carries at
    // least four terms and the fixed combine cost cannot dominate its own reduction.
    c.min_chunk = solve_gate_bounded("SDPA_SOLVE_MIN_CHUNK", 4 * c.chunks, 0, 1000000000ULL);
    const char *l = getenv("SDPA_SOLVE_LOG");
    c.log = (l != NULL && l[0] != '\0' && strcmp(l, "0") != 0);
    c.team_override = 0;
#ifdef SDPA_SPCHOL_TEST_HOOKS
    // Applied AFTER admission, never before: setting it earlier tests the pre-admission
    // team<2 rejection instead of the in-region branch it exists to reach. That exact mistake
    // was made and caught in the bMat assembly.
    const char *t = getenv("SDPA_SOLVE_TEAM_OVERRIDE");
    if (t != NULL && t[0] != '\0') {
        c.team_override = (int)solve_gate_bounded("SDPA_SOLVE_TEAM_OVERRIDE", 0, 0, 65536);
    }
#else
    if (getenv("SDPA_SOLVE_TEAM_OVERRIDE") != NULL) {
        rError("SDPA_SOLVE_TEAM_OVERRIDE is a test hook and this binary was not built with"
               " -DSDPA_SPCHOL_TEST_HOOKS, so the hook does not exist here");
    }
    // Refused HERE, in the config, and not at the point of use. Put beside the code it
    // perturbs it would only be reached on the sparse route, so a production binary would
    // silently accept it on a dense problem -- the identical hole the review found in the
    // configuration parsing, reintroduced. My own dry-run caught it; solve_cfg() is the one
    // place both routes pass through.
    if (getenv("SDPA_SOLVE_MUTATE_VECPREC") != NULL) {
        rError("SDPA_SOLVE_MUTATE_VECPREC is a test hook and this binary was not built with"
               " -DSDPA_SPCHOL_TEST_HOOKS, so the hook does not exist here");
    }
#endif
#ifndef _OPENMP
    if (c.mode == SOLVE_PARALLEL) {
        rError("SDPA_SOLVE_MODE=parallel was requested but this binary was built without"
               " OpenMP, so there is no parallel path to take");
    }
    if (c.bwd_mode == SOLVE_PARALLEL) {
        rError("SDPA_SOLVE_BACKWARD=parallel was requested but this binary was built without"
               " OpenMP, so there is no parallel path to take");
    }
#endif
    return c;
}

// Recover the row extents from the flat (row_index, column_index) arrays and, in the same
// single pass, PROVE the four properties the parallel forward pass depends on. Returns false
// with a reason rather than aborting, because the legacy loop needs none of these properties
// and remains correct whatever the pattern is -- falling back is the right answer, and the
// caller decides whether a fallback is acceptable or (under a forced mode) a failure.
bool solve_build_rows(SparseMatrix &aMat, int n, std::vector<int> &rowstart, const char *&why) {
    const int nnz = aMat.NonZeroCount;
    if (n <= 0 || nnz <= 0) {
        why = "empty matrix";
        return false;
    }
    // Every stored element must carry the GLOBAL DEFAULT precision. This is not pedantry: the
    // parallel passes hoist their scratch out of the inner loop instead of letting gmpxx build
    // a fresh temporary per nonzero, and a hoisted scratch is only equivalent to the temporary
    // it replaces if the precision gmpxx would have chosen is the one we set. When every
    // operand and the default agree -- which is what this solver does, since it calls
    // mpf_set_default_prec once and allocates everything after -- that question has one answer
    // and no rule of gmpxx's can make it another. When they do not agree, the parallel path is
    // refused rather than guessed at.
    const mp_bitcnt_t dflt = mpf_get_default_prec();
    rowstart.assign((size_t)n + 1, 0);
    int cur = -1;
    int prevcol = -1;
    for (int k = 0; k < nnz; ++k) {
        const int r = aMat.row_index[k];
        const int c = aMat.column_index[k];
        if (mpf_get_prec(aMat.sp_ele[k].get_mpf_t()) != dflt) {
            why = "a factor element does not carry the default precision";
            return false;
        }
        if (r != cur) {
            // (1) rows appear contiguously and in ascending order, starting at 0
            if (r != cur + 1) {
                why = "rows are not contiguous and ascending";
                return false;
            }
            // (2) each row's first stored entry is its diagonal
            if (c != r) {
                why = "a row does not begin with its diagonal";
                return false;
            }
            // Bound BEFORE indexing. `r == cur + 1` alone lets a malformed pattern with more
            // rows than the matrix walk straight off the end of rowstart, and the closing
            // `cur == n - 1` test happens far too late to prevent the write.
            if (r >= n) {
                why = "a row index is outside the matrix";
                return false;
            }
            cur = r;
            rowstart[r] = k;
            prevcol = c;
        } else {
            // (3) columns within a row STRICTLY ascend. This is the property the whole
            //     bit-identity argument rests on: strict ascent implies distinct columns,
            //     distinct columns imply the row's scatter writes distinct destinations, and
            //     distinct destinations imply no race and no reassociation.
            if (c <= prevcol) {
                why = "columns within a row do not strictly ascend";
                return false;
            }
            // (4) strictly upper triangular off the diagonal, so a row never scatters into
            //     itself or into an already-final entry
            if (c <= r) {
                why = "an off-diagonal entry is not strictly upper triangular";
                return false;
            }
            // (5) IN RANGE. The threaded kernels index xVec with this column directly, so an
            //     out-of-range value is an out-of-bounds write. Strict ascent and c > r bound
            //     it from below and say nothing about the top.
            if (c >= n) {
                why = "a column index is outside the matrix";
                return false;
            }
            prevcol = c;
        }
    }
    if (cur != n - 1) {
        why = "the last row index does not match the matrix dimension";
        return false;
    }
    rowstart[n] = nnz;
    return true;
}

#ifdef _OPENMP
// Runs the forward pass and reports through *actual how many threads it really got.
//
// It ALWAYS does the work, including when the team turns out to be one. The earlier version
// bailed out and left the caller to run the legacy loop, which review 2026-08-15 showed was
// wrong twice over: the caller's forced-mode check had already run, so a forced `parallel`
// request silently downgraded to serial -- the exact failure the surrounding comments promise
// cannot happen -- and the caller had no way to tell "ran on one thread" from "ran on many".
//
// Running the kernel with a team of one is safe here, and the general rule it appears to break
// does not apply. That rule -- never leave a serial route inside an OpenMP construct -- exists
// because a team of one makes an inner Rgemm call NESTED, and nested parallelism is off by
// default, which is how the bMat assembly silently lost 7.7x on gpp124-1. The innermost work
// here is scalar GMP arithmetic with no threaded call anywhere beneath it, so there is nothing
// to nest and nothing to lose. With one thread `omp for` and `omp master` simply execute on
// that thread, and the result is the same bits by the same argument as for any other team size.
bool solve_forward_parallel(Vector &xVec, SparseMatrix &aMat, const std::vector<int> &rowstart,
                            int n, int team, mp_bitcnt_t prec, int *actual) {
    *actual = 0;
#pragma omp parallel num_threads(team)
    {
        // num_threads() is a REQUEST. Acting on the requested size when the runtime gave a
        // smaller team is the error that reopened gate 6 in the sparse Cholesky review; the
        // size is read back inside the one region that will use it.
#pragma omp master
        *actual = omp_get_num_threads();
        {
            // Private, and its precision is set from the vector rather than inherited from
            // the global default, because mpf assignment truncates to the DESTINATION's
            // precision. A private scratch one bit narrower than xVec would round every
            // scaled diagonal and the answer would differ in the last place while every
            // structural check still passed.
            mpf_class src;
            mpf_set_prec(src.get_mpf_t(), prec);
            // Hoisted out of the inner loop, and this is a THROUGHPUT fix, not a style one.
            // Written as `mpf_class value = sp_ele[k]` and `value * src`, gmpxx allocates and
            // frees two GMP temporaries for every nonzero -- about 9.6 million malloc/free
            // pairs per dE3 pass. Serially that is merely wasteful; across a team it is
            // allocator contention, and it is the largest part of why the first measurement
            // came in at 4.25x on 8 threads against a predicted 7.0x. Precision equality with
            // the default is proved in solve_build_rows(), so these carry exactly the values,
            // and exactly the precisions, that the temporaries they replace would have.
            mpf_class value, prod;
            mpf_set_prec(value.get_mpf_t(), prec);
            mpf_set_prec(prod.get_mpf_t(), prec);
            for (int i = 0; i < n; ++i) {
                const int s = rowstart[i];
                const int e = rowstart[i + 1];
                {
                    // Exactly the legacy diagonal operation, on a private copy: the same copy
                    // of sp_ele at the same precision, then the same in-place multiply into a
                    // destination of xVec's precision.
                    value = aMat.sp_ele[s];
                    src = xVec.ele[i];
                    src *= value;
                }
#pragma omp for schedule(static)
                for (int k = s + 1; k < e; ++k) {
                    // Never nowait. The implicit barrier here is the ONE synchronisation per
                    // row, and it is what lets the next row's scale read a settled x.
                    value = aMat.sp_ele[k];
                    prod = value * src; // evaluated straight into prod: no temporary at all
                    xVec.ele[aMat.column_index[k]] -= prod;
                }
#pragma omp master
                // Safe without a barrier of its own, and this is the whole trick: after the
                // implicit barrier above, every thread has already read x[i] for this row,
                // and no later row can touch x[i] because row i' > i writes only columns
                // j > i'. The master may lag, but it must reach the next row's `omp for`
                // before any thread can leave it, so the write always lands in time.
                xVec.ele[i] = src;
            }
        }
    }
    return true; // the work was done; *actual says by how many threads
}
#endif // _OPENMP

// ---------------------------------------------------------------------------
// Phase 2: the BACKWARD substitution, threaded. This one does change the answer, and it is
// opt-in for that reason -- SDPA_SOLVE_BACKWARD defaults to serial.
//
// Row i's off-diagonals all accumulate into the SAME x[i], so threading them reorders a
// finite-precision sum. No arrangement avoids that; the census established it and Phase 1's
// scatter trick does not transfer, because the next row genuinely reads x[i] and the write
// cannot be deferred past it.
//
// So the question is not whether the answer changes but WHICH answer it changes to, and two
// properties are worth more than bit-identity with the old one:
//
//   REPRODUCIBLE. The row is split into a FIXED number of chunks -- 64 by default, and
//   independent of the team size. Chunk boundaries come only from the row's extent and that
//   constant, and the partials are combined in a fixed descending order. So 1, 8, 32 and 64
//   threads all produce the SAME bits, and so does a rerun. Chunking by thread count instead
//   would have made the answer a property of the machine, which is a far worse thing to ship
//   than a one-time change in the last digits.
//
//   NO LESS ACCURATE. Each chunk accumulates into a scratch at twice the vector's precision
//   and is folded in once, where the legacy loop rounded into x[i] at every one of the row's
//   ~800 subtractions. Pairwise-style accumulation at higher precision is not a theorem for
//   every input, so this is a design intent to be checked against the legacy answer, not a
//   proof -- which is why the verification measures agreement rather than asserting it.
//
// Rows shorter than min_chunk go to a single thread and are computed EXACTLY as the legacy
// loop computes them. That is still deterministic: whether a row is chunked depends only on
// its length and the fixed chunk count, never on the team.
//
// Two barriers per row here, against Phase 1's one: the combined x[i] is read by row i-1, so
// unlike the forward pass the write cannot be deferred past the next barrier.
// ---------------------------------------------------------------------------
#ifdef _OPENMP
//
// Like the forward kernel it ALWAYS does the work and reports the actual team through *actual,
// and here that is not a tidiness point but a correctness one. The earlier version bailed out
// on a team of one and let the caller run the LEGACY loop, which computes a deliberately
// DIFFERENT answer. So the advertised property -- that 1, 8, 32 and 128 threads all produce
// identical bits -- was false in exactly the case a user cannot predict, a runtime that hands
// back a smaller team than asked. Running the fixed-chunk kernel on one thread produces the
// fixed-chunk answer, which is the whole point of fixing the chunk count independently of the
// team.
bool solve_backward_parallel(Vector &xVec, SparseMatrix &aMat, const std::vector<int> &rowstart,
                             int n, int team, mp_bitcnt_t prec, int chunks, int min_chunk,
                             int *actual) {
    *actual = 0;
    // Shared across the team, written once per chunk per row. Accumulation happens in a
    // thread-private scratch and only the finished chunk total is published, so the shared
    // headers are touched K times per row rather than once per nonzero -- the difference
    // between negligible false sharing and pathological false sharing.
    std::vector<mpf_class> partial((size_t)chunks);
    for (int c = 0; c < chunks; ++c) {
        mpf_set_prec(partial[(size_t)c].get_mpf_t(), 2 * prec);
    }

#pragma omp parallel num_threads(team)
    {
#pragma omp master
        *actual = omp_get_num_threads();
        {
            mpf_class acc;
            mpf_set_prec(acc.get_mpf_t(), 2 * prec);
            mpf_class value, prod;
            mpf_set_prec(value.get_mpf_t(), prec);
            mpf_set_prec(prod.get_mpf_t(), prec);
            for (int i = n - 1; i >= 0; --i) {
                const int s = rowstart[i];
                const int e = rowstart[i + 1];
                const int len = e - s - 1;
                // Evaluated from shared data, so every thread takes the same branch and the
                // worksharing constructs are encountered by the whole team in the same order.
                if (len >= min_chunk) {
#pragma omp for schedule(static)
                    for (int c = 0; c < chunks; ++c) {
                        const int lo = s + 1 + (int)((long long)c * len / chunks);
                        const int hi = s + 1 + (int)((long long)(c + 1) * len / chunks);
                        acc = 0;
                        // Descending within the chunk, matching the direction the legacy loop
                        // walks the row.
                        for (int k = hi - 1; k >= lo; --k) {
                            value = aMat.sp_ele[k];
                            // Straight into acc at 2x precision, which is the whole point of
                            // the chunk scratch; no per-nonzero temporary.
                            acc += value * xVec.ele[aMat.column_index[k]];
                        }
                        partial[(size_t)c] = acc;
                    }
#pragma omp single
                    {
                        for (int c = chunks - 1; c >= 0; --c) {
                            xVec.ele[i] -= partial[(size_t)c];
                        }
                        value = aMat.sp_ele[s];
                        xVec.ele[i] *= value;
                    }
                } else {
#pragma omp single
                    {
                        // Byte for byte the legacy row, including the order of operations.
                        for (int k = e - 1; k > s; --k) {
                            value = aMat.sp_ele[k];
                            prod = value * xVec.ele[aMat.column_index[k]];
                            xVec.ele[i] -= prod;
                        }
                        value = aMat.sp_ele[s];
                        xVec.ele[i] *= value;
                    }
                }
            }
        }
    }
    return true; // the work was done, in fixed chunks; *actual says by how many threads
}
#endif // _OPENMP

} // namespace

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: public entry point for the canonical
   stream, so the bMat assembly and the factor share one serialiser. See sdpa_linear.h. */
CanonicalStream canonicalSparseStream(SparseMatrix &aMat, int *diagonalIndex, int nDIM,
                                      const char *tag, FILE *dump) {
    return spchol_digest_impl(aMat, diagonalIndex, nDIM, tag, dump);
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: canonical stream of a solved vector,
   the oracle for the threaded forward substitution. See sdpa_linear.h. */
// The oracle Phase 1 of the parallel solve is judged by, and it has to be this and not the
// printed answer. SDPA prints a fixed number of digits, so two solutions that differ in the
// last bits print the same -- exactly the divergence a reassociation would introduce, and
// exactly what a "byte-identical output file" check would therefore MISS. This emits every
// element's full mantissa, so a single changed bit anywhere in x changes the stream.
//
// Appended to, one record per solve call, so a run's whole sequence of solves is compared in
// order rather than only its last. Same framing and same helpers as the factor and assembly
// streams; a DIFFERENT tag, so a solve stream can never compare equal to either of those.
CanonicalStream canonicalVectorStream(Vector &v, const char *tag, FILE *dump) {
    CanonicalStream d;
    d.fnv = 14695981039346656037ULL;
    d.records = 0;
    d.bytes = 0;
    d.dump = dump;
    d.io_error = false;

    spchol_dg_bytes(d, tag, strlen(tag));
    spchol_dg_i64(d, v.nDim);

    void (*freefunc)(void *, size_t) = NULL;
    mp_get_memory_functions(NULL, NULL, &freefunc);
    spchol_dg_u64(d, (uint64_t)(v.nDim > 0 ? v.nDim : 0));
    for (int i = 0; i < v.nDim; ++i) {
        spchol_dg_byte(d, 'X'); // record tag
        spchol_dg_i64(d, i);
        mpf_srcptr x = v.ele[i].get_mpf_t();
        // Precision is part of the record. An element that silently changed precision would
        // otherwise be able to carry the same digits and compare equal.
        spchol_dg_u64(d, (uint64_t)mpf_get_prec(x));
        spchol_dg_i64(d, mpf_sgn(x));
        mp_exp_t ex = 0;
        // Base 16, 0 digits requested: the exact value, no rounding and no locale.
        char *s = mpf_get_str(NULL, &ex, 16, 0, x);
        spchol_dg_i64(d, (long long)ex);
        spchol_dg_bytes(d, (s != NULL) ? s : "", (s != NULL) ? strlen(s) : 0);
        if (s != NULL && freefunc != NULL) {
            freefunc(s, strlen(s) + 1);
        }
        d.records++;
    }
    spchol_dg_byte(d, '.'); // terminator, so a truncated record cannot look complete
    return d;
}

namespace {

// Emits the solved vector's canonical stream when asked, and only when asked.
//
//   SDPA_SOLVE_DUMP=f    the stream itself, APPENDED to f, one record per solve call, so two
//                        runs are compared with cmp(1) over the whole sequence. Proof-grade.
//   SDPA_SOLVE_DIGEST=1  fingerprint plus record and byte counts. Cheap enough to leave on,
//                        and enough to notice a change -- but a fingerprint, not a proof.
//
// Every failure path is fatal rather than best-effort. A truncated dump silently compares
// equal to another truncated dump, and the sparse Cholesky work already came within one step
// of reporting byte-identity from two 0-byte files.
void emit_solve_stream(Vector &xVec) {
    const char *dumpf = getenv("SDPA_SOLVE_DUMP");
    const char *want = getenv("SDPA_SOLVE_DIGEST");
    const bool digest = (want != NULL && want[0] != '\0' && strcmp(want, "0") != 0);
    const bool dumping = (dumpf != NULL && dumpf[0] != '\0');
    if (!dumping && !digest) {
        return;
    }

    FILE *fp = NULL;
    if (dumping) {
        fp = fopen(dumpf, "ab");
        if (fp == NULL) {
            rError("SDPA_SOLVE_DUMP could not open \"" << dumpf << "\" for appending");
        }
    }
    CanonicalStream s = canonicalVectorStream(xVec, "SOLVEXv1", fp);
    if (fp != NULL) {
        if (s.io_error || fflush(fp) != 0 || ferror(fp) != 0) {
            rError("SDPA_SOLVE_DUMP: writing \"" << dumpf << "\" failed, so the dump is"
                                                 << " truncated and must not be compared");
        }
        if (fclose(fp) != 0) {
            rError("SDPA_SOLVE_DUMP: closing \"" << dumpf << "\" failed, so the dump may be"
                                                 << " truncated and must not be compared");
        }
    }
    if (digest) {
        rMessage("solve x stream: records " << s.records << " bytes " << s.bytes << " fnv "
                                            << s.fnv);
    }
}

} // namespace

bool Lal::getCholesky(SparseMatrix &aMat, int *diagonalIndex) {
    if (aMat.type != SparseMatrix::SPARSE) {
        rError("Lal::getCholesky aMat is not sparse format");
    }
    const int nDIM = aMat.nRow;
    SpcholStats st;
    memset(&st, 0, sizeof(st));
    st.path = "serial";
    st.fallback = "";

    const SpcholMode mode = spchol_mode();
    // EVERY tunable is parsed HERE, before any mode-specific early return, and nowhere else.
    // spchol_parallel() runs only when the gates admit it, so anything parsed inside it is
    // silently ignored on the serial, legacy, nested, below-gate and no-OpenMP paths. A
    // variable that validates only on some code paths is not validated. SDPA_SPCHOL_FAIL_AT
    // was caught this way by CI (it accepted -1 on a below-gate fixture); the three threshold
    // variables had the identical defect and were found by review in the same commit that
    // recorded the lesson.
    const SpcholCfg cfg = spchol_cfg();
    if (cfg.fail_at >= nDIM) {
        rError("SDPA_SPCHOL_FAIL_AT=" << cfg.fail_at << " is past the last pivot index "
               << (nDIM - 1) << "; it would never fire and the test would pass vacuously");
    }
    if (mode == SPCHOL_LEGACY) {
        st.path = "legacy";
        return spchol_finish(aMat, diagonalIndex, nDIM, cfg, st,
                             spchol_legacy(aMat, diagonalIndex, st));
    }

    // Fix B is bit-identical to the pre-B expression only if the operand copies it removes
    // were themselves exact. Upstream copied into function-scope mpf_class temporaries,
    // which carry mpf_get_default_prec(), and the product temporary took the DESTINATION's
    // precision. So both must equal the factor's precision. True in this solver because the
    // parameter file sets the default before sp_ele is allocated -- asserted, not assumed.
    const mp_bitcnt_t factor_prec =
        (nDIM > 0) ? mpf_get_prec(aMat.sp_ele[diagonalIndex[0]].get_mpf_t())
                   : mpf_get_default_prec();
    if (factor_prec != mpf_get_default_prec()) {
        rError("Lal::getCholesky: factor precision " << (unsigned long)factor_prec
               << " != default precision " << (unsigned long)mpf_get_default_prec()
               << "; the scratch-based update is bit-identical only when they agree");
    }
    spchol_assert_uniform_precision(aMat, factor_prec);

    uint64_t total_work = 0;
    int useful_width = 0;
    for (int i = 0; i < nDIM; ++i) {
        const int a1 = diagonalIndex[i], a2 = diagonalIndex[i + 1];
        total_work += spchol_work(a1, a2);
        if (spchol_width(a1, a2) > useful_width)
            useful_width = spchol_width(a1, a2);
    }

#ifdef _OPENMP
    // Whole-factor admission, decided ONCE, on the final team size after every cap.
    // omp_get_level() rather than omp_in_parallel(): an inactive or serialized enclosing
    // region reports false for in_parallel() while still raising the nesting level, and a
    // nested sparse team is not something this routine is validated for.
    int team = omp_get_max_threads();
    const int tl = omp_get_thread_limit();
    if (tl > 0 && tl < team)
        team = tl;
    if (useful_width < team)
        team = useful_width; // capping to the widest pivot in the whole factor
    // cfg is already parsed, so the short circuit no longer decides whether _TOTAL is
    // validated. It used to: in forced-parallel mode the right-hand side never ran.
    const bool gate_ok = (mode == SPCHOL_PARALLEL) || (total_work >= cfg.gate_total);
    if (mode == SPCHOL_SERIAL) {
        st.fallback = "SDPA_SPCHOL_MODE=serial";
    } else if (omp_get_level() != 0) {
        st.fallback = "nested inside another OpenMP region";
    } else if (team < 2) {
        st.fallback = "team would be 1 (threads, thread limit or factor width)";
    } else if (!gate_ok) {
        st.fallback = "total work below SDPA_OMP_MIN_SPCHOL_TOTAL";
    } else {
        // No probe: the contraction check lives inside the single region that owns the team
        // (see spchol_parallel). A throwaway probe region's team size does not bind the next
        // region's, so dispatching on it was an inference the standard does not support.
        // spchol_parallel() reports back through st.path, and rewrites it to "serial" if the
        // runtime handed it one thread and it ran spchol_serial() on that thread.
        // Applied AFTER admission, so it reproduces a runtime that contracts a team the gates
        // had already accepted -- the state the in-region branch exists for. 0 in every build
        // without the test hooks, where the variable is refused outright.
        if (cfg.team_override > 0)
            team = cfg.team_override;
        st.team_requested = team;
        st.path = "parallel";
        return spchol_finish(aMat, diagonalIndex, nDIM, cfg, st,
                             spchol_parallel(aMat, diagonalIndex, factor_prec, team,
                                             mode == SPCHOL_PARALLEL, cfg, st));
    }
#else
    st.fallback = "built without OpenMP";
    if (mode == SPCHOL_PARALLEL) {
        rError("SDPA_SPCHOL_MODE=parallel but this binary was built without OpenMP");
    }
#endif

    return spchol_finish(aMat, diagonalIndex, nDIM, cfg, st,
                         spchol_serial(aMat, diagonalIndex, factor_prec, st));
}

bool Lal::getInvLowTriangularMatrix(DenseMatrix &retMat, DenseMatrix &aMat) {
    mpf_class MONE = 1.0;
    // Make inverse with refference only to lower triangular.
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.type != aMat.type) {
        rError("getCholesky:: different memory size");
    }
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        retMat.setIdentity();
        /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: Rtrsm -> Rtrsm_omp (B3, ported
           from sdpa-dd). This solve against an identity RHS is one of the two halves of the
           Cholesky-inverse phase and was entirely serial. Rtrsm_omp splits it over the columns
           of retMat, which leaves the arithmetic of each column and its order untouched, so the
           result is bit-identical to Rtrsm; it falls back to Rtrsm below its work gate and for
           any case but Left/Lower/NoTranspose. Deliberately NOT done by threading Rtrsm itself,
           which would also thread Rpotrf's and Rpotrf2's inner solves. See git log. */
        Rtrsm_omp("Left", "Lower", "NoTraspose", "NonUnitDiagonal", aMat.nRow, aMat.nCol, MONE, aMat.de_ele, aMat.nRow, retMat.de_ele, retMat.nRow);
        break;
    case DenseMatrix::COMPLETION:
        rError("DenseMatrix:: no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::getSymmetrize(DenseMatrix &aMat) {
    mpf_class MONE = 1.0;
    switch (aMat.type) {
    case DenseMatrix::DENSE:
        if (aMat.nRow != aMat.nCol) {
            rError("getSymmetrize:: different memory size");
        }
        for (int index = 0; index < aMat.nRow - 1; ++index) {
            int index1 = index + index * aMat.nRow + 1;
            int index2 = index + (index + 1) * aMat.nRow;
            int length = aMat.nRow - 1 - index;
            // aMat.de_ele[index1] += aMat.de_ele[index2]
            Raxpy(length, MONE, &aMat.de_ele[index2], aMat.nRow, &aMat.de_ele[index1], 1);
            // aMat.de_ele[index1] /= 2.0
            mpf_class half = 0.5;
            Rscal(length, half, &aMat.de_ele[index1], 1);
            // aMat.de_ele[index2] = aMat.de_ele[index1]
            Rcopy(length, &aMat.de_ele[index1], 1, &aMat.de_ele[index2], aMat.nRow);
        }
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::getTranspose(DenseMatrix &retMat, DenseMatrix &aMat) {
    if (aMat.nRow != aMat.nCol) {
        rError("getTranspose:: different memory size");
        // Of course, a non-symmetric matrix has
        // its transposed matrix,
        // but in this algorithm we have to make
        // transposed matrix only when symmetric matrix.
    }
    retMat.copyFrom(aMat);
    switch (aMat.type) {
    case DenseMatrix::DENSE:
#if 0
    for (int i=0; i<aMat.nRow; ++i) {
      for (int j=0; j<=i; ++j) {
	int index1 = i+aMat.nCol*j;
	int index2 = j+aMat.nCol*i;
	retMat.de_ele[index1] = aMat.de_ele[index2];
	retMat.de_ele[index2] = aMat.de_ele[index1];
      }
    }
#else
        for (int i = 0; i < aMat.nRow; ++i) {
            int shou = (i + 1) / 4;
            int amari = (i + 1) / 4;
            for (int j = 0; j < amari; ++j) {
                int index1 = i + aMat.nCol * j;
                int index2 = j + aMat.nCol * i;
                retMat.de_ele[index1] = aMat.de_ele[index2];
                retMat.de_ele[index2] = aMat.de_ele[index1];
            }
            for (int j = amari, counter = 0; counter < shou; ++counter, j += 4) {
                int index1 = i + aMat.nCol * j;
                int index_1 = j + aMat.nCol * i;
                retMat.de_ele[index1] = aMat.de_ele[index_1];
                retMat.de_ele[index_1] = aMat.de_ele[index1];
                int index2 = i + aMat.nCol * (j + 1);
                int index_2 = (j + 1) + aMat.nCol * i;
                retMat.de_ele[index2] = aMat.de_ele[index_2];
                retMat.de_ele[index_2] = aMat.de_ele[index2];
                int index3 = i + aMat.nCol * (j + 2);
                int index_3 = (j + 2) + aMat.nCol * i;
                retMat.de_ele[index3] = aMat.de_ele[index_3];
                retMat.de_ele[index_3] = aMat.de_ele[index3];
                int index4 = i + aMat.nCol * (j + 3);
                int index_4 = (j + 3) + aMat.nCol * i;
                retMat.de_ele[index4] = aMat.de_ele[index_4];
                retMat.de_ele[index_4] = aMat.de_ele[index4];
            }
        }
#endif
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

int Lal::rdpotf2_(char *uplo, int *n, double *a, int *lda, int *info) {
#if USE_DOUBLE
    int nRow = *lda;
    for (int j = 0; j < *n; ++j) {
        double ajj = a[j + nRow * j] - ddot_f77(&j, &a[j], lda, &a[j], lda);

        // Here is point.(start)
        if (ajj <= (float)-1.0e-6) {
            a[j + j * nRow] = ajj;
            *info = j + 1;
            return 0;
        }
        if (ajj <= (float)1.0e-14) {
            ajj = 1e100;
            a[j + j * nRow] = ajj;
        } else {
            ajj = sqrt(ajj);
            a[j + j * nRow] = ajj;
        }
        // Here is point.(end)

        if (j < *n - 1) {
            int i = *n - 1 - j;
            dgemv_f77("No transpose", &i, &j, &DMONE, &a[j + 1], lda, &a[j], lda, &DONE, &a[(j + 1) + nRow * j], &IONE, strlen("No transpose"));
            double d1 = 1.0 / ajj;
            dscal_f77(&i, &d1, &a[(j + 1) + nRow * j], &IONE);
        }
    }
#endif
    return 0;
}

int Lal::rdpotrf_(char *uplo, int *n, double *a, int *lda, int *info) {
#if USE_DOUBLE
    // This funciton makes Cholesky factorization
    // in only case Lower Triangular.
    // That is, A will be L*L**T, not U**T*U.
    int nRow = *lda;
    *info = 0;

    int nb = ilaenv_f77(&IONE, "DPOTRF", "L", n, &IMONE, &IONE, &IMONE, strlen("DPOTRF"), strlen("L"));
    if (nb <= 1 || nb >= *n) {
        // Here is point.
        rdpotf2_(uplo, n, a, lda, info);
    } else {

        for (int j = 0; j < *n; j += nb) {
            int jb = min(nb, *n - j);
            dsyrk_f77("Lower", "No transpose", &jb, &j, &DMONE, &a[j], lda, &DONE, &a[j + nRow * j], lda, strlen("Lower"), strlen("No transpose"));
            // Here is point.
            rdpotf2_("Lower", &jb, &a[j + nRow * j], lda, info);
            if (*info != 0) {
                *info = *info + j - 1;
                return 0;
            }
            if (j + jb <= *n - 1) {
                int i = *n - j - jb;
                dgemm_f77("No transpose", "Transpose", &i, &jb, &j, &DMONE, &a[j + jb], lda, &a[j], lda, &DONE, &a[(j + jb) + nRow * j], lda, strlen("No transpose"), strlen("Transpose"));
                dtrsm_f77("Right", "Lower", "Transpose", "Non-unit", &i, &jb, &DONE, &a[j + nRow * j], lda, &a[(j + jb) + nRow * j], lda, strlen("Right"), strlen("Lower"), strlen("Transpose"), strlen("Non-unit"));
            }
        }
    }
#endif
    return 0;
}

bool Lal::choleskyFactorWithAdjust(DenseMatrix &aMat) {
    mplapackint info = 0;
#if 1
    // aMat.display();
    TimeStart(START1);
    Rpotrf("Lower", aMat.nRow, aMat.de_ele, aMat.nRow, info);
    TimeEnd(END1);
    // rMessage("Schur colesky  ::"  << TimeCal(START1,END1));
    // aMat.display();
#elif 1
    dpotrf_f77("Lower", &aMat.nRow, aMat.de_ele, &aMat.nRow, &info, strlen("Lower"));
#else
    rdpotrf_("Lower", &aMat.nRow, aMat.de_ele, &aMat.nRow, &info);
#endif
    if (info < 0) {
        rMessage("cholesky argument is wrong " << -info);
    } else if (info > 0) {
        rMessage("cholesky miss condition :: not positive definite"
                 << " :: info = " << info);

        return FAILURE;
    }
    return _SUCCESS;
#if 0
  mpf_class ZERO_DETECT = 1.0e-3;
  mpf_class NONZERO = 1.0e-7;
  // no idea version
  // if Cholesky factorization failed, then exit soon.
  int info = 1; // info == 0 means success
  int start = 0;
  while (start<aMat.nRow) {
    int N = aMat.nRow - start;
    dpotf2_("Lower",&N,&aMat.de_ele[start+start*aMat.nRow],
	    &aMat.nRow,&info);
    if (info <=0) {
      // rMessage("Cholesky is very nice");
      break;
    }
    start += (info-1); // next target
    mpf_class wrong = aMat.de_ele[start+start*aMat.nRow];
    if (wrong < -ZERO_DETECT) {
      rMessage("cholesky adjust position " << start);
      rMessage("cannot cholesky decomposition"
	       " with adjust " << wrong);
      return FAILURE;
    }
    aMat.de_ele[start+start*aMat.nRow] = NONZERO;
    if (start<aMat.nRow-1) {
      // improve the right down element of 0
      for (int j=1; j<=aMat.nRow-1-start; ++j) {
	mpf_class& migi  = aMat.de_ele[start+(start+j)*aMat.nRow];
	mpf_class& shita = aMat.de_ele[(start+j)+start*aMat.nRow];
	mpf_class& mishi = aMat.de_ele[(start+j)+(start+j)*aMat.nRow];
	// rMessage(" mishi = " << mishi);
	if (mishi < NONZERO) {
	  // rMessage(" mishi < NONZERO ");
	  mishi = NONZERO;
	  migi  = NONZERO * 0.1;
	  shita = NONZERO * 0.1;
	} else if (migi*shita > NONZERO*mishi) {
	  // rMessage(" migi*migi > NONZERO*mishi ");
	  migi  = sqrt(NONZERO*mishi) * 0.99;
	  shita = sqrt(NONZERO*mishi) * 0.99;
	}
      }
    }
    rMessage("cholesky adjust position " << start);
  }
  if (info < 0) {
    rError("argument is something wrong " << info);
  }
  return _SUCCESS;
#endif
}

bool Lal::solveSystems(Vector &xVec, DenseMatrix &aMat, Vector &bVec) {
    // aMat must have done Cholesky factorized.
    if (aMat.nCol != xVec.nDim || aMat.nRow != bVec.nDim || aMat.nRow != aMat.nCol) {
        rError("solveSystems:: different memory size");
    }
    if (aMat.type != DenseMatrix::DENSE) {
        rError("solveSystems:: matrix type must be DENSE");
    }
    // Validate the solve options HERE too. They were parsed only inside the sparse overload,
    // so on a problem whose bMat goes dense a typo like SDPA_SOLVE_MODE=paralel was silently
    // ignored and a forced `parallel` silently did nothing -- making "every knob strictly
    // parsed" and "forced modes refuse rather than downgrade" untrue on one of the solver's
    // two routes. Parsing is unconditional now, so a malformed value is refused whichever
    // route the problem takes.
    const SolveCfg dcfg = solve_cfg();
    // A forced mode names the SPARSE triangular substitution, which does not exist on this
    // route. Refusing is consistent with how the sparse route treats a forced mode it cannot
    // honour; silently ignoring it is what the review objected to. The message says which
    // route was taken, because "it did nothing" is the confusing part, not the refusal.
    if (dcfg.mode == SOLVE_PARALLEL || dcfg.bwd_mode == SOLVE_PARALLEL) {
        rError("a forced SDPA_SOLVE_* parallel mode was requested, but this problem's bMat is"
               " DENSE and the threaded passes exist only on the sparse triangular solve."
               " Unset it, or use SDPA_BMAT_MODE=sparse if the sparse path is what you want");
    }
    solve_profile_note_dense();
    if (dcfg.log) {
        static bool dlogged = false;
        if (!dlogged) {
            dlogged = true;
            rMessage("solve: DENSE route (Rtrsv), n=" << aMat.nRow << ". The threaded passes"
                     " cover the sparse triangular solve only, so no solve knob applies here");
        }
    }
    xVec.copyFrom(bVec);
    Rtrsv("Lower", "NoTranspose", "NonUnit", aMat.nRow, aMat.de_ele, aMat.nCol, xVec.ele, 1);
    Rtrsv("Lower", "Transpose", "NonUnit", aMat.nRow, aMat.de_ele, aMat.nCol, xVec.ele, 1);
    // The solved-vector stream is emitted here too. SDPA_SOLVE_DUMP and SDPA_SOLVE_DIGEST were
    // live only in the sparse overload, so on a dense bMat they were silently ineffective --
    // the same shape of hole as the configuration parsing, and the README described all three
    // without a sparse-only qualification.
    emit_solve_stream(xVec);
    return _SUCCESS;
}

// nakata 2004/12/01
bool Lal::solveSystems(Vector &xVec, SparseMatrix &aMat, Vector &bVec) {
#define TUNEUP 0
#if TUNEUP
    if (aMat.nCol != xVec.nDim || aMat.nRow != bVec.nDim || aMat.nRow != aMat.nCol) {
        printf("A.row:%d A.col:%d x.row:%d b.row:%d\n", aMat.nCol, aMat.nRow, xVec.nDim, bVec.nDim);
        rError("solveSystems(sparse):: different memory size");
    }
    int length;
    int amari, shou, counter;

    switch (aMat.type) {
    case SparseMatrix::SPARSE:
#endif
        // Attension: in SPARSE case, only half elements
        // are stored. And bMat must be DENSE case.
        // rMessage("aMat.NonZeroCount == " << aMat.NonZeroCount);

        // Measurement only -- see solve_profile_dump(). Three clock reads per solve call
        // against passes that run for milliseconds; the perturbation is below the noise the
        // profile reports as fwd_sd/bwd_sd, which is why that spread is printed.
        const bool prof = solve_profile_enabled();
        double t_copy = 0.0, t_fwd = 0.0, t_bwd = 0.0;
        double t_mark = prof ? Time::rGetUseTime() : 0.0;

        xVec.copyFrom(bVec);

        if (prof) {
            const double now = Time::rGetUseTime();
            t_copy = now - t_mark;
            t_mark = now;
        }
#if TUNEUP

        shou = aMat.NonZeroCount / 4;
        amari = aMat.NonZeroCount % 4;
        int i, j;
        mpf_class value;

        for (int index = 0; index < amari; ++index) {
            int i = aMat.row_index[index];
            int j = aMat.column_index[index];
            mpf_class value = aMat.sp_ele[index];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
        }

        for (int index = amari, counter = 0; counter < shou; ++counter, index += 4) {
            i = aMat.row_index[index];
            j = aMat.column_index[index];
            value = aMat.sp_ele[index];
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
            i = aMat.row_index[index + 1];
            j = aMat.column_index[index + 1];
            value = aMat.sp_ele[index + 1];
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
            i = aMat.row_index[index + 2];
            j = aMat.column_index[index + 2];
            value = aMat.sp_ele[index + 2];
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
            i = aMat.row_index[index + 3];
            j = aMat.column_index[index + 3];
            value = aMat.sp_ele[index + 3];
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
        }

        for (int index = aMat.NonZeroCount - 1; index >= aMat.NonZeroCount - amari; --index) {
            i = aMat.row_index[index];
            j = aMat.column_index[index];
            value = aMat.sp_ele[index];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[j] -= value * xVec.ele[i];
            }
        }

        for (int index = aMat.NonZeroCount - amari - 1, counter = 0; counter < shou; ++counter, index -= 4) {
            i = aMat.row_index[index];
            j = aMat.column_index[index];
            value = aMat.sp_ele[index];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[i] -= value * xVec.ele[j];
            }
            i = aMat.row_index[index - 1];
            j = aMat.column_index[index - 1];
            value = aMat.sp_ele[index - 1];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[i] -= value * xVec.ele[j];
            }
            i = aMat.row_index[index - 2];
            j = aMat.column_index[index - 2];
            value = aMat.sp_ele[index - 2];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[i] -= value * xVec.ele[j];
            }
            i = aMat.row_index[index - 3];
            j = aMat.column_index[index - 3];
            value = aMat.sp_ele[index - 3];
            // rMessage("i=" << i << "  j=" << j);
            if (i == j) {
                xVec.ele[i] *= value;
            } else {
                xVec.ele[i] -= value * xVec.ele[j];
            }
        }
#else
    // FORWARD pass. Ascending index order walks rows ascending; within row i the first entry
    // is the diagonal (scale x[i]) and the rest scatter into DISTINCT columns j > i. That
    // distinctness is what lets the row's scatter be threaded without changing a bit; see
    // solve_forward_parallel(). Everything below falls back to this exact loop whenever the
    // parallel path is not taken, and the fallback runs OUTSIDE every OpenMP construct.
    bool fwd_parallel = false;
    bool bwd_parallel = false;
    double t_idx = 0.0;
    // Shared by both passes, so the pattern proof, the precision sweep and the team are paid
    // for ONCE per solve call rather than once per pass.
    const SolveCfg cfg = solve_cfg();
    const int n = aMat.nRow;
    std::vector<int> rowstart;
    mp_bitcnt_t prec = 0;
    int team = 0;
    const char *why = "";
    bool eligible = true; // structural: says nothing about whether either mode wants it
    bool size_ok = true;  // the auto gate; a forced mode bypasses this but not `eligible`
    {
        // The live path has NO dimension check -- the one at the top of this function is
        // inside `#if TUNEUP` and so is compiled out. Rather than change that (a run that
        // works today should not start aborting because the solve grew a threaded path),
        // require the dimensions for the parallel route only.
        if (n <= 0 || aMat.nCol != n || xVec.nDim != n) {
            eligible = false;
            why = "matrix is not square or does not match the vector";
        }
#ifdef _OPENMP
        if (eligible && omp_get_level() != 0) {
            // Already inside a region: a nested team is off by default, so "parallel" here
            // would mean a team of one wrapped in a construct -- strictly worse than serial.
            eligible = false;
            why = "already inside a parallel region";
        }
#else
        if (eligible) {
            eligible = false;
            why = "built without OpenMP";
        }
#endif

        // The size gate is evaluated FIRST, because it needs only n and NonZeroCount while the
        // pattern proof walks every nonzero. Measured on dE3 that walk costs 25 ms per solve
        // call, and the first version of this ran it before knowing whether either pass would
        // use the result -- so a run with both passes serial, or a problem that fails the gate,
        // paid for a proof it then discarded. On dE3 that was 1.685 s of the serial arm's
        // 78.8 s solve, for nothing.
        if (eligible) {
            const long long offdiag = (long long)aMat.NonZeroCount - (long long)n;
            const long long mean_row = offdiag / (long long)n;
            if ((uint64_t)n < cfg.min_rows) {
                size_ok = false;
                why = "too few rows for a per-row barrier to pay for itself";
                // Unsigned on both sides. Casting the gate to long long let a value above
                // LLONG_MAX become negative and INVERT the test, admitting what it was set to
                // exclude. mean_row is a count and cannot be negative.
            } else if ((uint64_t)mean_row < cfg.min_mean_row) {
                size_ok = false;
                why = "mean row is shorter than SDPA_SOLVE_MIN_ROW";
            }
        }
        const bool fwd_wants = (cfg.mode != SOLVE_SERIAL) && (size_ok || cfg.mode == SOLVE_PARALLEL);
        const bool bwd_wants =
            (cfg.bwd_mode != SOLVE_SERIAL) && (size_ok || cfg.bwd_mode == SOLVE_PARALLEL);
        if (eligible && !fwd_wants && !bwd_wants) {
            eligible = false;
            if (why[0] == '\0') {
                why = "neither pass is enabled";
            }
        }

        if (eligible) {
            const double t_i0 = Time::rGetUseTime();
            eligible = solve_build_rows(aMat, n, rowstart, why);
            if (eligible) {
                // Every element of x must share one precision, because the private scaled
                // diagonal is created at that precision and mpf assignment truncates to the
                // destination's. Checking the first element only is what the sparse Cholesky
                // review rejected, so this checks all of them; it is n integer reads against
                // a pass that does millions of multiplies.
#ifdef SDPA_SPCHOL_TEST_HOOKS
                // TEST ONLY: give one element a different precision, so the uniformity
                // refusal below has a control. CI previously mutated a FACTOR element's
                // precision and never an xVec one, so this branch was unexercised.
                if (getenv("SDPA_SOLVE_MUTATE_VECPREC") != NULL && n > 1) {
                    mpf_set_prec(xVec.ele[n - 1].get_mpf_t(),
                                 mpf_get_prec(xVec.ele[0].get_mpf_t()) + 64);
                }
#endif
                prec = mpf_get_prec(xVec.ele[0].get_mpf_t());
                for (int i = 1; i < n; ++i) {
                    if (mpf_get_prec(xVec.ele[i].get_mpf_t()) != prec) {
                        eligible = false;
                        why = "the solution vector does not have a uniform precision";
                        break;
                    }
                }
                // Mutual uniformity of the VECTOR is not enough, and claiming it was is a hole
                // review 2026-08-15 found. solve_build_rows proves every FACTOR element carries
                // the default precision; the threaded scratch is then created at the VECTOR's
                // precision. Where the legacy loop copies sp_ele into a temporary that inherits
                // the FACTOR's precision, the threaded path would use the vector's -- so a
                // caller handing in a differently-precised vector gets an operand rounded at a
                // point the legacy path does not round it, and bit-identity quietly fails.
                // Requiring all three to agree closes it; refusing is correct because the
                // parallel path has no claim to make otherwise.
                if (eligible && prec != mpf_get_default_prec()) {
                    eligible = false;
                    why = "the solution vector's precision differs from the factor's";
                }
            }
            t_idx = Time::rGetUseTime() - t_i0;
        }

#ifdef _OPENMP
        if (eligible) {
            team = omp_get_max_threads();
            // Cap, and it is not a micro-optimisation -- it is the difference between a 6.7x
            // win and a 0.88x LOSS. Measured on dE3 (EPYC 7532, two sockets of 32 cores): the
            // forward pass runs 6.85x at 16 threads and 6.69x at 32, then collapses to 0.88x at
            // 64, with the fastest single call of any arm (0.0588 s, 9.6x) sitting next to a
            // 5.21 s outlier. The backward pass on the same node, same team, same per-row
            // barriers, is stable at 64 (5.04x, sd 0.004). What separates them is the memory
            // traffic: forward SCATTERS across the whole of x, so once the team spans two
            // sockets every row's barrier waits on cross-socket writes, while backward gathers
            // and writes only 64 chunk slots and one element. The pass saturates by 16 threads
            // anyway, so a cap costs nothing and removes the cliff for anyone who simply runs
            // with OMP_NUM_THREADS=64.
            //
            // 32 is this machine's socket width and will be wrong elsewhere, which is exactly
            // the trap bMat gate 3 fell into. So it is a settable knob, it is logged, and the
            // number it caps does NOT change the answer: the backward chunk count is fixed
            // independently of the team, so capping alters speed and never results.
            if (cfg.max_team > 0 && (uint64_t)team > cfg.max_team) {
                team = (int)cfg.max_team;
            }
            if (cfg.team_override > 0) {
                team = cfg.team_override;
            }
            // NO team<2 rejection here any more, and that is the point. Killing eligibility
            // for the whole solve when only one thread is available made the two passes share
            // a decision they do not share:
            //
            //   forward   at one thread the legacy loop is BIT-IDENTICAL, so falling back to
            //             it costs nothing and saves entering a region for one thread;
            //   backward  the legacy loop computes a DELIBERATELY DIFFERENT answer, so falling
            //             back to it breaks the one property Phase 2 exists to provide.
            //
            // The earlier fix repaired only the case where a runtime CONTRACTS the team.
            // Review 2026-08-15 (second round) pointed out the shipped routes -- plain
            // OMP_NUM_THREADS=1, or SDPA_SOLVE_MAX_TEAM=1 -- still took the legacy backward
            // loop, so "1, 8, 32 and 128 threads produce identical bits" was still false, and
            // the cap really could change the answer. Admission is now per pass, below.
        }
#endif

        // A forced mode that cannot be honoured is refused, not quietly downgraded. Silently
        // serialising a forced-parallel request is precisely how the bMat scratch bug hid: it
        // collapsed the team to one and the run still looked like a parallel arm.
        if (!eligible && cfg.mode == SOLVE_PARALLEL) {
            rError("SDPA_SOLVE_MODE=parallel was requested but the forward pass cannot be"
                   " threaded here: " << why);
        }
        if (!eligible && cfg.bwd_mode == SOLVE_PARALLEL) {
            rError("SDPA_SOLVE_BACKWARD=parallel was requested but the backward pass cannot"
                   " be threaded here: " << why);
        }
    }

    // The forced-mode refusals below are repeated AFTER each kernel returns, not only before
    // it is called. Checking only beforehand was a hole: the runtime may hand back a smaller
    // team than requested, and the old code then fell through to the legacy loop with the
    // forced-mode check already behind it -- silently downgrading a forced `parallel` request,
    // which is precisely what the comment above promises cannot happen.
    int fwd_actual = 0, bwd_actual = 0;
#ifdef _OPENMP
    // FORWARD: only worth a region when there is a real team, because one thread through the
    // legacy loop produces the very same bits.
    if (eligible && cfg.mode != SOLVE_SERIAL && (size_ok || cfg.mode == SOLVE_PARALLEL) &&
        (team >= 2 || cfg.team_override > 0)) {
        fwd_parallel = solve_forward_parallel(xVec, aMat, rowstart, n, team, prec, &fwd_actual);
    }
    if (eligible && cfg.mode == SOLVE_PARALLEL && team < 2) {
        rError("SDPA_SOLVE_MODE=parallel was requested but only one thread is available");
    }
    {
        if (fwd_parallel && fwd_actual < 2 && cfg.mode == SOLVE_PARALLEL) {
            rError("SDPA_SOLVE_MODE=parallel was requested but the runtime returned a team of "
                   << fwd_actual << "; refusing rather than running on one thread");
        }
    }
#endif

    for (int index = 0; !fwd_parallel && index < aMat.NonZeroCount; ++index) {
        int i = aMat.row_index[index];
        int j = aMat.column_index[index];
        mpf_class value = aMat.sp_ele[index];
        // rMessage("i=" << i << "  j=" << j);
        if (i == j) {
            xVec.ele[i] *= value;
        } else {
            xVec.ele[j] -= value * xVec.ele[i];
        }
    }
    if (prof) {
        const double now = Time::rGetUseTime();
        t_fwd = now - t_mark;
        t_mark = now;
    }
    // BACKWARD pass. Descending index order walks rows descending; within row i the
    // off-diagonals gather into the SAME x[i], which is what makes this pass a reduction and
    // the forward pass not one. Threading it therefore changes the last digits, so it is
    // opt-in: SDPA_SOLVE_BACKWARD defaults to serial and this loop is what runs.
#ifdef _OPENMP
    // BACKWARD: no team condition. Once enabled it must run the FIXED-CHUNK algorithm even on
    // one thread, or the answer depends on the team size -- which is exactly the property the
    // fixed chunk count exists to remove. A one-member region costs a fork/join and returns
    // the same bits as any other team size.
    if (eligible && cfg.bwd_mode != SOLVE_SERIAL && (size_ok || cfg.bwd_mode == SOLVE_PARALLEL)) {
        bwd_parallel = solve_backward_parallel(xVec, aMat, rowstart, n, team < 1 ? 1 : team, prec,
                                               (int)cfg.chunks, (int)cfg.min_chunk, &bwd_actual);
        if (bwd_actual < 2 && cfg.bwd_mode == SOLVE_PARALLEL) {
            rError("SDPA_SOLVE_BACKWARD=parallel was requested but the runtime returned a team "
                   "of " << bwd_actual << "; refusing rather than running on one thread");
        }
    }
#endif

    for (int index = aMat.NonZeroCount - 1; !bwd_parallel && index >= 0; --index) {
        int i = aMat.row_index[index];
        int j = aMat.column_index[index];
        mpf_class value = aMat.sp_ele[index];
        value = aMat.sp_ele[index];
        // rMessage("i=" << i << "  j=" << j);
        if (i == j) {
            xVec.ele[i] *= value;
        } else {
            xVec.ele[i] -= value * xVec.ele[j];
        }
    }
    if (prof) {
        t_bwd = Time::rGetUseTime() - t_mark;
        // par_* now means "ran on an actual team of two or more", not merely "took the
        // threaded route". A run that contracted to one thread still produces the right
        // answer, but counting it as parallel would let a CI non-vacuity check pass on a
        // single-threaded run.
        solve_profile_record(t_idx, t_copy, t_fwd - t_idx, t_bwd, fwd_parallel && fwd_actual >= 2,
                             bwd_parallel && bwd_actual >= 2, (long long)aMat.nRow,
                             (long long)aMat.NonZeroCount);
    }

    if (cfg.log) {
        // Once per process. Per call this prints twice per iteration and buries the decision
        // it exists to expose.
        static bool logged = false;
        if (!logged) {
            logged = true;
            // Requested AND actual. Labelling the route from the request alone reported
            // "PARALLEL team=8" for a run the runtime gave one thread.
            rMessage("solve: forward " << (fwd_parallel && fwd_actual >= 2 ? "PARALLEL" : "serial")
                                       << " (team " << fwd_actual << " of " << team
                                       << " requested), backward "
                                       << (bwd_parallel ? "FIXED-CHUNK (answer differs from the"
                                                          " serial pass in the last digits)"
                                                        : "serial")
                                       << " (team " << bwd_actual << ")"
                                       << " :: n=" << n << " nnz=" << aMat.NonZeroCount
                                       << " mean_row=" << (n > 0 ? (aMat.NonZeroCount - n) / n : 0)
                                       << " team=" << team << " max_team=" << cfg.max_team
                                       << " chunks=" << cfg.chunks
                                       << " reason=" << why);
        }
    }

    // The oracle. OUTSIDE the profile guard and timed by neither, because emitting it costs
    // O(n) string conversions and folding that into a pass time would corrupt the very
    // measurement the phases are judged by -- which is how the first Make bMat timing run was
    // lost to a 1.95 GB/solve dump running inside the timer.
    emit_solve_stream(xVec);
#endif
#if TUNEUP
        break;
    case SparseMatrix::DENSE:
        xVec.copyFrom(bVec);
        F77_FUNC(dtrsv, DTRSV)("Lower", "NoTranspose", "NonUnit", &aMat.nRow, aMat.de_ele, &aMat.nCol, xVec.ele, &IONE);
        F77_FUNC(dtrsv, DTRSV)("Lower", "Transpose", "NonUnit", &aMat.nRow, aMat.de_ele, &aMat.nCol, xVec.ele, &IONE);
        return _SUCCESS;
    }
#endif
    return _SUCCESS;
}

bool Lal::multiply(DenseMatrix &retMat, DenseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || aMat.nCol != bMat.nRow || bMat.nCol != retMat.nCol || retMat.type != aMat.type || retMat.type != bMat.type) {
        rError("multiply :: different matrix size");
    }
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        if (scalar == NULL) {
            scalar = &MONE;
            // attension::scalar is loval variable.
        }
        Rgemm("NoTranspose", "NoTranspose", retMat.nRow, retMat.nCol, aMat.nCol, *scalar, aMat.de_ele, aMat.nRow, bMat.de_ele, bMat.nRow, 0.0, retMat.de_ele, retMat.nRow);

        break;
    case DenseMatrix::COMPLETION:
        rError("DenseMatrix:: no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::multiply(DenseMatrix &retMat, SparseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || aMat.nCol != bMat.nRow || bMat.nCol != retMat.nCol) {
        rError("multiply :: different matrix size");
    }
    retMat.setZero();
    switch (aMat.type) {
    case SparseMatrix::SPARSE:
        if (retMat.type != DenseMatrix::DENSE || bMat.type != DenseMatrix::DENSE) {
            rError("multiply :: different matrix type");
        }
        if (scalar == NULL) {
            for (int index = 0; index < aMat.NonZeroCount; ++index) {
                int i = aMat.row_index[index];
                int j = aMat.column_index[index];
                mpf_class value = aMat.sp_ele[index];
                if (i != j) {
#define MULTIPLY_NON_ATLAS 0
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[i + retMat.nRow * t] += value * bMat.de_ele[j + bMat.nRow * t];
                        retMat.de_ele[j + retMat.nRow * t] += value * bMat.de_ele[i + bMat.nRow * t];
                    }
#else
                    Raxpy(bMat.nCol, value, bMat.de_ele + j, bMat.nRow, retMat.de_ele + i, retMat.nRow);
                    Raxpy(bMat.nCol, value, bMat.de_ele + i, bMat.nRow, retMat.de_ele + j, retMat.nRow);
#endif
                } else {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[j + retMat.nRow * t] += value * bMat.de_ele[j + bMat.nRow * t];
                    }
#else
                    Raxpy(bMat.nCol, value, bMat.de_ele + j, bMat.nRow, retMat.de_ele + j, retMat.nRow);
#endif
                }
            }    // end of 'for index'
        } else { // scalar!=NULL
            for (int index = 0; index < aMat.NonZeroCount; ++index) {
                int i = aMat.row_index[index];
                int j = aMat.column_index[index];
                mpf_class value = aMat.sp_ele[index] * (*scalar);
                if (i != j) {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[i + retMat.nRow * t] += value * bMat.de_ele[j + bMat.nRow * t];
                        retMat.de_ele[j + retMat.nRow * t] += value * bMat.de_ele[i + bMat.nRow * t];
                    }
#else
                    Raxpy(bMat.nCol, value, bMat.de_ele + j, bMat.nRow, retMat.de_ele + i, retMat.nRow);
                    Raxpy(bMat.nCol, value, bMat.de_ele + i, bMat.nRow, retMat.de_ele + j, retMat.nRow);
#endif
                } else {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[j + retMat.nRow * t] += value * bMat.de_ele[j + bMat.nRow * t];
                    }
#else
                    Raxpy(bMat.nCol, value, bMat.de_ele + j, bMat.nRow, retMat.de_ele + j, retMat.nRow);
#endif
                }
            } // end of 'for index'
        }     // end of 'if (scalar==NULL)
        break;
    case SparseMatrix::DENSE:
        if (retMat.type != DenseMatrix::DENSE || bMat.type != DenseMatrix::DENSE) {
            rError("multiply :: different matrix type");
        }
        if (scalar == NULL) {
            scalar = &MONE;
            // attension:: scalar is local variable.
        }
        Rgemm("NoTranspose", "NoTranspose", retMat.nRow, retMat.nCol, aMat.nCol, *scalar, aMat.de_ele, aMat.nRow, bMat.de_ele, bMat.nRow, 0.0, retMat.de_ele, retMat.nRow);
        break;

    } // end of switch

    return _SUCCESS;
}

bool Lal::multiply(DenseMatrix &retMat, DenseMatrix &aMat, SparseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || aMat.nCol != bMat.nRow || bMat.nCol != retMat.nCol) {
        rError("multiply :: different matrix size");
    }
    retMat.setZero();
    switch (bMat.type) {
    case SparseMatrix::SPARSE:
        // rMessage("Here will be faster by atlas");
        if (retMat.type != DenseMatrix::DENSE || aMat.type != DenseMatrix::DENSE) {
            rError("multiply :: different matrix type");
        }
        if (scalar == NULL) {
            for (int index = 0; index < bMat.NonZeroCount; ++index) {
                int i = bMat.row_index[index];
                int j = bMat.column_index[index];
                mpf_class value = bMat.sp_ele[index];
                if (i != j) {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[t + retMat.nRow * j] += value * aMat.de_ele[t + aMat.nRow * i];
                        retMat.de_ele[t + retMat.nRow * i] += value * aMat.de_ele[t + aMat.nRow * j];
                    }
#else
                    Raxpy(bMat.nCol, value, &aMat.de_ele[aMat.nRow * j], 1, &retMat.de_ele[retMat.nRow * i], 1);
                    Raxpy(bMat.nCol, value, &aMat.de_ele[aMat.nRow * i], 1, &retMat.de_ele[retMat.nRow * j], 1);
#endif
                } else {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[t + retMat.nRow * j] += value * aMat.de_ele[t + aMat.nRow * j];
                    }
#else
                    Raxpy(bMat.nCol, value, &aMat.de_ele[aMat.nRow * j], 1, &retMat.de_ele[retMat.nRow * j], 1);
#endif
                }
            }    // end of 'for index'
        } else { // scalar!=NULL
            for (int index = 0; index < bMat.NonZeroCount; ++index) {
                int i = bMat.row_index[index];
                int j = bMat.column_index[index];
                mpf_class value = bMat.sp_ele[index] * (*scalar);
                if (i != j) {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[t + retMat.nCol * j] += value * aMat.de_ele[t + bMat.nCol * i];
                        retMat.de_ele[t + retMat.nCol * i] += value * aMat.de_ele[t + bMat.nCol * j];
                    }
#else
                    Raxpy(bMat.nCol, value, aMat.de_ele + (aMat.nRow * j), 1, retMat.de_ele + (retMat.nRow * i), 1);
                    Raxpy(bMat.nCol, value, aMat.de_ele + (aMat.nRow * i), 1, retMat.de_ele + (retMat.nRow * j), 1);
#endif
                } else {
#if MULTIPLY_NON_ATLAS
                    for (int t = 0; t < bMat.nCol; ++t) {
                        retMat.de_ele[t + retMat.nCol * j] += value * aMat.de_ele[t + aMat.nCol * j];
                    }
#else
                    Raxpy(bMat.nCol, value, aMat.de_ele + (aMat.nRow * j), 1, retMat.de_ele + (retMat.nRow * j), 1);
#endif
                }
            } // end of 'for index'
        }     // end of 'if (scalar==NULL)
        break;
    case SparseMatrix::DENSE:
        if (retMat.type != DenseMatrix::DENSE || aMat.type != DenseMatrix::DENSE) {
            rError("multiply :: different matrix type");
        }
        if (scalar == NULL) {
            scalar = &MONE;
            // attension: scalar is local variable.
        }
        Rgemm("NoTranspose", "NoTranspose", retMat.nRow, retMat.nCol, aMat.nCol, *scalar, aMat.de_ele, aMat.nRow, bMat.de_ele, bMat.nRow, 0.0, retMat.de_ele, retMat.nRow);
        break;
    } // end of switch

    return _SUCCESS;
}

bool Lal::multiply(DenseMatrix &retMat, DenseMatrix &aMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.type != aMat.type) {
        rError("multiply :: different matrix size");
    }
    if (scalar == NULL) {
        scalar = &MONE;
    }
    int length;
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        length = retMat.nRow * retMat.nCol;
        Rcopy(length, aMat.de_ele, 1, retMat.de_ele, 1);
        Rscal(length, *scalar, retMat.de_ele, 1);
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::multiply(Vector &retVec, Vector &aVec, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retVec.nDim != aVec.nDim) {
        rError("multiply :: different vector size");
    }
    if (scalar == NULL) {
        scalar = &MONE;
    }
    Rcopy(retVec.nDim, aVec.ele, 1, retVec.ele, 1);
    Rscal(retVec.nDim, *scalar, retVec.ele, 1);
    return _SUCCESS;
}

bool Lal::multiply(BlockVector &retVec, BlockVector &aVec, mpf_class *scalar) {
    if (retVec.nBlock != aVec.nBlock) {
        rError("multiply:: different memory size");
    }
    bool total_judge = _SUCCESS;
    for (int l = 0; l < aVec.nBlock; ++l) {
        bool judge = multiply(retVec.ele[l], aVec.ele[l], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }
    return total_judge;
}

bool Lal::multiply(Vector &retVec, DenseMatrix &aMat, Vector &bVec, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retVec.nDim != aMat.nRow || aMat.nCol != bVec.nDim || bVec.nDim != retVec.nDim) {
        rError("multiply :: different matrix size");
    }
    switch (aMat.type) {
    case DenseMatrix::DENSE:
        if (scalar == NULL) {
            scalar = &MONE;
        }
        Rgemv("NoTranspose", aMat.nRow, aMat.nCol, *scalar, aMat.de_ele, aMat.nRow, bVec.ele, 1, 0.0, retVec.ele, 1);
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::tran_multiply(DenseMatrix &retMat, DenseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nCol || aMat.nRow != bMat.nRow || bMat.nCol != retMat.nCol || retMat.type != aMat.type || retMat.type != bMat.type) {
        rError("multiply :: different matrix size");
    }
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        if (scalar == NULL) {
            scalar = &MONE;
            // scalar is local variable
        }
        // The Point is the first argument is "Transpose".
        Rgemm("Transpose", "NoTranspose", retMat.nRow, retMat.nCol, aMat.nCol, *scalar, aMat.de_ele, aMat.nCol, bMat.de_ele, bMat.nRow, 0.0, retMat.de_ele, retMat.nRow);
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }

    return _SUCCESS;
}

bool Lal::multiply_tran(DenseMatrix &retMat, DenseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || aMat.nCol != bMat.nCol || bMat.nRow != retMat.nRow || retMat.type != aMat.type || retMat.type != bMat.type) {
        rError("multiply :: different matrix size");
    }
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        if (scalar == NULL) {
            scalar = &MONE;
        }
        // The Point is the first argument is "NoTranspose".
        Rgemm("NoTranspose", "Transpose", retMat.nRow, retMat.nCol, aMat.nCol, *scalar, aMat.de_ele, aMat.nRow, bMat.de_ele, bMat.nCol, 0.0, retMat.de_ele, retMat.nRow);
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::plus(Vector &retVec, Vector &aVec, Vector &bVec, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retVec.nDim != aVec.nDim || aVec.nDim != bVec.nDim) {
        rError("plus :: different matrix size");
    }
    if (scalar == NULL) {
        scalar = &MONE;
    }
    if (retVec.ele != aVec.ele) {
        Rcopy(retVec.nDim, aVec.ele, 1, retVec.ele, 1);
    }
    Raxpy(retVec.nDim, *scalar, bVec.ele, 1, retVec.ele, 1);
    return _SUCCESS;
}

bool Lal::plus(DenseMatrix &retMat, DenseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.nRow != bMat.nRow || retMat.nCol != bMat.nCol || retMat.type != aMat.type || retMat.type != bMat.type) {
        rError("plus :: different matrix size");
    }
    if (scalar == NULL) {
        scalar = &MONE;
    }
    int length;
    switch (retMat.type) {
    case DenseMatrix::DENSE:
        length = retMat.nRow * retMat.nCol;
        if (retMat.de_ele != aMat.de_ele) {
            Rcopy(length, aMat.de_ele, 1, retMat.de_ele, 1);
        }
        Raxpy(length, *scalar, bMat.de_ele, 1, retMat.de_ele, 1);
        break;
    case DenseMatrix::COMPLETION:
        rError("no support for COMPLETION");
        break;
    }
    return _SUCCESS;
}

bool Lal::plus(DenseMatrix &retMat, SparseMatrix &aMat, DenseMatrix &bMat, mpf_class *scalar) {
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.nRow != bMat.nRow || retMat.nCol != bMat.nCol) {
        rError("plus :: different matrix size");
    }
    // ret = (*scalar) * b
    if (multiply(retMat, bMat, scalar) == FAILURE) {
        return FAILURE;
    }
    int length;
    // ret += a
    int shou, amari;
    switch (aMat.type) {
    case SparseMatrix::SPARSE:
        if (retMat.type != DenseMatrix::DENSE || bMat.type != DenseMatrix::DENSE) {
            rError("plus :: different matrix type");
        }
#if 0
    for (int index=0; index<aMat.NonZeroCount; ++index) {
      int        i = aMat.row_index   [index];
      int        j = aMat.column_index[index];
      mpf_class value = aMat.sp_ele      [index];
      if (i!=j) {
	retMat.de_ele[i+retMat.nCol*j] += value;
	retMat.de_ele[j+retMat.nCol*i] += value;
      } else {
	retMat.de_ele[i+retMat.nCol*i] += value;
      }
    } // end of 'for index'
#else
        shou = aMat.NonZeroCount / 4;
        amari = aMat.NonZeroCount % 4;
        for (int index = 0; index < amari; ++index) {
            int i = aMat.row_index[index];
            int j = aMat.column_index[index];
            mpf_class value = aMat.sp_ele[index];
            if (i != j) {
                retMat.de_ele[i + retMat.nCol * j] += value;
                retMat.de_ele[j + retMat.nCol * i] += value;
            } else {
                retMat.de_ele[i + retMat.nCol * i] += value;
            }
        } // end of 'for index'
        for (int index = amari, counter = 0; counter < shou; ++counter, index += 4) {
            int i1 = aMat.row_index[index];
            int j1 = aMat.column_index[index];
            mpf_class value1 = aMat.sp_ele[index];
            if (i1 != j1) {
                retMat.de_ele[i1 + retMat.nCol * j1] += value1;
                retMat.de_ele[j1 + retMat.nCol * i1] += value1;
            } else {
                retMat.de_ele[i1 + retMat.nCol * i1] += value1;
            }
            int i2 = aMat.row_index[index + 1];
            int j2 = aMat.column_index[index + 1];
            mpf_class value2 = aMat.sp_ele[index + 1];
            if (i2 != j2) {
                retMat.de_ele[i2 + retMat.nCol * j2] += value2;
                retMat.de_ele[j2 + retMat.nCol * i2] += value2;
            } else {
                retMat.de_ele[i2 + retMat.nCol * i2] += value2;
            }
            int i3 = aMat.row_index[index + 2];
            int j3 = aMat.column_index[index + 2];
            mpf_class value3 = aMat.sp_ele[index + 2];
            if (i3 != j3) {
                retMat.de_ele[i3 + retMat.nCol * j3] += value3;
                retMat.de_ele[j3 + retMat.nCol * i3] += value3;
            } else {
                retMat.de_ele[i3 + retMat.nCol * i3] += value3;
            }
            int i4 = aMat.row_index[index + 3];
            int j4 = aMat.column_index[index + 3];
            mpf_class value4 = aMat.sp_ele[index + 3];
            if (i4 != j4) {
                retMat.de_ele[i4 + retMat.nCol * j4] += value4;
                retMat.de_ele[j4 + retMat.nCol * i4] += value4;
            } else {
                retMat.de_ele[i4 + retMat.nCol * i4] += value4;
            }
        } // end of 'for index'
#endif
        break;
    case SparseMatrix::DENSE:
        if (retMat.type != DenseMatrix::DENSE || bMat.type != DenseMatrix::DENSE) {
            rError("plus :: different matrix type");
        }
        length = retMat.nRow * retMat.nCol;
        Raxpy(length, 1.0, aMat.de_ele, 1, retMat.de_ele, 1);
        break;
    } // end of switch
    return _SUCCESS;
}

bool Lal::plus(DenseMatrix &retMat, DenseMatrix &aMat, SparseMatrix &bMat, mpf_class *scalar) {
    mpf_class MONE = 1.0;
    if (retMat.nRow != aMat.nRow || retMat.nCol != aMat.nCol || retMat.nRow != bMat.nRow || retMat.nCol != bMat.nCol) {
        rError("plus :: different matrix size");
    }
    // ret = a
    if (retMat.copyFrom(aMat) == FAILURE) {
        return FAILURE;
    }
    if (scalar == NULL) {
        scalar = &MONE;
    }
    int length, shou, amari;
    // ret += (*scalar) * b
    switch (bMat.type) {
    case SparseMatrix::SPARSE:
        if (retMat.type != DenseMatrix::DENSE || aMat.type != DenseMatrix::DENSE) {
            rError("plus :: different matrix type");
        }
#if 0
    for (int index=0; index<bMat.NonZeroCount; ++index) {
      int        i = bMat.row_index   [index];
      int        j = bMat.column_index[index];
      mpf_class value = bMat.sp_ele      [index] * (*scalar);
      if (i!=j) {
	retMat.de_ele[i+retMat.nCol*j] += value;
	retMat.de_ele[j+retMat.nCol*i] += value;
      } else {
	retMat.de_ele[i+retMat.nCol*i] += value;
      }
    } // end of 'for index'
#else
        shou = bMat.NonZeroCount / 4;
        amari = bMat.NonZeroCount % 4;
        for (int index = 0; index < amari; ++index) {
            int i = bMat.row_index[index];
            int j = bMat.column_index[index];
            mpf_class value = bMat.sp_ele[index] * (*scalar);
            if (i != j) {
                retMat.de_ele[i + retMat.nCol * j] += value;
                retMat.de_ele[j + retMat.nCol * i] += value;
            } else {
                retMat.de_ele[i + retMat.nCol * i] += value;
            }
        } // end of 'for index'
        for (int index = amari, counter = 0; counter < shou; ++counter, index += 4) {
            int i1 = bMat.row_index[index];
            int j1 = bMat.column_index[index];
            mpf_class value1 = bMat.sp_ele[index] * (*scalar);
            if (i1 != j1) {
                retMat.de_ele[i1 + retMat.nCol * j1] += value1;
                retMat.de_ele[j1 + retMat.nCol * i1] += value1;
            } else {
                retMat.de_ele[i1 + retMat.nCol * i1] += value1;
            }
            int i2 = bMat.row_index[index + 1];
            int j2 = bMat.column_index[index + 1];
            mpf_class value2 = bMat.sp_ele[index + 1] * (*scalar);
            if (i2 != j2) {
                retMat.de_ele[i2 + retMat.nCol * j2] += value2;
                retMat.de_ele[j2 + retMat.nCol * i2] += value2;
            } else {
                retMat.de_ele[i2 + retMat.nCol * i2] += value2;
            }
            int i3 = bMat.row_index[index + 2];
            int j3 = bMat.column_index[index + 2];
            mpf_class value3 = bMat.sp_ele[index + 2] * (*scalar);
            if (i3 != j3) {
                retMat.de_ele[i3 + retMat.nCol * j3] += value3;
                retMat.de_ele[j3 + retMat.nCol * i3] += value3;
            } else {
                retMat.de_ele[i3 + retMat.nCol * i3] += value3;
            }
            int i4 = bMat.row_index[index + 3];
            int j4 = bMat.column_index[index + 3];
            mpf_class value4 = bMat.sp_ele[index + 3] * (*scalar);
            if (i4 != j4) {
                retMat.de_ele[i4 + retMat.nCol * j4] += value4;
                retMat.de_ele[j4 + retMat.nCol * i4] += value4;
            } else {
                retMat.de_ele[i4 + retMat.nCol * i4] += value4;
            }
        } // end of 'for index'
#endif
        break;
    case SparseMatrix::DENSE:
        if (retMat.type != DenseMatrix::DENSE || aMat.type != DenseMatrix::DENSE) {
            rError("plus :: different matrix type");
        }
        length = retMat.nRow * retMat.nCol;
        Raxpy(length, *scalar, bMat.de_ele, 1, retMat.de_ele, 1);
        break;
    } // end of switch
    return _SUCCESS;
}

bool Lal::plus(BlockVector &retVec, BlockVector &aVec, BlockVector &bVec, mpf_class *scalar) {
    if (retVec.nBlock != aVec.nBlock || retVec.nBlock != bVec.nBlock) {
        rError("plus:: different nBlock size");
    }
    bool total_judge = _SUCCESS;
    for (int l = 0; l < retVec.nBlock; ++l) {
        bool judge = plus(retVec.ele[l], aVec.ele[l], bVec.ele[l], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }
    return total_judge;
}

// ret = a '*' (*scalar)
bool Lal::let(Vector &retVec, const char eq, Vector &aVec, const char op, mpf_class *scalar) {
    switch (op) {
    case '*':
        return multiply(retVec, aVec, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '*' (*scalar)
bool Lal::let(BlockVector &retVec, const char eq, BlockVector &aVec, const char op, mpf_class *scalar) {
    switch (op) {
    case '*':
        return multiply(retVec, aVec, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' b*(*scalar)
bool Lal::let(Vector &retVec, const char eq, Vector &aVec, const char op, Vector &bVec, mpf_class *scalar) {
    mpf_class MMONE = -1.0;
    mpf_class minus_scalar;
    switch (op) {
    case '+':
        return plus(retVec, aVec, bVec, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retVec, aVec, bVec, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' '*' 't' 'T' b*(*scalar)
bool Lal::let(DenseMatrix &retMat, const char eq, DenseMatrix &aMat, const char op, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class MMONE = -1.0;
    mpf_class minus_scalar;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '*':
        return multiply(retMat, aMat, bMat, scalar);
        break;
    case 't':
        // ret = aMat**T * bMat
        return tran_multiply(retMat, aMat, bMat, scalar);
        break;
    case 'T':
        // ret = aMat * bMat**T
        return multiply_tran(retMat, aMat, bMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' '*' b*(*scalar)
bool Lal::let(DenseMatrix &retMat, const char eq, SparseMatrix &aMat, const char op, DenseMatrix &bMat, mpf_class *scalar) {
    mpf_class minus_scalar;
    mpf_class MMONE = -1.0;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '*':
        return multiply(retMat, aMat, bMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' '*' b*(*scalar)
bool Lal::let(DenseMatrix &retMat, const char eq, DenseMatrix &aMat, const char op, SparseMatrix &bMat, mpf_class *scalar) {
    mpf_class minus_scalar;
    mpf_class MMONE = -1.0;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '*':
        return multiply(retMat, aMat, bMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = aMat '*' '/' bVec
bool Lal::let(Vector &rVec, const char eq, DenseMatrix &aMat, const char op, Vector &bVec) {
    switch (op) {
    case '*':
        return multiply(rVec, aMat, bVec, NULL);
        break;
    case '/':
        // ret = aMat^{-1} * bVec;
        // aMat is positive definite
        // and already colesky factorized.
        return solveSystems(rVec, aMat, bVec);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// nakata 2004/12/01
// ret = aMat '*' '/' bVec
bool Lal::let(Vector &rVec, const char eq, SparseMatrix &aMat, const char op, Vector &bVec) {
    switch (op) {
    case '/':
        // ret = aMat^{-1} * bVec;
        // aMat is positive definite
        // and already colesky factorized.
        return solveSystems(rVec, aMat, bVec);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, Vector &aVec, const char op, Vector &bVec) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aVec, bVec);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, DenseMatrix &aMat, const char op, DenseMatrix &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aMat, bMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, DenseMatrix &aMat, const char op, SparseMatrix &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, bMat, aMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, SparseMatrix &aMat, const char op, DenseMatrix &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aMat, bMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, BlockVector &aVec, const char op, BlockVector &bVec) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aVec, bVec);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

/////////////////////////////////////////////////////////////////////////

bool Lal::getInnerProduct(mpf_class &ret, DenseLinearSpace &aMat, DenseLinearSpace &bMat) {
    bool total_judge = _SUCCESS;
    ret = 0.0;
    mpf_class tmp_ret;

    // for SDP
    if (aMat.SDP_nBlock != bMat.SDP_nBlock) {
        rError("getInnerProduct:: different memory size");
    }
    for (int l = 0; l < aMat.SDP_nBlock; ++l) {
        bool judge = Lal::getInnerProduct(tmp_ret, aMat.SDP_block[l], bMat.SDP_block[l]);
        ret += tmp_ret;
        if (judge == FAILURE) {
            rMessage(" something failed");
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  if (aMat.SOCP_nBlock != bMat.SOCP_nBlock) {
    rError("getInnerProduct:: different memory size");
  }
  for (int l=0; l<aMat.SOCP_nBlock; ++l) {
    bool judge = Lal::getInnerProduct(tmp_ret,aMat.SOCP_block[l],bMat.SOCP_block[l]);
    ret += tmp_ret;
    if (judge == FAILURE) {
      rMessage(" something failed");
      total_judge = FAILURE;
    }
  }
#endif

    // for LP
    if (aMat.LP_nBlock != bMat.LP_nBlock) {
        rError("getInnerProduct:: different memory size");
    }
    for (int l = 0; l < aMat.LP_nBlock; ++l) {
        tmp_ret = aMat.LP_block[l];
        tmp_ret *= bMat.LP_block[l];
        ret += tmp_ret;
    }

    return total_judge;
}

bool Lal::getInnerProduct(mpf_class &ret, SparseLinearSpace &aMat, DenseLinearSpace &bMat) {
    bool total_judge = _SUCCESS;
    ret = 0.0;
    mpf_class tmp_ret;

    // for SDP
    for (int l = 0; l < aMat.SDP_sp_nBlock; ++l) {
        int index = aMat.SDP_sp_index[l];
        bool judge = Lal::getInnerProduct(tmp_ret, aMat.SDP_sp_block[l], bMat.SDP_block[index]);
        ret += tmp_ret;
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  for (int l=0; l<aMat.SOCP_sp_nBlock; ++l) {
    int index = aMat.SOCP_sp_index[l];
    bool judge = Lal::getInnerProduct(tmp_ret,aMat.SOCP_sp_block[l],bMat.SOCP_block[index]);
    ret += tmp_ret;
    if (judge == FAILURE) {
      total_judge = FAILURE;
    }
  }
#endif

    for (int l = 0; l < aMat.LP_sp_nBlock; ++l) {
        int index = aMat.LP_sp_index[l];
        tmp_ret = aMat.LP_sp_block[l];
        tmp_ret *= bMat.LP_block[index];
        ret += tmp_ret;
    }

    return total_judge;
}

bool Lal::multiply(DenseLinearSpace &retMat, DenseLinearSpace &aMat, mpf_class *scalar) {
    bool total_judge = _SUCCESS;

    // for SDP
    if (retMat.SDP_nBlock != aMat.SDP_nBlock) {
        rError("multiply:: different memory size");
    }
    for (int l = 0; l < aMat.SDP_nBlock; ++l) {
        bool judge = Lal::multiply(retMat.SDP_block[l], aMat.SDP_block[l], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  if (retMat.SOCP_nBlock!=aMat.SOCP_nBlock) {
    rError("multiply:: different memory size");
  }
  for (int l=0; l<aMat.SOCP_nBlock; ++l) {
    bool judge = Lal::multiply(retMat.SOCP_block[l],aMat.SOCP_block[l],scalar);
    if (judge == FAILURE) {
      total_judge = FAILURE;
    }
  }
#endif

    // fo LP
    if (retMat.LP_nBlock != aMat.LP_nBlock) {
        rError("multiply:: different memory size");
    }
    /* scalar defaults to NULL in the declaration (sdpa_linear.h), and this branch
       dereferenced it unconditionally. Every in-tree LP caller happens to pass one, so
       the defect is latent rather than live -- but the API contract said otherwise.
       NULL means "no scaling", i.e. a plain copy, which is what the dense paths do. */
    for (int l = 0; l < aMat.LP_nBlock; ++l) {
        retMat.LP_block[l] = (scalar == NULL) ? aMat.LP_block[l]
                                              : aMat.LP_block[l] * (*scalar);
    }

    return total_judge;
}

bool Lal::plus(DenseLinearSpace &retMat, DenseLinearSpace &aMat, DenseLinearSpace &bMat, mpf_class *scalar) {
    bool total_judge = _SUCCESS;

    // for SDP
    if (retMat.SDP_nBlock != aMat.SDP_nBlock || retMat.SDP_nBlock != bMat.SDP_nBlock) {
        rError("plus:: different nBlock size");
    }
    for (int l = 0; l < retMat.SDP_nBlock; ++l) {
        bool judge = Lal::plus(retMat.SDP_block[l], aMat.SDP_block[l], bMat.SDP_block[l], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  if (retMat.SOCP_nBlock!=aMat.SOCP_nBlock 
      || retMat.SOCP_nBlock!=bMat.SOCP_nBlock) {
    rError("plus:: different nBlock size");
  }
  for (int l=0; l<retMat.SOCP_nBlock; ++l) {
    bool judge = Lal::plus(retMat.SOCP_block[l],aMat.SOCP_block[l],
			   bMat.SOCP_block[l],scalar);
    if (judge == FAILURE) {
      total_judge = FAILURE;
    }
  }
#endif

    // for LP
    if (retMat.LP_nBlock != aMat.LP_nBlock || retMat.LP_nBlock != bMat.LP_nBlock) {
        rError("plus:: different nBlock size");
    }
    for (int l = 0; l < retMat.LP_nBlock; ++l) {
        if (scalar == NULL) {
            retMat.LP_block[l] = aMat.LP_block[l] + bMat.LP_block[l];
        } else {
            retMat.LP_block[l] = aMat.LP_block[l] + bMat.LP_block[l] * (*scalar);
        }
    }

    return total_judge;
}

// CAUTION!!! We don't initialize retMat to zero matrix for efficiently.
bool Lal::plus(DenseLinearSpace &retMat, SparseLinearSpace &aMat, DenseLinearSpace &bMat, mpf_class *scalar) {
    bool total_judge = _SUCCESS;

    // for SDP
    for (int l = 0; l < aMat.SDP_sp_nBlock; ++l) {
        int index = aMat.SDP_sp_index[l];
        bool judge = Lal::plus(retMat.SDP_block[index], aMat.SDP_sp_block[l], bMat.SDP_block[index], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  for (int l=0; l<aMat.SOCP_sp_nBlock; ++l) {
    int index = aMat.SOCP_sp_index[l];
    bool judge = Lal::plus(retMat.SOCP_block[index],aMat.SOCP_sp_block[l],
			   bMat.SOCP_block[index],scalar);
    if (judge == FAILURE) {
      total_judge = FAILURE;
    }
  }
#endif

    // for LP
    for (int l = 0; l < aMat.LP_sp_nBlock; ++l) {
        int index = aMat.LP_sp_index[l];
        if (scalar == NULL) {
            retMat.LP_block[index] = aMat.LP_sp_block[l] + bMat.LP_block[index];
        } else {
            retMat.LP_block[index] = aMat.LP_sp_block[l] + bMat.LP_block[index] * (*scalar);
        }
    }

    return total_judge;
}

// CAUTION!!! We don't initialize retMat to zero matrix for efficiently.
bool Lal::plus(DenseLinearSpace &retMat, DenseLinearSpace &aMat, SparseLinearSpace &bMat, mpf_class *scalar) {
    bool total_judge = _SUCCESS;

    // for SDP
    for (int l = 0; l < bMat.SDP_sp_nBlock; ++l) {
        int index = bMat.SDP_sp_index[l];
        bool judge = Lal::plus(retMat.SDP_block[index], aMat.SDP_block[index], bMat.SDP_sp_block[l], scalar);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }

    // for SOCP
#if 0
  for (int l=0; l<bMat.SOCP_sp_nBlock; ++l) {
    int index = bMat.SOCP_sp_index[l];
    bool judge = Lal::plus(retMat.SOCP_block[index],aMat.SOCP_block[index],
			   bMat.SOCP_sp_block[l],scalar);
    if (judge == FAILURE) {
      total_judge = FAILURE;
    }
  }
#endif

    // for LP
    for (int l = 0; l < bMat.LP_sp_nBlock; ++l) {
        int index = bMat.LP_sp_index[l];
        if (scalar == NULL) {
            retMat.LP_block[index] = aMat.LP_block[index] + bMat.LP_sp_block[l];
        } else {
            retMat.LP_block[index] = aMat.LP_block[index] + bMat.LP_sp_block[l] * (*scalar);
        }
    }

    return total_judge;
}

bool Lal::getSymmetrize(DenseLinearSpace &aMat) {
    bool total_judge = _SUCCESS;
    // for SDP
    for (int l = 0; l < aMat.SDP_nBlock; ++l) {
        bool judge = Lal::getSymmetrize(aMat.SDP_block[l]);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }
    return total_judge;
}

bool Lal::getTranspose(DenseLinearSpace &retMat, DenseLinearSpace &aMat) {
    // for SDP
    if (retMat.SDP_nBlock != aMat.SDP_nBlock) {
        rError("getTranspose:: different memory size");
    }
    bool total_judge = _SUCCESS;
    for (int l = 0; l < aMat.SDP_nBlock; ++l) {
        bool judge = Lal::getTranspose(retMat.SDP_block[l], aMat.SDP_block[l]);
        if (judge == FAILURE) {
            total_judge = FAILURE;
        }
    }
    return total_judge;
}

// ret = a '*' (*scalar)
bool Lal::let(DenseLinearSpace &retMat, const char eq, DenseLinearSpace &aMat, const char op, mpf_class *scalar) {
    switch (op) {
    case '*':
        return multiply(retMat, aMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' b*(*scalar)
bool Lal::let(DenseLinearSpace &retMat, const char eq, DenseLinearSpace &aMat, const char op, DenseLinearSpace &bMat, mpf_class *scalar) {
    mpf_class MMONE = -1.0;
    mpf_class minus_scalar;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' b*(*scalar)
bool Lal::let(DenseLinearSpace &retMat, const char eq, SparseLinearSpace &aMat, const char op, DenseLinearSpace &bMat, mpf_class *scalar) {
    mpf_class MMONE = -1.0;
    mpf_class minus_scalar;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '*':
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = a '+' '-' b*(*scalar)
bool Lal::let(DenseLinearSpace &retMat, const char eq, DenseLinearSpace &aMat, const char op, SparseLinearSpace &bMat, mpf_class *scalar) {
    mpf_class minus_scalar;
    mpf_class MMONE = -1.0;
    switch (op) {
    case '+':
        return plus(retMat, aMat, bMat, scalar);
        break;
    case '-':
        if (scalar) {
            minus_scalar = -(*scalar);
            scalar = &minus_scalar;
        } else {
            scalar = &MMONE;
        }
        return plus(retMat, aMat, bMat, scalar);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, DenseLinearSpace &aMat, const char op, DenseLinearSpace &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aMat, bMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, SparseLinearSpace &aMat, const char op, DenseLinearSpace &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, aMat, bMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

// ret = inner_product(a,b) // op = '.'
bool Lal::let(mpf_class &ret, const char eq, DenseLinearSpace &aMat, const char op, SparseLinearSpace &bMat) {
    switch (op) {
    case '.':
        return getInnerProduct(ret, bMat, aMat);
        break;
    default:
        rError("let:: operator error");
        break;
    }
    return FAILURE;
}

} // namespace sdpa
