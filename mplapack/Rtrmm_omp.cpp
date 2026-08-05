/*
 * Copyright (c) 2008-2021
 *      Nakata, Maho
 *      All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: new file. Column-parallel Rtrmm for the
   single case the solver uses (Left / Lower / Transpose); every other case is handed to the
   serial Rtrmm. Ported from sdpa-dd's mplapack/Rtrmm_omp.cpp (B3). See git log. */

/*  PORT NOTES (sdpa-dd -> sdpa-gmp). What is NOT copied from dd:

    1. The parallel body is a transcription of THIS fork's Rtrmm.cpp (the Left/Lower/Transpose
       branch, Rtrmm.cpp:179-196), not of dd's. Two differences matter:
         - the A3 `product_scratch` convention -- the accumulate is
               product_scratch  = a[k][i];
               product_scratch *= b[k][j];
               temp            += product_scratch;
           and not dd's `temp += a*b`. dd's form would reintroduce the per-flop gmpxx heap
           temporary A3 removed, and would not be bit-identical to this fork's serial Rtrmm.
         - the zero-skip `if (b[k][j] != zero)` on the accumulate, which this fork added to
           Rtrmm.cpp on 2026-08-03 and netlib dtrmm does not have. It must be copied, or the
           parallel kernel and the serial one it must match would skip different terms.
       Rtrmm.cpp and this file have to be kept in step; if one gains an arithmetic change the
       other needs the same one, and the bit-identity gate is what catches the omission.

    2. temp and product_scratch are declared INSIDE the parallel loop body. Rtrmm.cpp declares
       `mpf_class temp` and `mpf_class product_scratch` at FUNCTION scope and shares them across
       all eight (side, uplo, transa) branches; under a pragma both would be shared and every
       thread would clobber them. Loop-body scope makes them private by construction rather than
       by a private() clause that a later edit can drift away from. Cost is one mpf_init/
       mpf_clear pair per COLUMN, not per flop, so A3's point is preserved. Bit-neutral: a
       freshly constructed mpf_class and a reused one carry the same precision (this fork never
       constructs at an explicit precision) and `=` rounds to the destination's precision either
       way. Note that temp must NOT be given an initialiser that the serial kernel does not
       have a counterpart for -- the first thing the i-loop does is `temp = b[i][j]`, so its
       incoming value is dead in both.

    Why a separate entry point rather than a pragma inside Rtrmm.cpp:
    Rtrmm is also called 24 times by Rlarfb (mplapack/Rlarfb.cpp), every one of them with
    side == "Right", on the small k-wide workspace of the tridiagonal reduction. Threading the
    generic kernel would thread those too, with no measurement behind them. Only
    Jal::getInvCholAndInv (sdpa_jordan.cpp) calls this file's symbol.

    Why columns:
    for side == "Left" the outermost loop of every branch of Rtrmm runs over the n columns of B,
    and every read and write inside it is b[.. + (j-1)*ldb] for that one j; a is read-only. So
    the columns of B are independent and the arithmetic within a column is untouched by the
    split -- which is what makes the result bit-identical to the serial kernel at any thread
    count. This is NOT true for side == "Right", where the "B*A" loops read b[..+(k-1)*ldb] from
    other columns and it is the m ROWS that are independent instead. This file implements Left
    only and refuses anything else, so that distinction cannot be got wrong by accident. */

#include <mpblas_gmp.h>
#include "mplapack_omp_tuning.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void Rtrmm_omp(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, mpf_class const alpha, mpf_class *a, mplapackint const lda, mpf_class *b, mplapackint const ldb) {
    const mpf_class zero = 0.0;
    //
    //     The only case this file implements: B := alpha*A**T*B with A lower triangular,
    //     applied from the left. Each test is positive (Mlsame against the wanted letter),
    //     never "not the other one", so a malformed argument string falls through to the
    //     serial kernel instead of being silently treated as this case. "C" is accepted
    //     alongside "T" because Rtrmm itself treats them identically for a real matrix.
    //
    bool handled = Mlsame_gmp(side, "L") && Mlsame_gmp(uplo, "L") && (Mlsame_gmp(transa, "T") || Mlsame_gmp(transa, "C")) && (Mlsame_gmp(diag, "N") || Mlsame_gmp(diag, "U"));
    //
    //     Argument errors and the quick-return/alpha==0 paths are left to Rtrmm so that
    //     Mxerbla reporting lives in exactly one place. nrowa is m because side is "L".
    //
    bool valid = (m > 0) && (n > 0) && (lda >= std::max((mplapackint)1, m)) && (ldb >= std::max((mplapackint)1, m)) && (alpha != zero);
    //
    bool parallel = handled && valid;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    //
    //     Work gate; m*m*n is twice the true mpf multiply-add count of the branch below.
    //
    parallel = parallel && (nthreads > 1) && !omp_in_parallel() && ((double)m * (double)m * (double)n >= MPLAPACK_OMP_TRI_WORK(MPLAPACK_OMP_MIN_TRMM_WORK, nthreads)) && (n >= MPLAPACK_OMP_MIN_TRI_WIDTH);
#else
    /* Without OpenMP there is nothing to gate; everything goes to the serial kernel. */
    parallel = false;
#endif
    if (!parallel) {
        Rtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }
#ifdef _OPENMP
    bool nounit = Mlsame_gmp(diag, "N");
    //
    //           Form  B := alpha*A**T*B.   (lower)
    //
    //     schedule(dynamic,1) for the same reason as Rtrsm_omp: the B this call is given is
    //     inv(L), lower triangular, so the zero-skip makes column j cost O((m-j)^2) and a
    //     contiguous block split is badly unbalanced.
    //
    //     num_threads() is clamped to n so a narrow B never creates idle team members.
    //
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads < (int)n ? nthreads : (int)n)
    for (mplapackint j = 1; j <= n; j = j + 1) {
        mplapackint i = 0;
        mplapackint k = 0;
        mpf_class temp = 0.0;
        mpf_class product_scratch; // per-thread; see PORT NOTES 2
        for (i = 1; i <= m; i = i + 1) {
            temp = b[(i - 1) + (j - 1) * ldb];
            if (nounit) {
                temp = temp * a[(i - 1) + (i - 1) * lda];
            }
            for (k = i + 1; k <= m; k = k + 1) {
                if (b[(k - 1) + (j - 1) * ldb] != zero) {
                    product_scratch = a[(k - 1) + (i - 1) * lda];
                    product_scratch *= b[(k - 1) + (j - 1) * ldb];
                    temp += product_scratch;
                }
            }
            b[(i - 1) + (j - 1) * ldb] = alpha * temp;
        }
    }
#endif
    //
    //     End of Rtrmm_omp .
    //
}
