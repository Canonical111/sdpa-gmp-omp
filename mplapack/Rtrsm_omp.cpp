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

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: new file. Column-parallel Rtrsm for the
   single case the solver uses (Left / Lower / NoTranspose); every other case is handed to the
   serial Rtrsm. Ported from sdpa-dd's mplapack/Rtrsm_omp.cpp (B3). See git log. */

/*  PORT NOTES (sdpa-dd -> sdpa-gmp). Three things are NOT copied from dd:

    1. Only the LEFT case is implemented. dd's Rtrsm_omp also handles Right/Lower/Transpose,
       which it split over the m ROWS -- that arrived with dd's B1 (Rpotrf panel) and its gate
       constant was calibrated for dd_real on shapes this fork has not been measured on. This
       fork has no B1, so there is no caller for it here and it is deliberately absent rather
       than carried across unmeasured.

    2. The parallel body is a transcription of THIS fork's Rtrsm.cpp, not of dd's. sdpa-gmp
       carries the A3 `product_scratch` convention: the inner update is three statements
           product_scratch  = b[k][j];
           product_scratch *= a[i][k];
           b[i][j]         -= product_scratch;
       and not dd's single `b = b - b*a` expression. Writing dd's expression here would
       reintroduce the per-flop gmpxx heap temporary A3 removed, and -- because the two forms
       round differently in general -- would not be bit-identical to this fork's own serial
       Rtrsm. Bit-identity against the serial kernel is the ship gate, so the transcription
       must stay a transcription.

    3. product_scratch is declared INSIDE the parallel loop body, not at function scope.
       Rtrsm.cpp keeps one function-scope scratch per kernel (A3's convention, one mpf_init/
       mpf_clear per call instead of per flop); under a pragma that object would be shared and
       every thread would clobber it. Loop-body scope makes it private by construction rather
       than by a private() clause that a later edit can drift away from -- and it costs one
       mpf_init/mpf_clear per COLUMN, not per flop, so A3's point is preserved. It is
       bit-neutral: a freshly constructed mpf_class and a reused one have the same precision
       (this fork never constructs at an explicit precision), and `=` rounds to the
       destination's precision either way.

    Why a separate entry point rather than a pragma inside Rtrsm.cpp:
    Rtrsm is also called from Rpotrf's UPPER path (Rtrsm.cpp's caller mplapack/Rpotrf.cpp:116)
    and from Rpotrf2's recursion (mplapack/Rpotrf2.cpp:119,136), at shapes for which nothing has
    been measured. Threading the generic kernel would thread those too. Exactly one call site
    reaches this file's symbol: Lal::getInvLowTriangularMatrix (sdpa_linear.cpp).

    Why columns:
    for side == "Left" the outermost loop of every branch of Rtrsm runs over the n columns of B,
    and every read and write inside it is b[.. + (j-1)*ldb] for that one j; a is read-only. So
    the columns of B are independent and the arithmetic within a column is untouched by the
    split -- which is what makes the result bit-identical to the serial kernel at any thread
    count. This is NOT true for side == "Right", where the "B*inv(A)" loops update column j from
    column k for every k < j and it is the m ROWS that are independent instead. A column split
    of a Right-side call compiles, converges, and returns wrong answers. This file implements
    Left only and refuses anything else, so that distinction cannot be got wrong by accident. */

#include <mpblas_gmp.h>
#include "mplapack_omp_tuning.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void Rtrsm_omp(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, mpf_class const alpha, mpf_class *a, mplapackint const lda, mpf_class *b, mplapackint const ldb) {
    const mpf_class zero = 0.0;
    //
    //     The only case this file implements: B := alpha*inv(A)*B with A lower triangular,
    //     not transposed, applied from the left. Each test is positive (Mlsame against the
    //     wanted letter), never "not the other one", so a malformed argument string falls
    //     through to the serial kernel instead of being silently treated as this case.
    //
    bool handled = Mlsame_gmp(side, "L") && Mlsame_gmp(uplo, "L") && Mlsame_gmp(transa, "N") && (Mlsame_gmp(diag, "N") || Mlsame_gmp(diag, "U"));
    //
    //     Argument errors and the quick-return/alpha==0 paths are left to Rtrsm so that
    //     Mxerbla reporting lives in exactly one place. nrowa is m because side is "L".
    //
    bool valid = (m > 0) && (n > 0) && (lda >= std::max((mplapackint)1, m)) && (ldb >= std::max((mplapackint)1, m)) && (alpha != zero);
    //
    bool parallel = handled && valid;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    //
    //     Work gate. m*m*n is twice the true mpf multiply-add count of the branch below; the
    //     (double) casts are kept although mplapackint is 64-bit here, so that the expression
    //     is the same one the gmp/qd/dd headers all compare against.
    //
    parallel = parallel && (nthreads > 1) && !omp_in_parallel() && ((double)m * (double)m * (double)n >= MPLAPACK_OMP_TRI_WORK(MPLAPACK_OMP_MIN_TRSM_WORK, nthreads)) && (n >= MPLAPACK_OMP_MIN_TRI_WIDTH);
#else
    /* Without OpenMP there is nothing to gate; everything goes to the serial kernel. */
    parallel = false;
#endif
    if (!parallel) {
        Rtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }
#ifdef _OPENMP
    const mpf_class one = 1.0;
    bool nounit = Mlsame_gmp(diag, "N");
    //
    //           Form  B := alpha*inv( A )*B.   (lower, no transpose)
    //
    //     schedule(dynamic,1), not the default static split: against the identity RHS this
    //     call is given, column j costs O((m-j)^2) because the leading zeros of the identity
    //     are skipped by the `!= zero` test, so a contiguous block split hands the first
    //     thread roughly a third of the total work.
    //
    //     num_threads() is clamped to n so a narrow B never creates idle team members.
    //
    //     `one` and `zero` are read-only inside the region (mpf_cmp does not mutate its
    //     operands); every MUTABLE mpf_class is declared in the body below.
    //
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads < (int)n ? nthreads : (int)n)
    for (mplapackint j = 1; j <= n; j = j + 1) {
        mplapackint i = 0;
        mplapackint k = 0;
        mpf_class product_scratch; // per-thread; see PORT NOTES 3
        if (alpha != one) {
            for (i = 1; i <= m; i = i + 1) {
                b[(i - 1) + (j - 1) * ldb] = alpha * b[(i - 1) + (j - 1) * ldb];
            }
        }
        for (k = 1; k <= m; k = k + 1) {
            if (b[(k - 1) + (j - 1) * ldb] != zero) {
                if (nounit) {
                    b[(k - 1) + (j - 1) * ldb] = b[(k - 1) + (j - 1) * ldb] / a[(k - 1) + (k - 1) * lda];
                }
                for (i = k + 1; i <= m; i = i + 1) {
                    product_scratch = b[(k - 1) + (j - 1) * ldb];
                    product_scratch *= a[(i - 1) + (k - 1) * lda];
                    b[(i - 1) + (j - 1) * ldb] -= product_scratch;
                }
            }
        }
    }
#endif
    //
    //     End of Rtrsm_omp .
    //
}
