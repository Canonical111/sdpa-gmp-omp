/*
 * Copyright (c) 2008-2012
 *	Nakata, Maho
 * 	All rights reserved.
 *
 * $Id: Rgemm_TT.cpp,v 1.1 2010/12/28 06:13:53 nakatamaho Exp $
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
/* MODIFIED from upstream (BSD 2-clause; this file is BSD-licensed MPLAPACK, not GPL -- the original copyright notice above is retained and the change is recorded here), 2026-08-03: per-flop gmpxx heap temporary (__gmp_temp = mpf_init2+mpf_clear) removed by accumulating through a function-scope product_scratch scratch. Bit-neutral only while every mpf_class shares one precision (verified: this fork has no explicit-precision construction). If the commented-out private(i,j,l,temp) pragma below is re-enabled, product_scratch MUST be added to the private list or it becomes shared and racy. See git log. */
#include <mpblas_gmp.h>

void Rgemm_TT_omp(mplapackint m, mplapackint n, mplapackint k, mpf_class alpha, mpf_class *A, mplapackint lda, mpf_class *B, mplapackint ldb, mpf_class beta, mpf_class *C, mplapackint ldc) {
    // Form  C := alpha*A'*B' + beta*C.
    mplapackint i, j, l;
    mpf_class temp;
    mpf_class product_scratch; // scratch for the product; keeps the accumulate off the heap
    for (j = 0; j < n; j++) {
        if (beta == 0.0) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
            }
        } else if (beta != 1.0) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] = beta * C[i + j * ldc];
            }
        }
    }
// main loop
#ifdef _OPENMP
//#pragma omp parallel for private(i, j, l, temp)
#endif
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            temp = 0.0;
            for (l = 0; l < k; l++) {
                product_scratch = A[l + i * lda];
                product_scratch *= B[j + l * ldb];
                temp += product_scratch;
            }
            product_scratch = alpha;
            product_scratch *= temp;
            C[i + j * ldc] += product_scratch;
        }
    }
    return;
}
