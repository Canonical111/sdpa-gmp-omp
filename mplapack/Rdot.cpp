/*
 * Copyright (c) 2008-2012
 *	Nakata, Maho
 * 	All rights reserved.
 *
 * $Id: Rdot.cpp,v 1.5 2010/08/07 05:50:10 nakatamaho Exp $
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

#include <mpblas_gmp.h>

mpf_class Rdot_serial(mplapackint n, mpf_class *dx, mplapackint incx, mpf_class *dy, mplapackint incy);

/* 2026-08-05: dispatch to Rdot_serial. This used to read `if (0) { Rdot_serial } else
   { Rdot_omp }`, i.e. Rdot_serial was dead code behind a constant-false branch and every dot
   product went to Rdot_omp.

   Rdot_omp is NOT parallel and never has been in this fork: every one of its pragmas is
   commented out in the source (`//#pragma omp parallel`, `//#pragma omp for`,
   `//#pragma omp critical`), so the name is a misnomer and the live path was a strictly
   slower serial loop. It carries `mpf_class temp = dx[i]; temp *= dy[i]; local_result +=
   temp;`, which constructs and destroys a GMP temporary -- a malloc and a free -- on EVERY
   element. Rdot_serial does the same arithmetic on raw mpf_t with the two temporaries
   mpf_init'ed once outside the loop, so its inner loop allocates nothing.

   COLD. This is not a speedup to quote. The kernel measures 1.69-1.73x, but Rdot is a
   negligible share of a real solve and the end-to-end effect is about 1%, which is inside
   this project's noise on any machine it has been measured on. It is here because it deletes
   a dead branch and an allocation per element, not because it makes the solver faster.

   Bit-identity is not assumed from the shape of the code -- both accumulate sequentially in
   ascending index order at the same working precision, and Rdot_omp's extra
   `result += local_result` starts from an exact zero, but that is an argument, not evidence.
   The evidence is patches/regress.sh over the standard problem set, run against a golden
   recorded from the parent commit; see patches/tierc_notes/hygiene_batch.md. */
mpf_class Rdot(mplapackint const n, mpf_class *dx, mplapackint const incx, mpf_class *dy, mplapackint const incy) { return Rdot_serial(n, dx, incx, dy, incy); }
