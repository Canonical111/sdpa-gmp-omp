#ifndef MPLAPACK_OMP_TUNING_H
#define MPLAPACK_OMP_TUNING_H

/* New file added by this fork, 2026-08-04. It lives in the bundled MPLAPACK tree and is
   offered under the same 2-clause BSD terms as the files around it:
   Redistribution and use in source and binary forms, with or without modification, are
   permitted provided that the copyright notice and this list of conditions are retained. No upstream file corresponds
   to it, so there is nothing to state changes against; the notices below record its history. */

/* MODIFIED from upstream (BSD 2-clause; this file is BSD-licensed, not GPL), 2026-08-05: added the Left-side triangular-kernel
   work gates used by Rtrsm_omp/Rtrmm_omp (the B3 port from sdpa-dd). See git log. */

/* Bump when macros are added or removed; the generator refuses a stale header.
   3: added MIN_TRSM_WORK / MIN_TRMM_WORK / TRI_WORK / MIN_TRI_WIDTH for the Left-side
      triangular kernels of the Cholesky-inverse phase (Rtrsm_omp, Rtrmm_omp).
   NOTE: patches/gmp_rgemm.py is a FROM-PRISTINE generator and its `_want` string still reads
   "MPLAPACK_OMP_TUNING_VERSION 2"; it writes this header only when it does not already exist,
   so it is unaffected on the path it is meant for (a pristine upstream tree). Re-running it
   against this fork now fails closed with a misleading "predates this generator" message. That
   is exactly the state sdpa-dd has been in since its own header reached VERSION 5 while
   patches/optimize_mplapack_omp.py still wants 2; the precedent is deliberate and the
   generator is not part of this fork's build. */
#define MPLAPACK_OMP_TUNING_VERSION 3

/* Minimum gemm work (m*n*k multiply-adds) before an OpenMP fork/join pays for itself.

   UNCALIBRATED, and unlike the triangular gates below it carries no measurement of its own.
   The constants that DO exist for this fork are in the triangular block: an mpf multiply-add
   at 256 bits is 74-76 ns on thanos and 33-35 ns on pi, and a fork/join is ~0.6 us on thanos
   and 0.6-2.1 us on pi. On those, 20000 is 0.7-1.5 ms of arithmetic, i.e. three orders of
   magnitude above the fork/join -- so this gate is conservative rather than wrong, but the
   margin is an accident and not a decision anyone measured. If gemm ever matters here,
   calibrate it the way the triangular gates were calibrated; do not tune it by reasoning.

   NOTE, because the mistake has been made in the sibling fork: a fork/join is ~0.6-2.1 us in
   TOTAL, not per thread. A proposal to make these gates a function of the team size on the
   basis of "~7 us per thread" (~168 us at 24 threads) is off by two orders of magnitude, and
   the measurements below show the break-even does not usefully track team size in any case.
   Read the "NOT scaled by team size" paragraph before changing the shape of any gate here. */
#ifndef MPLAPACK_OMP_MIN_GEMM_WORK
#define MPLAPACK_OMP_MIN_GEMM_WORK 20000.0
#endif

/* Minimum element count before parallelising the beta-scaling sweeps over C. */
#ifndef MPLAPACK_OMP_MIN_SCALE_WORK
#define MPLAPACK_OMP_MIN_SCALE_WORK 20000.0
#endif

/* Minimum width of the parallelised gemm loop (over j, so n iterations). A tall thin gemm
   with n=1 can clear MIN_GEMM_WORK yet offer a single iteration to share out. */
#ifndef MPLAPACK_OMP_MIN_GEMM_WIDTH
#define MPLAPACK_OMP_MIN_GEMM_WIDTH 2
#endif

/* ---------------- Left-side triangular kernels: Rtrsm_omp / Rtrmm_omp ----------------

   These two are the Cholesky-inverse phase -- Rtrsm against an identity RHS inside
   Lal::getInvLowTriangularMatrix, and Rtrmm forming Z^-1 = L**T * L inside
   Jal::getInvCholAndInv. Both are parallelised over the n COLUMNS of B (see the header comment
   in either file for why it is columns and not rows). Both are square at the call site, m == n
   == the SDP block order, so the work expression m*m*n is just n^3 and the gate is a threshold
   on the block order.

   MEASURED FOR THIS FORK, 2026-08-05. dd's constant was NOT carried across, and it would have
   been wrong by a factor of 15.6 in work (2.5 in n) if it had been.

   Method: the two kernels were called directly, gate removed, on the operands the solver
   actually hands them -- Rtrsm: A = the lower Cholesky factor, B = the identity, alpha = 1;
   Rtrmm: A = B = L^-1, alpha = 1. For mpf_class the operand VALUES matter, not just the shape:
   GMP's mpf_mul cost scales with the operand size field, which is smaller than the precision
   whenever trailing limbs are zero, so a matrix of random doubles is 1 limb where a genuine
   Cholesky factor at precision 256 is 4. The factor was therefore produced by running this
   fork's own Rpotrf, and the Rtrmm operand by running Rtrsm against an identity, i.e. the same
   two steps the solver performs. Two independent designs were run and agreed at every cell to
   within 2-8%:
     A  serial baseline in a SEPARATE process with OMP_NUM_THREADS=1, so no team is ever
        created (trap 1: idle GOMP workers spin in the barrier and get co-scheduled onto the
        master's SMT sibling, which once manufactured 12-14x phantom speedups);
     B  the same source compiled a second and a third time with only the gate constant
        differing (0.0 and 1e18) and the symbol renamed, timed INTERLEAVED in one process --
        B5's design, which exists because comparing a no-OpenMP against a -fopenmp compilation
        of the same source gave an arithmetically impossible 3.1x on two threads.
   One binary, one set of flags. Threads bound one per physical core, 0.4 s warm-up per shape
   against thanos' 1500 MHz schedutil idle floor, 3 rounds with the arm order rotated, worst
   round reported, foreign load recorded per row. Full tables:
   patches/port_notes/06_b3_gmp_gate_calibration_thanos.md (thanos) and
   patches/port_notes/07_b3_gmp_pi_calibration_and_gate_decision.md (pi, plus the gate
   decision). The ship gate and the proof that this constant is not a silent no-op are in
   patches/port_notes/11_b3_gmp_ship_gate_and_gate_fires.md.

   AN mpf MULTIPLY-ADD IS 74-76 ns ON THANOS and 33-35 ns ON PI (both from Rtrsm/Rtrmm at
   n=35..100, where the figure is flat). THE NUMBER DOES NOT TRAVEL -- a factor of 2.2 between
   two x86-64 boxes at the same precision -- so do not reuse either as a portable constant.
   For comparison a dd_real multiply-add is 2.6-4.1 ns and a qd_real one 123 ns MEASURED ON
   THANOS (b5_notes/03). Against gmp's 74-76 ns on that same box, gmp is ~1.6-2.1x FASTER per
   operation than qd, not slower as the porting brief assumed. Keep that comparison on one
   machine: gmp-on-pi against qd-on-thanos would read as ~4x and would be measuring the box.
   `precision 200` in param.sdpa is rounded up by GMP to 256 bits = four 64-bit limbs, and a
   4-limb mpf_mul is a handful of instructions where qd_real pays ~20 double operations plus
   renormalisation. Any claim about gmp's per-operation cost has to name BOTH a precision and a
   machine; these are valid only at 256 bits. Fork/join is ~0.6 us on thanos and 0.6-2.1 us on
   pi; the serial kernel's own fixed cost (four mpf_class constructions plus the Mlsame_gmp
   tests) is ~0.67 us on thanos and 0.27-0.31 us on pi.

   Speedup (serial / threaded), worst of 3 rounds, min statistic. thanos = EPYC 7232P (8
   physical cores, taskset 0-7); pi = i9-13900K (8 P-cores + 16 E-cores = 24 physical).

                          Rtrsm                              Rtrmm
       n  work     thanos 2/4/8    pi 2/4/8/24        thanos 2/4/8    pi 2/4/8/24
       2     8     0.41 0.44 0.46  0.09 0.10 0.09 0.09  0.39 0.45 0.45  0.08 0.09 0.08 0.08
       3    27     0.58 0.37 0.38  0.24 0.20 0.19 0.22  0.60 0.29 0.28  0.26 0.24 0.21 0.22
       4    64     0.76 0.43 0.42  0.49 0.40 0.39 0.38  0.82 0.44 0.44  0.52 0.50 0.48 0.48
       5   125     1.00 0.56 0.52  0.63 0.57 0.53 0.51  0.95 0.66 0.60  0.71 0.60 0.64 0.62
       6   216     1.12 0.80 0.77  0.80 0.77 0.67 0.67  1.09 0.94 0.88  0.89 0.97 0.87 0.88
       8   512     1.26 1.22 1.18  1.05 1.24 1.12 1.08  1.22 1.36 1.52  1.20 1.51 1.54 1.43
      10  1000     1.40 1.67 1.63  1.30 1.78 1.72 1.52  1.30 1.67 2.31  1.46 2.18 2.52 2.22
      12  1728     1.52 1.93 2.26  1.48 2.21 2.41 2.06  1.45 1.97 2.83  1.58 2.43 3.30 3.02
      16  4096     1.58 2.18 3.27  1.60 2.72 3.68 3.32  1.48 2.15 3.58  1.71 3.05 4.88 3.58
      19  6859     1.59 2.36 3.93  1.69 3.04 4.49 4.14  1.49 2.25 3.87  1.77 3.26 5.50 4.40
      25 15625     1.53 2.44 4.25  1.75 3.36 5.81 4.69  1.43 2.35 3.96  1.83 3.50 6.39 6.14
      35 42875     1.54 2.53 4.41  1.83 3.57 6.76 7.20  1.42 2.38 4.21  1.89 3.66 7.05 10.33
      50 1.25e5    1.54 2.61 4.81  1.89 3.74 7.28 10.57 1.42 2.35 4.36  1.83 3.24 6.66 9.65
      70 3.43e5    1.53 2.61 4.67  1.85 3.38 5.38 8.79  1.45 2.50 4.44  1.91 3.60 6.32 8.93
     100 1e6       1.55 2.63 4.44  1.91 3.66 6.81 10.46 1.38 2.33 4.32  1.95 3.82 7.25 12.17

   THE BREAK-EVEN IS n = 5-8, i.e. work 125-512 -- roughly 30x below dd's 15625 in work.
   n = 10 (work 1000) is the smallest size that wins at EVERY (machine, thread count, kernel)
   cell on BOTH the min and the median statistic; its worst cell is 1.30x by min and 1.26x by
   median. n = 8 wins on the min everywhere but falls to 0.90-1.01x on the median at pi's 4 and
   24 threads and at thanos' 8, so it is not far enough past the crossing to sit a gate on.
   That is the same criterion dd used (dd's chosen size had a worst cell of 1.37x).

   WHY NOT HIGHER, which is the failure mode that would be invisible. An over-large gate turns
   these kernels into no-ops that still link, still converge and still pass the bit-identity
   ship gate. A census of the blockStruct line of all 92 SDPLIB *.dat-s gives these SDP block
   orders at the small end, with occurrence counts:

       n:      2    3    4   5   6  7  8  9  10  11 12 14 15 16 18  19  20 25 26 30 35 37 40
       count: 156  156  35  25  13  1  3  1  37   1  1  1  1  1  1  33   2  1  1  6  1  1  2

   n = 10 is the third most common order in the whole library (37 blocks: control1, control2,
   hinf11, hinf14, ...) and n = 19 the fourth (33). dd's 15625 would reject both, and every
   block up to n = 24 -- which is where the entire measured gain between 1.3x and 5.8x lives.
   Meanwhile orders 2 and 3 are the two commonest of all (156 each) and lose 0.08-0.63x, so a
   work gate is mandatory; MIN_TRI_WIDTH alone would not do it, since it only rejects n = 1.
   1000 rejects everything up to n = 9 -- which is 235 blocks that lose or barely break even --
   and admits everything from n = 10 up.

   Note that n = 10 is admitted only because the comparison in Rtrsm_omp.cpp/Rtrmm_omp.cpp is
   `>=`. 1000 is exactly 10 cubed and sits ON a very common block order; anyone changing that
   comparison to `>` must re-derive the constant rather than assume it still means "n >= 10".

   A CORRECTION TO A GUESS THAT WOULD HAVE BEEN WRONG: on thanos these kernels saturate at
   4.4-4.8x on 8 cores (efficiency 0.55-0.60) and 2 threads only reach 1.38-1.58x, which
   invited an explanation in terms of gmpxx heap temporaries and mpf_class's scattered limb
   allocations. pi refutes that: the same code on 8 cores reaches 6.8-7.3x (0.85-0.91) and on 24
   cores 10.5-12.2x. The low ceiling is a property of thanos' EPYC 7232P (8 cores over 2 CCDs
   with separate L3s and modest memory bandwidth), not of gmp arithmetic. It is not contention
   from the team either: on thanos the `closed` arm running beside seven spinning workers
   matches the single-threaded `ser` arm to within 2%.

   NOT scaled by team size within 2..24 threads. The break-even rises only slightly with the
   team (pi's n = 8 is 1.24x at 4 threads and 1.08x at 24), and 1000 clears it at every measured
   thread count on both machines. Beyond 24 threads nothing is measured, so the threshold is
   grown linearly there as a guard rather than extrapolated silently.

   The two kernels get the same number. Rtrmm crosses slightly earlier than Rtrsm at 4 and 8
   threads on both machines, but the gap is smaller than the thanos-to-pi spread and no block
   order lies between the two crossings, so splitting them would be false precision.

   THE CONSTANT WAS VERIFIED AGAINST THE REAL SOLVER, NOT ONLY AGAINST THE MICROBENCHMARK.
   An instrumented build counting entries and threaded entries over 11 SDPLIB problems at 8
   threads (patches/port_notes/11) gives: 3930 of 21068 Rtrsm calls threaded (18.7%), and the
   cut is EXACTLY at this constant over all 31602 Rtrsm+Rtrmm calls -- every block order with
   m^3 < 1000 rejected, every order with m^3 >= 1000 threaded, no exceptions. control1, which
   has one n=5 block and one n=10 block, threads exactly half its calls. dd's 15625 in the same
   census would have threaded 670 calls instead of 3930 (-83%) while still admitting 99.06% of
   the cube-weighted work: by call count it guts the port, by weighted work it barely dents it,
   which is precisely why this had to be measured rather than reasoned about. The calls this
   gate rejects (81.3% of them) carry 0.158% of the cube-weighted work. */

/* Rtrsm, Left/Lower/NoTranspose: minimum work (m*m*n) to thread. 1000 = n >= 10. Measured
   break-even is n = 5-8 on thanos and pi; this is the first order that wins on both machines at
   2, 4, 8 and 24 threads on both the min and the median. */
#ifndef MPLAPACK_OMP_MIN_TRSM_WORK
#define MPLAPACK_OMP_MIN_TRSM_WORK 1000.0
#endif

/* Rtrmm, Left/Lower/Transpose: minimum work (m*m*n) to thread. Same value, same reason. */
#ifndef MPLAPACK_OMP_MIN_TRMM_WORK
#define MPLAPACK_OMP_MIN_TRMM_WORK 1000.0
#endif

/* Flat across the measured range (2..24 threads); grown linearly past 24, where nothing has
   been measured on either machine, so an unvalidated team size errs towards the serial kernel. */
#ifndef MPLAPACK_OMP_TRI_WORK
#define MPLAPACK_OMP_TRI_WORK(w, nt) ((nt) <= 24 ? (double)(w) : (double)(w) * (double)(nt) / 24.0)
#endif

/* Minimum number of columns to share out. Work alone is not sufficient: a tall thin solve with
   n=1 can clear the work gate yet offer a single iteration to the team. */
#ifndef MPLAPACK_OMP_MIN_TRI_WIDTH
#define MPLAPACK_OMP_MIN_TRI_WIDTH 2
#endif

#endif /* MPLAPACK_OMP_TUNING_H */
