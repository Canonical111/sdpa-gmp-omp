#ifndef MPLAPACK_OMP_TUNING_H
#define MPLAPACK_OMP_TUNING_H

/* Bump when macros are added or removed; the generator refuses a stale header. */
#define MPLAPACK_OMP_TUNING_VERSION 2

/* Minimum gemm work (m*n*k multiply-adds) before an OpenMP fork/join pays for itself. */
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

#endif /* MPLAPACK_OMP_TUNING_H */
