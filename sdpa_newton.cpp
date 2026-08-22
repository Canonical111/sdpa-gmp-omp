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

/* MODIFIED from upstream (GPLv2 2a notice), 2026-07-31: Schur-complement (bMat) construction threaded. See git log. */
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: the dense Schur complement bMat is built in its LOWER TRIANGLE ONLY; the strict upper half was accumulated every iteration and never read. See git log. */
#include <sdpa_newton.h>
#include <sdpa_parts.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <climits>
#include <cstdlib>
#include <iostream>

// review2 dimension edge 2: m and SDP_nBlock are each bounded by the reader,
// but their PRODUCT was formed in signed int at the allocation sites below.
// Bounding the factors by file size does not prove the product fits.
// Outside any _OPENMP guard: the call sites are unconditional, and a serial
// or flag-overridden build (CI's sanitizer/warnings jobs) needs this too.
static int checkedProductInt(int a, int b, const char *what) {
    const long long p = static_cast<long long>(a) * static_cast<long long>(b);
    if (a < 0 || b < 0 || p > INT_MAX) {
        std::cerr << "allocation size overflow: " << what << " = " << a << " * " << b << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(p);
}


// ---------------------------------------------------------------------------
// Thresholds for threading the Schur-complement (bMat) construction. Override with -D.
// The k1 x k2 loop costs roughly nConstraint^2 * blockDim; below this there is not
// enough work to amortise an OpenMP fork/join.
// ---------------------------------------------------------------------------
#ifndef SDPA_OMP_MIN_CONSTRAINTS
#define SDPA_OMP_MIN_CONSTRAINTS 8
#endif
#ifndef SDPA_OMP_MIN_BMAT_WORK
#define SDPA_OMP_MIN_BMAT_WORK 20000.0
#endif
// Hard ceiling on the extra memory used to privatise work1/work2 across threads.
// Cost is 2 * blockDim^2 * bytes-per-element per extra thread; on a large block that would
// otherwise grow without bound. If the full thread count would exceed this, the thread
// count for the block is reduced rather than the memory.
// Per-allocation malloc bookkeeping, charged once per separately allocated mantissa.
#ifndef SDPA_OMP_ALLOC_OVERHEAD
#define SDPA_OMP_ALLOC_OVERHEAD 16.0
#endif
#ifndef SDPA_OMP_MAX_PRIV_MB
#define SDPA_OMP_MAX_PRIV_MB 256.0
#endif
// Bytes actually occupied by one scalar. For mpf_class/qd_real the mantissa is stored inline,
// so sizeof() is exact. For mpf_class it is NOT: the object is a 24-byte descriptor whose
// limbs are allocated separately, so sizeof() undercounts by ~3x at 256-bit precision and
// the memory cap above would admit several times its nominal budget.
static inline double sdpa_omp_bytes_per_elem() {
    const double limbs =
        (double)((mpf_get_default_prec() + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) + 1.0;
    return (double)sizeof(mpf_class) + limbs * (double)sizeof(mp_limb_t) +
           SDPA_OMP_ALLOC_OVERHEAD;
}
// Choosing the parallel axis. For an F1/F2-dominated block the per-constraint setup is a
// blockDim^3 dense gemm, which Rgemm already threads well on its own; threading k1 instead
// makes each of those gemms serial and gains nothing. Below this gemm size Rgemm cannot
// parallelise effectively (blocks of 25-80 in the control* family) and threading k1 wins
// several-fold. Measured crossover on an i9-13900K lies between 80^3 and 100^3.
#ifndef SDPA_OMP_RGEMM_OWNS_BLOCK
#define SDPA_OMP_RGEMM_OWNS_BLOCK 700000.0
#endif

namespace sdpa {

Newton::Newton() {
    useFormula = NULL;

    bMat_type = DENSE;

    // Caution: if SDPA doesn't use sparse bMat,
    //          following variables are indefinite.
    this->SDP_nBlock = -1;
    SDP_number = NULL;
    SDP_location_sparse_bMat = NULL;
    SDP_constraint1 = NULL;
    SDP_constraint2 = NULL;
    SDP_blockIndex1 = NULL;
    SDP_blockIndex2 = NULL;
    this->SOCP_nBlock = -1;
    SOCP_number = NULL;
    SOCP_location_sparse_bMat = NULL;
    SOCP_constraint1 = NULL;
    SOCP_constraint2 = NULL;
    SOCP_blockIndex1 = NULL;
    SOCP_blockIndex2 = NULL;
    this->LP_nBlock = -1;
    LP_number = NULL;
    LP_location_sparse_bMat = NULL;
    LP_constraint1 = NULL;
    LP_constraint2 = NULL;
    LP_blockIndex1 = NULL;
    LP_blockIndex2 = NULL;

    ordering = NULL;
    reverse_ordering = NULL;
    diagonalIndex = NULL;
}

Newton::Newton(int m, int SDP_nBlock, int *SDP_blockStruct, int SOCP_nBlock, int *SOCP_blockStruct, int LP_nBlock) { initialize(m, SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock); }

Newton::~Newton() { terminate(); }

void Newton::initialize(int m, int SDP_nBlock, int *SDP_blockStruct, int SOCP_nBlock, int *SOCP_blockStruct, int LP_nBlock) {
    gVec.initialize(m);

    DxMat.initialize(SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock);
    DyVec.initialize(m);
    DzMat.initialize(SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock);
    r_zinvMat.initialize(SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock);
    x_rd_zinvMat.initialize(SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock);

    rNewCheck();
    useFormula = new FormulaType[checkedProductInt(m, SDP_nBlock, "useFormula")];
    if (useFormula == NULL) {
        rError("Newton:: memory exhausted ");
    }

    bMat_type = DENSE;

    // Caution: if SDPA doesn't use sparse bMat,
    //          following variables are indefinite.
    this->SDP_nBlock = -1;
    SDP_number = NULL;
    SDP_location_sparse_bMat = NULL;
    SDP_constraint1 = NULL;
    SDP_constraint2 = NULL;
    SDP_blockIndex1 = NULL;
    SDP_blockIndex2 = NULL;
    this->SOCP_nBlock = -1;
    SOCP_number = NULL;
    SOCP_location_sparse_bMat = NULL;
    SOCP_constraint1 = NULL;
    SOCP_constraint2 = NULL;
    SOCP_blockIndex1 = NULL;
    SOCP_blockIndex2 = NULL;
    this->LP_nBlock = -1;
    LP_number = NULL;
    LP_location_sparse_bMat = NULL;
    LP_constraint1 = NULL;
    LP_constraint2 = NULL;
    LP_blockIndex1 = NULL;
    LP_blockIndex2 = NULL;

    ordering = NULL;
    reverse_ordering = NULL;
    diagonalIndex = NULL;
}

void Newton::terminate() {

    if (bMat_type == SPARSE) {

        if (SDP_location_sparse_bMat && SDP_constraint1 && SDP_constraint2 && SDP_blockIndex1 && SDP_blockIndex2) {
            for (int k = 0; k < SDP_nBlock; ++k) {
                delete[] SDP_location_sparse_bMat[k];
                delete[] SDP_constraint1[k];
                delete[] SDP_constraint2[k];
                delete[] SDP_blockIndex1[k];
                delete[] SDP_blockIndex2[k];
                SDP_location_sparse_bMat[k] = NULL;
                SDP_constraint1[k] = NULL;
                SDP_constraint2[k] = NULL;
                SDP_blockIndex1[k] = NULL;
                SDP_blockIndex2[k] = NULL;
            }
            delete[] SDP_number;
            delete[] SDP_location_sparse_bMat;
            delete[] SDP_constraint1;
            delete[] SDP_constraint2;
            delete[] SDP_blockIndex1;
            delete[] SDP_blockIndex2;
            SDP_number = NULL;
            SDP_location_sparse_bMat = NULL;
            SDP_constraint1 = NULL;
            SDP_constraint2 = NULL;
            SDP_blockIndex1 = NULL;
            SDP_blockIndex2 = NULL;
        }
#if 0
    if (SOCP_location_sparse_bMat && SOCP_constraint1 && SOCP_constraint2
	&& SOCP_blockIndex1 && SOCP_blockIndex2) {
      for (int k=0; k<SOCP_nBlock; ++k) {
	delete[] SOCP_location_sparse_bMat[k];
	delete[] SOCP_constraint1[k];    delete[] SOCP_constraint2[k];
	delete[] SOCP_blockIndex1[k];    delete[] SOCP_blockIndex2[k];
	SOCP_location_sparse_bMat[k] = NULL;
	SOCP_constraint1[k] = NULL;   SOCP_constraint2[k] = NULL;
	SOCP_blockIndex1[k] = NULL;   SOCP_blockIndex2[k] = NULL;
      }
      delete[] SOCP_number;  delete[] SOCP_location_sparse_bMat;
      delete[] SOCP_constraint1;  delete[] SOCP_constraint2;
      delete[] SOCP_blockIndex1;  delete[] SOCP_blockIndex2;
      SOCP_number = NULL;  SOCP_location_sparse_bMat = NULL;
      SOCP_constraint1 = NULL;  SOCP_constraint2 = NULL;
      SOCP_blockIndex1 = NULL;  SOCP_blockIndex2 =NULL;
    }
#endif
        if (LP_location_sparse_bMat && LP_constraint1 && LP_constraint2 && LP_blockIndex1 && LP_blockIndex2) {
            for (int k = 0; k < LP_nBlock; ++k) {
                delete[] LP_location_sparse_bMat[k];
                delete[] LP_constraint1[k];
                delete[] LP_constraint2[k];
                delete[] LP_blockIndex1[k];
                delete[] LP_blockIndex2[k];
                LP_location_sparse_bMat[k] = NULL;
                LP_constraint1[k] = NULL;
                LP_constraint2[k] = NULL;
                LP_blockIndex1[k] = NULL;
                LP_blockIndex2[k] = NULL;
            }
            delete[] LP_number;
            delete[] LP_location_sparse_bMat;
            delete[] LP_constraint1;
            delete[] LP_constraint2;
            delete[] LP_blockIndex1;
            delete[] LP_blockIndex2;
            LP_number = NULL;
            LP_location_sparse_bMat = NULL;
            LP_constraint1 = NULL;
            LP_constraint2 = NULL;
            LP_blockIndex1 = NULL;
            LP_blockIndex2 = NULL;
        }

        if (ordering) {
            delete[] ordering;
            ordering = NULL;
        }
        if (reverse_ordering) {
            delete[] reverse_ordering;
            reverse_ordering = NULL;
        }
        if (diagonalIndex) {
            delete[] diagonalIndex;
            diagonalIndex = NULL;
        }
        sparse_bMat.terminate();

    } else { // bMat_type == DENSE
        bMat.terminate();
    }

    gVec.terminate();
    DxMat.terminate();
    DyVec.terminate();
    DzMat.terminate();
    r_zinvMat.terminate();
    x_rd_zinvMat.terminate();

    if (useFormula != NULL) {
        delete[] useFormula;
    }
    useFormula = NULL;
}

void Newton::initialize_dense_bMat(int m) {
    //  bMat_type = DENSE;
    //  printf("DENSE computations\n");
    bMat.initialize(m, m, DenseMatrix::DENSE);
}

// 2008/03/12 kazuhide nakata
void Newton::initialize_sparse_bMat(int m, IV *newToOldIV, IVL *symbfacIVL) {

    //  bMat_type = SPARSE;
    //  printf("SPARSE computation\n");

    int i, j, k;
    int *newToOld;

    newToOld = IV_entries(newToOldIV);

    rNewCheck();
    ordering = new int[m];
    if (ordering == NULL) {
        rError("Newton::initialize_sparse_bMat memory exhausted ");
    }
    for (i = 0; i < m; i++) {
        ordering[i] = newToOld[i];
    }

    rNewCheck();
    reverse_ordering = new int[m];
    if (reverse_ordering == NULL) {
        rError("Newton::initialize_sparse_bMat memory exhausted ");
    }
    for (i = 0; i < m; i++) {
        reverse_ordering[ordering[i]] = i;
    }

    // separate front or back node
    int *counter;
    int nClique = IVL_nlist(symbfacIVL);
    int psize;
    int *pivec;
    bool *bnode;
    int *nFront;

    rNewCheck();
    counter = new int[m];
    bnode = new bool[m];
    nFront = new int[nClique];

    if ((counter == NULL) || (bnode == NULL) || (nFront == NULL)) {
        rError("Newton::initialize_sparse_bMat memory exhausted ");
    }

    for (i = 0; i < m; i++) {
        bnode[i] = false;
        counter[i] = -1;
    }

    // search number of front
    for (int l = nClique - 1; l >= 0; l--) {
        IVL_listAndSize(symbfacIVL, l, &psize, &pivec);
        for (i = 0; i < psize; i++) {
            int ii = reverse_ordering[pivec[i]];
            if (bnode[ii] == false) {
                counter[ii] = psize - i;
                bnode[ii] = true;
            } else {
                nFront[l] = i;
                break;
            }
        }
        if (i == psize) {
            nFront[l] = psize;
        }
    }

    // error check
    for (i = 0; i < m; i++) {
        if (counter[i] == -1) {
            rError("Newton::initialize_sparse_bMat: program bug");
        }
    }

    // make index of diagonal
    rNewCheck();
    diagonalIndex = new int[m + 1];
    if (diagonalIndex == NULL) {
        rError("Newton::initialize_sparse_bMat memory exhausted ");
    }

    /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-12: this prefix sum is the stored
       non-zero count of the sparse bMat, and it is an int. Past INT_MAX it wraps negative
       and is handed straight to SparseMatrix::initialize as an allocation length. Checked
       in double alongside so an over-large factor is a diagnostic rather than undefined
       behaviour -- which is what lets Chordal::ordering_bMat say the sparse path "will be
       attempted" for m > 46340 without also promising it must succeed. Unreachable for any
       problem upstream could allocate. */
    diagonalIndex[0] = 0;
    long long diagonalIndex_exact = 0;   /* 64-bit running count: validate, THEN narrow */
    for (i = 1; i < m + 1; i++) {
        diagonalIndex_exact += (long long)counter[i - 1];
        if (diagonalIndex_exact > (long long)INT_MAX) {
            rError("Newton::initialize_sparse_bMat: the sparse bMat needs "
                   << diagonalIndex_exact << " stored elements at row " << i
                   << ", past INT_MAX=" << INT_MAX << ". The dense representation is also"
                   << " unavailable above m=46340, so this problem is too large for this"
                   << " build.");
        }
        diagonalIndex[i] = diagonalIndex[i - 1] + counter[i - 1];
    }

    // initialize sparse_bMat
    sparse_bMat.initialize(m, m, SparseMatrix::SPARSE, diagonalIndex[m]);

    // initialize index of sparse_bmat
    int nonzeros = 0;
    for (int l = 0; l < nClique; l++) {
        IVL_listAndSize(symbfacIVL, l, &psize, &pivec);
        for (i = 0; i < nFront[l]; i++) {
            int ii = reverse_ordering[pivec[i]];
            for (j = i; j < psize; j++) {
                int jj = reverse_ordering[pivec[j]];
                int index = diagonalIndex[ii] + j - i;
                sparse_bMat.row_index[index] = ii;
                sparse_bMat.column_index[index] = jj;
                nonzeros++;
            }
        }
    }
    // error check
    if (nonzeros != sparse_bMat.NonZeroNumber) {
        rError("Newton::initialize_sparse_bMat  probram bug");
    }
    sparse_bMat.NonZeroCount = nonzeros;
    //  sparse_bMat.display();

    delete[] counter;
    delete[] bnode;
    delete[] nFront;
}

// 2008/03/12 kazuhide nakata
void Newton::initialize_bMat(int m, Chordal &chordal, InputData &inputData, FILE *fpOut) {
    /* Create clique tree */

    switch (chordal.best) {
    case -1: {
        bMat_type = DENSE;
        printf("DENSE computations\n");
        fprintf(fpOut, "DENSE computation\n");
        initialize_dense_bMat(m);
        break;
    }
    case 0: {
        rError("no support for METIS");
        break;
    }
    case 1: {
        bMat_type = SPARSE;
        printf("SPARSE computation\n");
        fprintf(fpOut, "SPARSE computation\n");
        initialize_sparse_bMat(m, chordal.newToOldIV_MMD, chordal.symbfacIVL_MMD);
        make_aggrigateIndex(inputData);
        break;
    }
    case 2: {
        bMat_type = SPARSE;
        printf("SPARSE computation\n");
        fprintf(fpOut, "SPARSE computation\n");
        initialize_sparse_bMat(m, chordal.newToOldIV_ND, chordal.symbfacIVL_ND);
        make_aggrigateIndex(inputData);
        break;
    }
    case 3: {
        bMat_type = SPARSE;
        printf("SPARSE computation\n");
        fprintf(fpOut, "SPARSE computation\n");
        initialize_sparse_bMat(m, chordal.newToOldIV_MS, chordal.symbfacIVL_MS);
        make_aggrigateIndex(inputData);
        break;
    }
    case 4: {
        bMat_type = SPARSE;
        printf("SPARSE computation\n");
        fprintf(fpOut, "SPARSE computation\n");
        initialize_sparse_bMat(m, chordal.newToOldIV_NDMS, chordal.symbfacIVL_NDMS);
        make_aggrigateIndex(inputData);
        break;
    }
    }
}

void Newton::make_aggrigateIndex_SDP(InputData &inputData) {
    int t, ii, jj;
    const int m = inputData.b.nDim; // constraint count; bounds the source indices below

    SDP_nBlock = inputData.SDP_nBlock;
    rNewCheck();
    SDP_number = new int[SDP_nBlock];
    if (SDP_number == NULL) {
        rError("Newton::make_aggrigateIndex_SDP memory exhausted ");
    }

    // memory allocate for aggrigateIndex
    rNewCheck();
    SDP_constraint1 = new int *[SDP_nBlock];
    SDP_constraint2 = new int *[SDP_nBlock];
    SDP_blockIndex1 = new int *[SDP_nBlock];
    SDP_blockIndex2 = new int *[SDP_nBlock];
    SDP_location_sparse_bMat = new int *[SDP_nBlock];
    if ((SDP_constraint1 == NULL) || (SDP_constraint2 == NULL) || (SDP_blockIndex1 == NULL) || (SDP_blockIndex2 == NULL) || (SDP_location_sparse_bMat == NULL)) {
        rError("Newton::make_aggrigateIndex_SDP memory exhausted ");
    }

    /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: validate the aggregate map's
       source constraints BEFORE the triangular size is computed and allocated.

       Two distinct failure classes, and conflating them was a real error in an earlier draft
       of the threading plan:

         INVALID  -- a source constraint index out of [0, m), or the same index appearing
                     twice in one block. Either can make the pair enumeration below write
                     outside the n(n+1)/2 allocation, or produce two pairs mapping to one
                     destination. Serial execution does not make that safe, so it is a hard
                     failure here, in EVERY mode. Validity is not a performance knob.
         NOT PARALLELISABLE -- a structurally valid map that some future admission rule
                     declines to thread. That is a fallback, decided elsewhere.

       The dat-s reader sorts and deduplicates block indices, so ordinary parsed input passes.
       Programmatic or internal callers are not bound by that, and this is the last point at
       which a violation is cheap to detect rather than expensive to debug.

       The size is also formed in 64-bit and checked before narrowing: n(n+1)/2 overflows int
       at n = 65536, and `(n+1)*n/2` in int arithmetic overflows even earlier, at n = 46341.
       See git log. */
    for (int l = 0; l < SDP_nBlock; l++) {
        const int n = inputData.SDP_nConstraint[l];
        if (n < 0) {
            rError("Newton::make_aggrigateIndex_SDP: block " << l << " has negative constraint"
                   << " count " << n);
        }
        {
            std::vector<char> seen(m, 0);
            for (int k = 0; k < n; ++k) {
                const int ii = inputData.SDP_constraint[l][k];
                if (ii < 0 || ii >= m) {
                    rError("Newton::make_aggrigateIndex_SDP: block " << l << " entry " << k
                           << " names constraint " << ii << ", outside [0," << m << ")."
                           << " The aggregate map would index outside its allocation.");
                }
                if (seen[ii]) {
                    rError("Newton::make_aggrigateIndex_SDP: block " << l << " names"
                           << " constraint " << ii << " more than once. The pair enumeration"
                           << " assumes one entry per constraint, so duplicates would write"
                           << " two contributions to one bMat location and overrun the"
                           << " triangular allocation. This is a malformed aggregate map,"
                           << " not a slow one -- it cannot be made safe by running serially.");
                }
                seen[ii] = 1;
            }
        }
        const long long tmp64 = ((long long)n + 1LL) * (long long)n / 2LL;
        if (tmp64 > (long long)INT_MAX) {
            rError("Newton::make_aggrigateIndex_SDP: block " << l << " with " << n
                   << " constraints needs " << tmp64 << " pairs, past the INT_MAX="
                   << INT_MAX << " limit of the aggregate index arrays");
        }
        int tmp = (int)tmp64;
        rNewCheck();
        SDP_number[l] = tmp;
        SDP_constraint1[l] = new int[tmp];
        SDP_constraint2[l] = new int[tmp];
        SDP_blockIndex1[l] = new int[tmp];
        SDP_blockIndex2[l] = new int[tmp];
        SDP_location_sparse_bMat[l] = new int[tmp];
        if ((SDP_constraint1[l] == NULL) || (SDP_constraint2[l] == NULL) || (SDP_blockIndex1[l] == NULL) || (SDP_blockIndex2[l] == NULL) || (SDP_location_sparse_bMat[l] == NULL)) {
            rError("Newton::make_aggrigateIndex_SDP memory exhausted ");
        }
    }

    for (int l = 0; l < SDP_nBlock; l++) {
        int NonZeroCount = 0;

        for (int k1 = 0; k1 < inputData.SDP_nConstraint[l]; k1++) {
            int i = inputData.SDP_constraint[l][k1];
            int ib = inputData.SDP_blockIndex[l][k1];
            int inz = inputData.A[i].SDP_sp_block[ib].NonZeroEffect;

            for (int k2 = 0; k2 < inputData.SDP_nConstraint[l]; k2++) {
                int j = inputData.SDP_constraint[l][k2];
                int jb = inputData.SDP_blockIndex[l][k2];
                int jnz = inputData.A[j].SDP_sp_block[jb].NonZeroEffect;

                if ((inz < jnz) || ((inz == jnz) && (i < j))) {
                    continue;
                }

                // set index which A_i and A_j are not zero matrix
                SDP_constraint1[l][NonZeroCount] = i;
                SDP_constraint2[l][NonZeroCount] = j;
                SDP_blockIndex1[l][NonZeroCount] = ib;
                SDP_blockIndex2[l][NonZeroCount] = jb;
                if (reverse_ordering[i] < reverse_ordering[j]) {
                    ii = reverse_ordering[i];
                    jj = reverse_ordering[j];
                } else {
                    jj = reverse_ordering[i];
                    ii = reverse_ordering[j];
                }

                // binary search for index of sparse_bMat
                t = -1;
                int begin = diagonalIndex[ii];
                int end = diagonalIndex[ii + 1] - 1;
                int target = (begin + end) / 2;
                while (end - begin > 1) {
                    if (sparse_bMat.column_index[target] < jj) {
                        begin = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] > jj) {
                        end = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] == jj) {
                        t = target;
                        break;
                    }
                }
                if (t == -1) {
                    if (sparse_bMat.column_index[begin] == jj) {
                        t = begin;
                    } else if (sparse_bMat.column_index[end] == jj) {
                        t = end;
                    } else {
                        rError("Newton::make_aggrigateIndex_SDP  program bug");
                    }
                }

                SDP_location_sparse_bMat[l][NonZeroCount] = t;
                NonZeroCount++;
            }
        } // for k1

        /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: the two invariants the
           threaded assembly's ownership argument rests on, asserted where the map is BUILT,
           in every mode -- not scanned later in a debug build.

           (1) The enumeration must have produced exactly the n(n+1)/2 pairs the arrays were
               sized for. Fewer means the keep-filter dropped a pair (both orders rejected),
               and the trailing slots hold indeterminate indices that the assembly would read.
           (2) Within a block, no two pairs may map to one sparse_bMat location. The parallel
               assembly writes destinations concurrently on the strength of this; a duplicate
               would be a data race AND a changed accumulation order. */
        if (NonZeroCount != SDP_number[l]) {
            rError("Newton::make_aggrigateIndex_SDP: block " << l << " enumerated "
                   << NonZeroCount << " pairs but " << SDP_number[l] << " were allocated;"
                   << " the aggregate map is inconsistent");
        }
        {
            static std::vector<char> loc_seen; // sized once; cleared per block below
            if ((int)loc_seen.size() < sparse_bMat.NonZeroCount) {
                loc_seen.assign(sparse_bMat.NonZeroCount, 0);
            }
            for (int q = 0; q < SDP_number[l]; ++q) {
                const int loc = SDP_location_sparse_bMat[l][q];
                if (loc < 0 || loc >= sparse_bMat.NonZeroCount) {
                    rError("Newton::make_aggrigateIndex_SDP: block " << l << " pair " << q
                           << " maps to location " << loc << ", outside the sparse bMat");
                }
                if (loc_seen[loc]) {
                    rError("Newton::make_aggrigateIndex_SDP: block " << l
                           << " maps two pairs to sparse bMat location " << loc
                           << "; concurrent assembly over this block would race");
                }
                loc_seen[loc] = 1;
            }
            for (int q = 0; q < SDP_number[l]; ++q) {
                loc_seen[SDP_location_sparse_bMat[l][q]] = 0;
            }
        }
    }     // for k  kth block
}

void Newton::make_aggrigateIndex_SOCP(InputData &inputData) {
    int t, ii, jj;

    SOCP_nBlock = inputData.SOCP_nBlock;
    rNewCheck();
    SOCP_number = new int[SOCP_nBlock];
    if (SOCP_number == NULL) {
        rError("Newton::make_aggrigateIndex_SOCP memory exhausted ");
    }

    // memory allocate for aggrigateIndex
    rNewCheck();
    SOCP_constraint1 = new int *[SOCP_nBlock];
    SOCP_constraint2 = new int *[SOCP_nBlock];
    SOCP_blockIndex1 = new int *[SOCP_nBlock];
    SOCP_blockIndex2 = new int *[SOCP_nBlock];
    SOCP_location_sparse_bMat = new int *[SOCP_nBlock];
    if ((SOCP_constraint1 == NULL) || (SOCP_constraint2 == NULL) || (SOCP_blockIndex1 == NULL) || (SOCP_blockIndex2 == NULL) || (SOCP_location_sparse_bMat == NULL)) {
        rError("Newton::make_aggrigateIndex_SOCP memory exhausted ");
    }

    for (int l = 0; l < SOCP_nBlock; l++) {
        int tmp = (inputData.SOCP_nConstraint[l] + 1) * inputData.SOCP_nConstraint[l] / 2;
        rNewCheck();
        SOCP_number[l] = tmp;
        SOCP_constraint1[l] = new int[tmp];
        SOCP_constraint2[l] = new int[tmp];
        SOCP_blockIndex1[l] = new int[tmp];
        SOCP_blockIndex2[l] = new int[tmp];
        SOCP_location_sparse_bMat[l] = new int[tmp];
        if ((SOCP_constraint1[l] == NULL) || (SOCP_constraint2[l] == NULL) || (SOCP_blockIndex1[l] == NULL) || (SOCP_blockIndex2[l] == NULL) || (SOCP_location_sparse_bMat[l] == NULL)) {
            rError("Newton::make_aggrigateIndex_SOCP memory exhausted ");
        }
    }

    for (int l = 0; l < SOCP_nBlock; l++) {
        int NonZeroCount = 0;

        for (int k1 = 0; k1 < inputData.SOCP_nConstraint[l]; k1++) {
            int i = inputData.SOCP_constraint[l][k1];
            int ib = inputData.SOCP_blockIndex[l][k1];
            int inz = inputData.A[i].SOCP_sp_block[ib].NonZeroEffect;

            for (int k2 = 0; k2 < inputData.SOCP_nConstraint[l]; k2++) {
                int j = inputData.SOCP_constraint[l][k2];
                int jb = inputData.SOCP_blockIndex[l][k2];
                int jnz = inputData.A[j].SOCP_sp_block[jb].NonZeroEffect;

                if ((inz < jnz) || ((inz == jnz) && (i < j))) {
                    continue;
                }

                // set index which A_i and A_j are not zero matrix
                SOCP_constraint1[l][NonZeroCount] = i;
                SOCP_constraint2[l][NonZeroCount] = j;
                SOCP_blockIndex1[l][NonZeroCount] = ib;
                SOCP_blockIndex2[l][NonZeroCount] = jb;
                if (reverse_ordering[i] < reverse_ordering[j]) {
                    ii = reverse_ordering[i];
                    jj = reverse_ordering[j];
                } else {
                    jj = reverse_ordering[i];
                    ii = reverse_ordering[j];
                }

                // binary search for index of sparse_bMat
                t = -1;
                int begin = diagonalIndex[ii];
                int end = diagonalIndex[ii + 1] - 1;
                int target = (begin + end) / 2;
                while (end - begin > 1) {
                    if (sparse_bMat.column_index[target] < jj) {
                        begin = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] > jj) {
                        end = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] == jj) {
                        t = target;
                        break;
                    }
                }
                if (t == -1) {
                    if (sparse_bMat.column_index[begin] == jj) {
                        t = begin;
                    } else if (sparse_bMat.column_index[end] == jj) {
                        t = end;
                    } else {
                        rError("Newton::make_aggrigateIndex_SDP  program bug");
                    }
                }

                SOCP_location_sparse_bMat[l][NonZeroCount] = t;
                NonZeroCount++;
            }
        } // for k1
    }     // for k  kth block
}

void Newton::make_aggrigateIndex_LP(InputData &inputData) {
    int t, ii, jj;

    LP_nBlock = inputData.LP_nBlock;
    rNewCheck();
    LP_number = new int[LP_nBlock];
    if (LP_number == NULL) {
        rError("Newton::make_aggrigateIndex_LP memory exhausted ");
    }

    // memory allocate for aggrigateIndex
    rNewCheck();
    LP_constraint1 = new int *[LP_nBlock];
    LP_constraint2 = new int *[LP_nBlock];
    LP_blockIndex1 = new int *[LP_nBlock];
    LP_blockIndex2 = new int *[LP_nBlock];
    LP_location_sparse_bMat = new int *[LP_nBlock];
    if ((LP_constraint1 == NULL) || (LP_constraint2 == NULL) || (LP_blockIndex1 == NULL) || (LP_blockIndex2 == NULL) || (LP_location_sparse_bMat == NULL)) {
        rError("Newton::make_aggrigateIndex_LP memory exhausted ");
    }

    /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: same validation as the SDP
       builder -- source uniqueness and bounds before the triangular size is computed, size
       formed wide and checked before narrowing. The LP assembly's writes are only provably
       disjoint if this holds, and §4b of the status document cannot call them disjoint on an
       unvalidated map. See git log. */
    const int m_LP = inputData.b.nDim;
    for (int l = 0; l < LP_nBlock; l++) {
        const int nlp = inputData.LP_nConstraint[l];
        if (nlp < 0) {
            rError("Newton::make_aggrigateIndex_LP: block " << l << " has negative constraint"
                   << " count " << nlp);
        }
        {
            std::vector<char> seen(m_LP, 0);
            for (int k = 0; k < nlp; ++k) {
                const int ii2 = inputData.LP_constraint[l][k];
                if (ii2 < 0 || ii2 >= m_LP) {
                    rError("Newton::make_aggrigateIndex_LP: block " << l << " entry " << k
                           << " names constraint " << ii2 << ", outside [0," << m_LP << ")");
                }
                if (seen[ii2]) {
                    rError("Newton::make_aggrigateIndex_LP: block " << l << " names constraint "
                           << ii2 << " more than once; the pair enumeration would overrun its"
                           << " triangular allocation");
                }
                seen[ii2] = 1;
            }
        }
        const long long tmp64 = ((long long)nlp + 1LL) * (long long)nlp / 2LL;
        if (tmp64 > (long long)INT_MAX) {
            rError("Newton::make_aggrigateIndex_LP: block " << l << " with " << nlp
                   << " constraints needs " << tmp64 << " pairs, past INT_MAX");
        }
        int tmp = (int)tmp64;
        rNewCheck();
        LP_number[l] = tmp;
        LP_constraint1[l] = new int[tmp];
        LP_constraint2[l] = new int[tmp];
        LP_blockIndex1[l] = new int[tmp];
        LP_blockIndex2[l] = new int[tmp];
        LP_location_sparse_bMat[l] = new int[tmp];
        if ((LP_constraint1[l] == NULL) || (LP_constraint2[l] == NULL) || (LP_blockIndex1[l] == NULL) || (LP_blockIndex2[l] == NULL) || (LP_location_sparse_bMat[l] == NULL)) {
            rError("Newton::make_aggrigateIndex_LP memory exhausted ");
        }
    }

    for (int l = 0; l < LP_nBlock; l++) {
        int NonZeroCount = 0;

        for (int k1 = 0; k1 < inputData.LP_nConstraint[l]; k1++) {
            int i = inputData.LP_constraint[l][k1];
            int ib = inputData.LP_blockIndex[l][k1];

            for (int k2 = 0; k2 < inputData.LP_nConstraint[l]; k2++) {
                int j = inputData.LP_constraint[l][k2];
                int jb = inputData.LP_blockIndex[l][k2];

                if (i < j) {
                    continue;
                }

                // set index which A_i and A_j are not zero matrix
                LP_constraint1[l][NonZeroCount] = i;
                LP_constraint2[l][NonZeroCount] = j;
                LP_blockIndex1[l][NonZeroCount] = ib;
                LP_blockIndex2[l][NonZeroCount] = jb;
                if (reverse_ordering[i] < reverse_ordering[j]) {
                    ii = reverse_ordering[i];
                    jj = reverse_ordering[j];
                } else {
                    jj = reverse_ordering[i];
                    ii = reverse_ordering[j];
                }

                // binary search for index of sparse_bMat
                t = -1;
                int begin = diagonalIndex[ii];
                int end = diagonalIndex[ii + 1] - 1;
                int target = (begin + end) / 2;
                while (end - begin > 1) {
                    if (sparse_bMat.column_index[target] < jj) {
                        begin = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] > jj) {
                        end = target;
                        target = (begin + end) / 2;
                    } else if (sparse_bMat.column_index[target] == jj) {
                        t = target;
                        break;
                    }
                }
                if (t == -1) {
                    if (sparse_bMat.column_index[begin] == jj) {
                        t = begin;
                    } else if (sparse_bMat.column_index[end] == jj) {
                        t = end;
                    } else {
                        rError("Newton::make_aggrigateIndex_SDP  program bug");
                    }
                }

                LP_location_sparse_bMat[l][NonZeroCount] = t;
                NonZeroCount++;
            }
        } // for k1
    }     // for k  kth block
}

void Newton::make_aggrigateIndex(InputData &inputData) {
    make_aggrigateIndex_SDP(inputData);
    //  make_aggrigateIndex_SOCP(inputData);
    make_aggrigateIndex_LP(inputData);
}

void Newton::computeFormula_SDP(InputData &inputData, mpf_class DenseRatio, mpf_class Kappa) {
    int m = inputData.b.nDim;
    int SDP_nBlock = inputData.SDP_nBlock;

    int *upNonZeroCount;
    rNewCheck();
    upNonZeroCount = new int[checkedProductInt(m, SDP_nBlock, "upNonZeroCount")];
    if (upNonZeroCount == NULL) {
        rError("Newton:: memory exhausted ");
    }

    // We have no chance to use DenseRatio
    if (upNonZeroCount == NULL || useFormula == NULL) {
        rError("Newton:: failed initialization");
    }

    SparseLinearSpace *A = inputData.A;

#if 0
  for (int k=0; k<m; ++k) {
    for (int l=0; l<inputData.A[0].nBlock; ++l) {
      rMessage("A[" << k << "].ele[" << l << "] ="
	       << inputData.A[k].ele[l].NonZeroEffect);
    }
  }
#endif

    // Count sum of number of elements
    // that each number of elements are less than own.

    for (int iter = 0; iter < m * SDP_nBlock; iter++) {
        upNonZeroCount[iter] = 0;
    }

    for (int l = 0; l < SDP_nBlock; ++l) {
        for (int k1 = 0; k1 < inputData.SDP_nConstraint[l]; k1++) {
            int i = inputData.SDP_constraint[l][k1];
            int ib = inputData.SDP_blockIndex[l][k1];
            int inz = A[i].SDP_sp_block[ib].NonZeroEffect;
            int up = inz;
            // rMessage("up = " << up);

            for (int k2 = 0; k2 < inputData.SDP_nConstraint[l]; k2++) {
                int j = inputData.SDP_constraint[l][k2];
                int jb = inputData.SDP_blockIndex[l][k2];
                int jnz = A[j].SDP_sp_block[jb].NonZeroEffect;
                //	printf("%d %d %d %d %d %d\n",i,ib,inz, j, jb,jnz);
                if (jnz < inz) {
                    up += jnz;
                }
#if 1
                else if ((jnz == inz) && (j < i)) {
                    up += jnz;
                }
#endif
            }
            upNonZeroCount[i * SDP_nBlock + l] = up;
            // rMessage("up = " << up);
        }
    }

    // Determine which formula
    for (int l = 0; l < SDP_nBlock; ++l) {
        int countf1, countf2, countf3;
        countf1 = countf2 = countf3 = 0;
        for (int k = 0; k < inputData.SDP_nConstraint[l]; k++) {
            int i = inputData.SDP_constraint[l][k];
            int ib = inputData.SDP_blockIndex[l][k];
            mpf_class inz = inputData.A[i].SDP_sp_block[ib].NonZeroEffect;

            mpf_class f1, f2, f3;
            mpf_class n = inputData.A[i].SDP_sp_block[ib].nRow;
            mpf_class up = upNonZeroCount[i * SDP_nBlock + l];

            f1 = Kappa * n * inz + n * n * n + Kappa * up;
            f2 = Kappa * n * inz + Kappa * (n + 1) * up;
#if 1
            f3 = Kappa * (2 * Kappa * inz + 1) * up / Kappa;
#else
            f3 = Kappa * (2 * Kappa * inz + 1) * up;
#endif
            // rMessage("up = " << up << " nonzero = " << nonzero);
            // rMessage("f1=" << f1 << " f2=" << f2 << " f3=" << f3);
            // printf("%d %d %lf %lf %lf %lf\n",k,l,nonzero,f1,f2,f3);
            if (inputData.A[i].SDP_sp_block[ib].type == SparseMatrix::DENSE) {
                // if DENSE, we use only F1 or F2,
                // that is we don't use F3
                if (f1 < f2) {
                    useFormula[i * SDP_nBlock + l] = F1;
                    countf1++;
                } else {
                    useFormula[i * SDP_nBlock + l] = F2;
                    countf2++;
                }
            } else {
                // this case is SPARSE
                if (f1 < f2 && f1 < f3) {
                    //	   rMessage("line " << k << " is F1");
                    useFormula[i * SDP_nBlock + l] = F1;
                    countf1++;
                } else if (f2 < f3) {
                    //	   rMessage("line " << k << " is F2");
                    useFormula[i * SDP_nBlock + l] = F2;
                    countf2++;
                } else {
                    //	   rMessage("line " << k << " is F3");
                    useFormula[i * SDP_nBlock + l] = F3;
                    countf3++;
                }
            }
        }
// rMessage("Kappa = " << Kappa);
#if 0
    rMessage("count f1 = " << countf1
	     << ":: count f2 = " << countf2
	     << ":: count f3 = " << countf3);
#endif
    } // end of 'for (int l)'

    if (upNonZeroCount != NULL) {
        delete[] upNonZeroCount;
    }
    upNonZeroCount = NULL;

    return;
}

void Newton::compute_rMat(Newton::WHICH_DIRECTION direction, AverageComplementarity &mu, DirectionParameter &beta, Solutions &currentPt, WorkVariables &work) {

    mpf_class MMONE = -1.0;
    //     CORRECTOR ::  r_zinv = (-XZ -dXdZ + mu I)Z^{-1}
    // not CORRECTOR ::  r_zinv = (-XZ + mu I)Z^{-1}
    mpf_class target = beta.value * mu.current;
    Lal::let(r_zinvMat, '=', currentPt.invzMat, '*', &target);
    Lal::let(r_zinvMat, '=', r_zinvMat, '+', currentPt.xMat, &MMONE);

    if (direction == CORRECTOR) {
        // work.DLS1 = Dx Dz Z^{-1}
        Jal::ns_jordan_triple_product(work.DLS1, DxMat, DzMat, currentPt.invzMat, work.DLS2);
        Lal::let(r_zinvMat, '=', r_zinvMat, '+', work.DLS1, &MMONE);
    }

    //  rMessage("r_zinvMat = ");
    //  r_zinvMat.display();
}

void Newton::Make_gVec(Newton::WHICH_DIRECTION direction, InputData &inputData, Solutions &currentPt, Residuals &currentRes, AverageComplementarity &mu, DirectionParameter &beta, Phase &phase, WorkVariables &work, ComputeTime &com) {
    TimeStart(START1);
    // rMessage("mu = " << mu.current);
    // rMessage("beta = " << beta.value);
    mpf_class MMONE = -1.0;
    compute_rMat(direction, mu, beta, currentPt, work);

    TimeEnd(END1);

    com.makerMat += TimeCal(START1, END1);

    TimeStart(START2);
    TimeStart(START_GVEC_MUL);

    // work.DLS1 = R Z^{-1} - X D Z^{-1} = r_zinv - X D Z^{-1}
    if (phase.value == SolveInfo::pFEAS || phase.value == SolveInfo::noINFO) {

        if (direction == CORRECTOR) {
            // x_rd_zinvMat is computed in PREDICTOR step
            Lal::let(work.DLS1, '=', r_zinvMat, '+', x_rd_zinvMat, &MMONE);
        } else {
            // currentPt is infeasilbe, that is the residual
            // dualMat is not 0.
            //      x_rd_zinvMat = X D Z^{-1}
            Jal::ns_jordan_triple_product(x_rd_zinvMat, currentPt.xMat, currentRes.dualMat, currentPt.invzMat, work.DLS2);
            Lal::let(work.DLS1, '=', r_zinvMat, '+', x_rd_zinvMat, &MMONE);
        } // if (direction == CORRECTOR)

    } else {
        // dualMat == 0
        work.DLS1.copyFrom(r_zinvMat);
    }

    //  rMessage("work.DLS1");
    //  work.DLS1.display();

    TimeEnd(END_GVEC_MUL);
    com.makegVecMul += TimeCal(START_GVEC_MUL, END_GVEC_MUL);

    inputData.multi_InnerProductToA(work.DLS1, gVec);
    Lal::let(gVec, '=', gVec, '*', &MMONE);
    // rMessage("gVec =  ");
    // gVec.display();

#if 0
  if (phase.value == SolveInfo:: dFEAS
      || phase.value == SolveInfo::noINFO) {
#endif
    Lal::let(gVec, '=', gVec, '+', currentRes.primalVec);
#if 0
  }
#endif

    TimeEnd(END2);
    com.makegVec += TimeCal(START2, END2);
}

void Newton::calF1(mpf_class &ret, DenseMatrix &G, SparseMatrix &Aj) { Lal::let(ret, '=', Aj, '.', G); }

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: optional `lazy_secs` reports the cost
   of the G = X*F gemm this routine performs on demand. That gemm is invisible to every
   existing timer -- B_PRE covers only the explicit group-boundary precompute -- so the phase
   audit could not say whether Make bMat is gemm-shaped or dot-shaped. NULL disables all
   timing, and the normal path pays one pointer test. See git log. */
void Newton::calF2(mpf_class &ret, DenseMatrix &F, DenseMatrix &G, DenseMatrix &X, SparseMatrix &Aj, bool &hasF2Gcal, double *lazy_secs) {
    int alpha, beta;
    mpf_class value1, value2;

    int n = Aj.nRow;
    // rMessage(" using F2 ");
    switch (Aj.type) {
    case SparseMatrix::SPARSE:
        // rMessage("F2::SPARSE  " << Aj.NonZeroCount);
        ret = 0.0;
        for (int index = 0; index < Aj.NonZeroCount; ++index) {
            alpha = Aj.row_index[index];
            beta = Aj.column_index[index];
            value1 = Aj.sp_ele[index];

            // value2 = F77_FUNC (ddot, DDOT)(&n, &X.de_ele[alpha+n*0], &n,
            //	     &F.de_ele[0+n*beta], &IONE);
            value2 = Rdot(n, X.de_ele + alpha, n, F.de_ele + (n * beta), 1);
            ret += value1 * value2;
            if (alpha != beta) {
                // value2 = F77_FUNC (ddot, DDOT)(&n, &X.de_ele[beta+n*0], &n,
                //        &F.de_ele[0+n*alpha], &IONE);
                value2 = Rdot(n, X.de_ele + beta, n, F.de_ele + (n * alpha), 1);
                ret += value1 * value2;
            }
        }
        break;
    case SparseMatrix::DENSE:
        // G is temporary matrix
        // rMessage("F2::DENSE");
        if (hasF2Gcal == false) {
            // rMessage(" using F2 changing to F1");
            if (lazy_secs != NULL) {
                const double t0 = Time::rGetUseTime();
                Lal::let(G, '=', X, '*', F);
                *lazy_secs += Time::rGetUseTime() - t0;
            } else {
                Lal::let(G, '=', X, '*', F);
            }
            hasF2Gcal = true;
        }
        Lal::let(ret, '=', Aj, '.', G);
        break;
    } // end of switch
}

void Newton::calF3(mpf_class &ret, DenseMatrix &F, DenseMatrix &G, DenseMatrix &X, DenseMatrix &invZ, SparseMatrix &Ai, SparseMatrix &Aj) {
    ret = 0.0;

//#pragma omp parallel
    {
        mpf_class local_ret = 0.0;
        mpf_class sum, value1, value2, plu;

//#pragma omp for nowait
        for (int index1 = 0; index1 < Aj.NonZeroCount; ++index1) {
            int alpha = Aj.row_index[index1];
            int beta = Aj.column_index[index1];
            value1 = Aj.sp_ele[index1];
            sum = 0.0;

            for (int index2 = 0; index2 < Ai.NonZeroCount; ++index2) {
                int gamma = Ai.row_index[index2];
                int delta = Ai.column_index[index2];
                value2 = Ai.sp_ele[index2];

                plu = value2;
                plu *= invZ.de_ele[delta + invZ.nCol * beta];
                plu *= X.de_ele[alpha + X.nCol * gamma];
                sum += plu;

                if (gamma != delta) {
                    plu = value2;
                    plu *= invZ.de_ele[gamma + invZ.nCol * beta];
                    plu *= X.de_ele[alpha + X.nCol * delta];
                    sum += plu;
                }
            }

            plu = value1;
            plu *= sum;
            local_ret += plu;

            if (alpha == beta) {
                continue;
            }

            sum = 0.0;
            for (int index2 = 0; index2 < Ai.NonZeroCount; ++index2) {
                int gamma = Ai.row_index[index2];
                int delta = Ai.column_index[index2];
                value2 = Ai.sp_ele[index2];

                plu = value2;
                plu *= invZ.de_ele[delta + invZ.nCol * alpha];
                plu *= X.de_ele[beta + X.nCol * gamma];
                sum += plu;

                if (gamma != delta) {
                    plu = value2;
                    plu *= invZ.de_ele[gamma + invZ.nCol * alpha];
                    plu *= X.de_ele[beta + X.nCol * delta];
                    sum += plu;
                }
            }
            plu = value1;
            plu *= sum;
            local_ret += plu;
        }
//#pragma omp critical
        ret += local_ret;
    }
}

void Newton::compute_bMat_dense_SDP(InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    int m = currentPt.mDim;
    int SDP_nBlock = inputData.SDP_nBlock;

    for (int l = 0; l < SDP_nBlock; ++l) {
        DenseMatrix &xMat = currentPt.xMat.SDP_block[l];
        DenseMatrix &invzMat = currentPt.invzMat.SDP_block[l];
        DenseMatrix &work1_master = work.DLS1.SDP_block[l];
        DenseMatrix &work2_master = work.DLS2.SDP_block[l];
        const int nConstraint = inputData.SDP_nConstraint[l];

        // ------------------------------------------------------------------
        // Decide whether to run the k1 loop in parallel.
        //
        // Threading is only safe if k1 -> i is injective within this block, because the
        // proof that bMat writes are disjoint relies on each constraint index having a
        // single owner. SDP_constraint[l] is built by appending i once per sub-block of
        // A_i that lands in block l, so a repeat is possible in principle. Check it
        // rather than assume it, and fall back to serial if it does not hold.
        //
        // Also require enough work to be worth a fork/join, in the spirit of SDPB's
        // minimal_split_factor.
        // ------------------------------------------------------------------
        bool injective = true;
#ifdef _OPENMP
        {
            std::vector<char> seen(m, 0);
            for (int k = 0; k < nConstraint; ++k) {
                const int ii = inputData.SDP_constraint[l][k];
                if (ii < 0 || ii >= m || seen[ii]) {
                    injective = false;
                    break;
                }
                seen[ii] = 1;
            }
        }
        bool anyF12 = false;
        for (int k = 0; k < nConstraint; ++k) {
            const FormulaType f = useFormula[inputData.SDP_constraint[l][k] * SDP_nBlock + l];
            if (f == F1 || f == F2) {
                anyF12 = true;
                break;
            }
        }
        // If the block has any F1/F2 constraint AND its setup gemm is big enough for
        // Rgemm to thread well, leave k1 serial and let Rgemm own the parallelism.
        // Threading k1 there makes each blockDim^3 gemm serial inside a thread and gains
        // nothing -- measured 1.35x SLOWER than upstream on gpp124-1 (blockDim 124).
        // The test is on PRESENCE, not count: a single F1 constraint can dominate the
        // block's cost, so a count-based majority test misses it (gpp124-1 has few F1
        // constraints but they carry ~97% of the bMat time). Blocks with no F1/F2 at all
        // (arch0, truss5) have no setup gemm and keep their large k1-threading win.
        const double setup_gemm =
            (double)xMat.nRow * (double)xMat.nRow * (double)xMat.nRow;
        // Let Rgemm own the block, rather than threading k1, under two conditions:
        //   (a) a single setup gemm is big enough for Rgemm to thread well at all, and
        //   (b) there is NOT abundant k1 work relative to the block dimension.
        //
        // (b) is an EMPIRICALLY CALIBRATED PROXY, not a cost model. It compares constraint
        // count against block dimension (nConstraint >= 2*blockDim) and nothing else -- it
        // does not total the setup gemms and weigh them against the k1 x k2 pair work,
        // which is what a real cost comparison would do. It is kept because it is the only
        // rule of four tried that got both ends right, not because it is derived:
        //   theta3    1106 constraints over a 150 block (7.4x) -- wants k1 threading
        //   gpp124-1   125 constraints over a 124 block (1.0x) -- wants Rgemm
        // Measured over 4 runs each, both effects consistent and outside run-to-run noise:
        //   gpp124-1  upstream ~0.542  k1-threaded ~0.706  Rgemm-owned ~0.448
        //   theta3    upstream ~12.6   k1-threaded ~12.2   Rgemm-owned ~13.9
        // Without (b) theta3 would be handed to Rgemm and lose the k1 parallelism it has in
        // abundance (1.18x slower than upstream); without (a) the control* family
        // (blockDim 25-80) would be handed blocks Rgemm cannot thread, forfeiting ~3x.
        // Do not re-derive this from a constraint-type count without new benchmarks.
        const bool enough_k1_work =
            (double)nConstraint >= 2.0 * (double)xMat.nRow;
        const bool rgemm_owns_block = anyF12 && (setup_gemm >= SDPA_OMP_RGEMM_OWNS_BLOCK) &&
                                      !enough_k1_work;
        const bool par = injective && !rgemm_owns_block && nConstraint >= SDPA_OMP_MIN_CONSTRAINTS &&
                         (double)nConstraint * (double)nConstraint * (double)xMat.nRow >= SDPA_OMP_MIN_BMAT_WORK;

        // Cap the team by the work actually available. There are exactly nConstraint k1
        // tasks, so a larger team only pays fork/join cost and privatises scratch for
        // workers that will never be handed a task. This is reachable in practice: the
        // work threshold admits blocks with as few as SDPA_OMP_MIN_CONSTRAINTS (8)
        // constraints, which on a 24-core machine would otherwise start 24 threads.
        int max_threads = omp_get_max_threads();
        if (max_threads > nConstraint)
            max_threads = nConstraint < 1 ? 1 : nConstraint;

        // Then cap so privatising work1/work2 cannot exceed the memory budget.
        // Only relevant when the block actually has F1/F2 constraints; F3 needs no scratch.
        if (anyF12 && max_threads > 1) {
            const double per_thread_mb =
                2.0 * (double)work1_master.nRow * (double)work1_master.nCol *
                sdpa_omp_bytes_per_elem() / 1048576.0;
            if (per_thread_mb > 0.0) {
                const int allowed = 1 + (int)(SDPA_OMP_MAX_PRIV_MB / per_thread_mb);
                if (allowed < max_threads)
                    max_threads = allowed < 1 ? 1 : allowed;
            }
        }
#else
        const bool par = false;
        const bool anyF12 = true;
        (void)injective;
#endif

        double acc_pre = 0.0, acc_f1 = 0.0, acc_f2 = 0.0, acc_f3 = 0.0;

        // The body is a lambda so that the SERIAL path can run without entering any
        // OpenMP construct at all. This matters: "#pragma omp parallel if(false)" still
        // creates a parallel region (a team of one), which makes every inner Rgemm call
        // *nested* -- and nested parallelism is off by default, so Rgemm's own threading
        // would be silently disabled. On gpp124-1 that cost 7.7x in bMat (0.035s -> 0.269s)
        // versus upstream, because the k1 loop did not engage while Rgemm's threading was
        // lost anyway.
        auto run_k1 = [&](int k1, DenseMatrix *w1, DenseMatrix *w2,
                          double &a_pre, double &a_f1, double &a_f2, double &a_f3,
                          bool may_need_priv, bool &owns_priv,
                          DenseMatrix &priv1, DenseMatrix &priv2) {
            // Per-thread scratch. Thread 0 (and the serial case) reuses the existing
            // per-block work matrices, so only the extra threads allocate.
                int i = inputData.SDP_constraint[l][k1];
                int ib = inputData.SDP_blockIndex[l][k1];
                int inz = inputData.A[i].SDP_sp_block[ib].NonZeroEffect;
                SparseMatrix &Ai = inputData.A[i].SDP_sp_block[ib];

                FormulaType formula = useFormula[i * SDP_nBlock + l];

                // Plain locals, not TimeStart/TimeEnd: those macros declare `static
                // double`, which would be shared across threads.
                const double t_start1 = Time::rGetUseTime();
                const double t_start2 = t_start1;

                if (may_need_priv && !owns_priv && (formula == F1 || formula == F2)) {
                    priv1.initialize(work1_master.nRow, work1_master.nCol, work1_master.type);
                    priv2.initialize(work2_master.nRow, work2_master.nCol, work2_master.type);
                    owns_priv = true;
                }
                DenseMatrix &work1 = owns_priv ? priv1 : *w1;
                DenseMatrix &work2 = owns_priv ? priv2 : *w2;

                bool hasF2Gcal = false;
                if (formula == F1) {
                    Lal::let(work1, '=', Ai, '*', invzMat);
                    Lal::let(work2, '=', xMat, '*', work1);
                } else if (formula == F2) {
                    Lal::let(work1, '=', Ai, '*', invzMat);
                    hasF2Gcal = false;
                }
                a_pre += Time::rGetUseTime() - t_start2;

                for (int k2 = 0; k2 < nConstraint; k2++) {
                    int j = inputData.SDP_constraint[l][k2];
                    int jb = inputData.SDP_blockIndex[l][k2];
                    int jnz = inputData.A[j].SDP_sp_block[jb].NonZeroEffect;
                    SparseMatrix &Aj = inputData.A[j].SDP_sp_block[jb];

                    // Select the formula A[i] or the formula A[j].
                    // Use formula that has more NonZeroEffects than others.
                    // We must calculate i==j.
                    // This test is also what makes the bMat writes below disjoint across
                    // k1: it gives each unordered pair {i,j} exactly one owner.
                    if ((inz < jnz) || ((inz == jnz) && (i < j))) {
                        continue;
                    }

                    mpf_class value;
                    switch (formula) {
                    case F1:
                        calF1(value, work2, Aj);
                        break;
                    case F2:
                        calF2(value, work1, work2, xMat, Aj, hasF2Gcal);
                        break;
                    case F3:
                        calF3(value, work1, work2, xMat, invzMat, Ai, Aj);
                        break;
                    } // end of switch
                    // Write the LOWER triangle only (row >= col).
                    //
                    // Every consumer of the dense bMat is Lower-only: Rpotrf("Lower") in
                    // Lal::choleskyFactorWithAdjust, and the two Rtrsv("Lower") in
                    // Lal::solveSystems that the '/' operator dispatches to. The strict
                    // upper half was therefore accumulated on every iteration and never
                    // read. The one routine that would read it, Newton::permuteMat, has
                    // no call sites -- see the note on its definition below.
                    //
                    // Disjointness across k1 is unchanged: the (inz, i) vs (jnz, j) test
                    // above gives each unordered pair {i, j} exactly one owner, and this
                    // writes a strict subset of what that owner wrote before.
                    const int brow = (i > j) ? i : j;
                    const int bcol = (i > j) ? j : i;
                    bMat.de_ele[brow + m * bcol] += value;
                } // end of 'for (int j)'

                const double t = Time::rGetUseTime() - t_start1;
                switch (formula) {
                case F1:
                    a_f1 += t;
                    break;
                case F2:
                    a_f2 += t;
                    break;
                case F3:
                    a_f3 += t;
                    break;
                }
        }; // end of run_k1 lambda

        // Decide AFTER every cap, not before. `par` is computed from the work thresholds,
        // but max_threads is then reduced by the constraint count and the scratch-memory
        // budget, and either can bring it to 1. Entering `omp parallel num_threads(1)`
        // creates a team of one, which is exactly the case the serial path below exists to
        // avoid: it makes any inner Rgemm call nested, and nested parallelism is off by
        // default, so Rgemm's own threading is silently lost. That is most likely to bite
        // large GMP blocks, where the memory cap does reduce the team.
        // !omp_in_parallel() additionally keeps this correct if the routine is ever reached
        // from an enclosing parallel region.
        // The WHOLE decision is inside the guard: max_threads exists only when _OPENMP is
        // defined, so referencing it outside fails to compile in a serial build.
#ifdef _OPENMP
        const bool use_parallel = par && max_threads > 1 && !omp_in_parallel();
#else
        const bool use_parallel = false;
#endif

        if (use_parallel) {
#ifdef _OPENMP
#pragma omp parallel num_threads(max_threads) reduction(+ : acc_pre, acc_f1, acc_f2, acc_f3)
#endif
            {
                // Scratch is only needed by threads other than 0, and only for F1/F2, so
                // allocate LAZILY on the first F1/F2 constraint a thread actually reaches.
                // A block may hold a handful of F1 constraints whose cost rounds to zero;
                // allocating eagerly for every thread then wastes 2*blockDim^2*32 bytes
                // each. On theta3 (blockDim 150, 24 threads) that was +15 MB for nothing.
                DenseMatrix *w1 = &work1_master;
                DenseMatrix *w2 = &work2_master;
                DenseMatrix priv1, priv2;
                bool owns_priv = false;
                bool may_need_priv = false;
#ifdef _OPENMP
                may_need_priv = (omp_get_num_threads() > 1 && omp_get_thread_num() > 0);
#endif
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                for (int k1 = 0; k1 < nConstraint; k1++)
                    run_k1(k1, w1, w2, acc_pre, acc_f1, acc_f2, acc_f3,
                           may_need_priv, owns_priv, priv1, priv2);
                if (owns_priv) {
                    priv1.terminate();
                    priv2.terminate();
                }
            }
        } else {
            // No OpenMP construct at all here, so inner Rgemm/Rdot keep their own threading.
            DenseMatrix priv1, priv2;
            bool owns_priv = false;
            for (int k1 = 0; k1 < nConstraint; k1++)
                run_k1(k1, &work1_master, &work2_master, acc_pre, acc_f1, acc_f2, acc_f3,
                       false, owns_priv, priv1, priv2);
        }

        com.B_PRE += acc_pre;
        com.B_F1 += acc_f1;
        com.B_F2 += acc_f2;
        com.B_F3 += acc_f3;
    }     // end of 'for (int l)'
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: added. See sdpa_newton.h. */
// Structure of the sparse assembly, per block: how the (i,j) pairs group by i, which formula
// each group uses, and how many pairs have a DENSE Aj. Reports nothing unless asked.
//
// The last column is why this exists first. `hasF2Gcal` in compute_bMat_sparse_SDP is declared
// per PAIR and assigned only on a group's first pair, so every later pair reads an
// indeterminate value. It is only CONSUMED when the formula is F2 and Aj is dense, so a
// fixture reaching the defect needs a group with >= 2 such pairs. "f2dense_after_first" counts
// exactly those pairs: the ones whose behaviour is undefined today.
void Newton::census_bMat_sparse_SDP(InputData &inputData, FILE *fp) {
    const int SDP_nBlock = inputData.SDP_nBlock;
    long long tot_pairs = 0, tot_groups = 0, tot_f2dense_after = 0, tot_f2dense_total = 0;
    fprintf(fp, "bmat census: block  pairs groups maxgrp   F1    F2    F3  denseAj  f2dense_after_first\n");
    for (int l = 0; l < SDP_nBlock; ++l) {
        long long pairs = 0, groups = 0, maxgrp = 0, nf[3] = {0, 0, 0};
        long long denseAj = 0, f2dense_after = 0, f2dense_total = 0;
        int previous_i = -1;
        long long cur = 0;
        long long seen_in_group_f2dense = 0;
        for (int iter = 0; iter < SDP_number[l]; ++iter) {
            const int i = SDP_constraint1[l][iter];
            const int j = SDP_constraint2[l][iter];
            const int jb = SDP_blockIndex2[l][iter];
            const FormulaType formula = useFormula[i * SDP_nBlock + l];
            if (i != previous_i) {
                if (cur > maxgrp)
                    maxgrp = cur;
                groups++;
                cur = 0;
                seen_in_group_f2dense = 0;
            }
            cur++;
            pairs++;
            nf[(int)formula]++;
            const bool aj_dense =
                (inputData.A[j].SDP_sp_block[jb].type == SparseMatrix::DENSE);
            if (aj_dense)
                denseAj++;
            if (formula == F2 && aj_dense) {
                f2dense_total++;
                // The first such pair in a group sets hasF2Gcal; any AFTER it reads a value
                // that was never assigned on this iteration of the pair loop.
                if (seen_in_group_f2dense > 0)
                    f2dense_after++;
                seen_in_group_f2dense++;
            }
            previous_i = i;
        }
        if (cur > maxgrp)
            maxgrp = cur;
        fprintf(fp, "bmat census: %5d %6lld %6lld %6lld %4lld %5lld %5lld %8lld %10lld %8lld\n",
                l, pairs, groups, maxgrp, nf[0], nf[1], nf[2], denseAj, f2dense_after,
                f2dense_total);
        tot_pairs += pairs;
        tot_groups += groups;
        tot_f2dense_after += f2dense_after;
        tot_f2dense_total += f2dense_total;
    }
    // Verbose: one line per group, with the inputs the formula heuristic actually used.
    // Reasoning about that heuristic from the source got its direction wrong twice; printing
    // what it decided is cheaper than another guess.
    const char *v = getenv("SDPA_BMAT_ASM_CENSUS_VERBOSE");
    if (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0) {
        for (int l = 0; l < SDP_nBlock; ++l) {
            int previous_i = -1;
            for (int iter = 0; iter < SDP_number[l]; ++iter) {
                const int i = SDP_constraint1[l][iter];
                const int ib = SDP_blockIndex1[l][iter];
                const int j = SDP_constraint2[l][iter];
                const int jb = SDP_blockIndex2[l][iter];
                if (i != previous_i) {
                    const FormulaType f = useFormula[i * SDP_nBlock + l];
                    fprintf(fp, "bmat group : blk %d i=%d formula=%s Ai=%s inz=%d\n", l, i,
                            (f == F1 ? "F1" : (f == F2 ? "F2" : "F3")),
                            (inputData.A[i].SDP_sp_block[ib].type == SparseMatrix::DENSE
                                 ? "DENSE" : "sparse"),
                            inputData.A[i].SDP_sp_block[ib].NonZeroEffect);
                }
                fprintf(fp, "bmat pair  :   blk %d i=%d j=%d Aj=%s\n", l, i, j,
                        (inputData.A[j].SDP_sp_block[jb].type == SparseMatrix::DENSE
                             ? "DENSE" : "sparse"));
                previous_i = i;
            }
        }
    }
    // f2dense_total is the number of pairs that CONSUME hasF2Gcal at all -- formula F2 with a
    // dense Aj. If it is zero everywhere, the uninitialised read is never executed, and the
    // defect is a latent fragility rather than a live wrong answer. That distinction is worth
    // measuring rather than asserting in either direction.
    fprintf(fp, "bmat census: TOTAL pairs=%lld groups=%lld f2dense_after_first=%lld"
                " f2dense_total=%lld\n",
            tot_pairs, tot_groups, tot_f2dense_after, tot_f2dense_total);
    // FLUSH. stdout is block-buffered when piped, so a census printed by a run that is later
    // killed (a sweep under `timeout`, say) is simply lost -- which silently turned a 92
    // problem sweep into a 12 problem one and looked like "most problems do not reach it".
    fflush(fp);
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: configuration for the threaded sparse
   bMat assembly. Every variable is parsed HERE, at one site, and strictly: a malformed value
   is an error, never a silent fallback to the default. The sparse-Cholesky work learned that
   a validator reached only on some code paths is not a validator. See git log. */
enum BmatAsmMode { BMAT_ASM_AUTO, BMAT_ASM_SERIAL, BMAT_ASM_PARALLEL };

struct BmatAsmCfg {
    BmatAsmMode mode;
    uint64_t min_pairs;   // per-block admission gate
    uint64_t scratch_mb;  // budget for EXTRA workers' private matrices; 0 = unbounded
    int team_override;    // 0 = none; TEST HOOK, forces the requested team size
};

/* TEST-ONLY, compile-gated behind its own macro -- not SDPA_SPCHOL_TEST_HOOKS, which belongs
   to a different subsystem. Forces the REQUESTED team after admission, so the in-region
   one-thread branch (no work, fall out to the plain serial loop) can be exercised
   deterministically: no environment setting can compel a runtime to contract a team. A build
   without the macro REFUSES the variable rather than ignoring it. */
static int bmat_asm_team_override() {
    const char *e = getenv("SDPA_BMAT_ASM_TEAM_OVERRIDE");
    if (e == NULL || e[0] == '\0') {
        return 0;
    }
#ifndef SDPA_BMAT_ASM_TEST_HOOKS
    rError("SDPA_BMAT_ASM_TEAM_OVERRIDE is a test hook and this binary was not built with"
           " -DSDPA_BMAT_ASM_TEST_HOOKS, so the hook does not exist here");
    return 0;
#else
    errno = 0;
    char *endp = NULL;
    const long v = strtol(e, &endp, 10);
    if (endp == e || *endp != '\0' || errno == ERANGE || v < 1 || v > INT_MAX) {
        rError("SDPA_BMAT_ASM_TEAM_OVERRIDE must be a positive thread count (got \"" << e
               << "\")");
    }
    return (int)v;
#endif
}

static uint64_t bmat_asm_u64(const char *name, uint64_t dflt) {
    const char *e = getenv(name);
    if (e == NULL) {
        return dflt;
    }
    if (e[0] == '\0') {
        rError(name << " is set but empty; unset it to use the default");
    }
    const char *p = e;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    // strtoull accepts a leading sign and WRAPS a minus, with errno unset -- so "-5" would
    // become an astronomically large gate that silently reads as "never parallel".
    if (*p == '-' || *p == '+') {
        rError(name << " must be a non-negative integer without a sign (got \"" << e << "\")");
    }
    errno = 0;
    char *endp = NULL;
    const unsigned long long v = strtoull(p, &endp, 10);
    if (endp == p || *endp != '\0' || errno == ERANGE) {
        rError(name << " must be a non-negative integer (got \"" << e << "\")");
    }
    return (uint64_t)v;
}

static BmatAsmCfg bmat_asm_cfg() {
    BmatAsmCfg c;
    const char *e = getenv("SDPA_BMAT_ASM_MODE");
    if (e == NULL || e[0] == '\0' || strcmp(e, "auto") == 0) {
        c.mode = BMAT_ASM_AUTO;
    } else if (strcmp(e, "serial") == 0) {
        c.mode = BMAT_ASM_SERIAL;
    } else if (strcmp(e, "parallel") == 0) {
        c.mode = BMAT_ASM_PARALLEL;
    } else {
        rError("SDPA_BMAT_ASM_MODE must be auto, serial or parallel (got \"" << e << "\")");
        c.mode = BMAT_ASM_AUTO;
    }
    c.min_pairs = bmat_asm_u64("SDPA_BMAT_ASM_MIN_PAIRS", 4000);
    c.scratch_mb = bmat_asm_u64("SDPA_BMAT_ASM_SCRATCH_MB", 4096);
    c.team_override = bmat_asm_team_override();
#ifndef _OPENMP
    // A forced-parallel request that a build cannot honour is refused, never downgraded:
    // silently running serial while the caller believes they measured parallel is how wrong
    // configurations become results.
    if (c.mode == BMAT_ASM_PARALLEL) {
        rError("SDPA_BMAT_ASM_MODE=parallel but this binary was built without OpenMP");
    }
#endif
    return c;
}

static bool bmat_asm_log() {
    const char *e = getenv("SDPA_BMAT_ASM_LOG");
    return e != NULL && e[0] != '\0' && strcmp(e, "0") != 0;
}

static bool bmat_asm_profile() {
    const char *e = getenv("SDPA_BMAT_ASM_PROFILE");
    return e != NULL && e[0] != '\0' && strcmp(e, "0") != 0;
}

// Bytes per mpf_class at the configured precision: the object, its limbs, and allocator
// overhead. sdpa_chordal.cpp has the same estimate for the dense bMat, but it is file-local
// there; duplicating four lines is preferable to widening that file's interface for this.
static double bmat_asm_bytes_per_elem() {
    const double limbs =
        (double)((mpf_get_default_prec() + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) + 1.0;
    return (double)sizeof(mpf_class) + limbs * (double)sizeof(mp_limb_t) + 16.0;
}

void Newton::compute_bMat_sparse_SDP(InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    TimeStart(B_NDIAG_START1);
    TimeStart(B_NDIAG_START2);

    /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: EVERY SDPA_BMAT_ASM_* variable is
       parsed HERE, once, on every entry to the sparse assembly -- in both builds, before the
       block loop, before any admission decision. The first version parsed inside the
       OpenMP-only per-block admission, so a no-OpenMP binary, an empty-SDP problem or a path
       with no eligible block silently ignored malformed values. That is the conditional-
       validator defect this project has now hit three times; the rule is one parse site on
       the unconditional path. See git log. */
    const BmatAsmCfg asm_cfg = bmat_asm_cfg();
    double asm_pre_total = 0.0, asm_lazy_total = 0.0;
    const bool asm_prof = bmat_asm_profile();

    // Once per solve, and only when asked: see census_bMat_sparse_SDP.
    {
        static bool census_done = false;
        const char *ce = getenv("SDPA_BMAT_ASM_CENSUS");
        if (!census_done && ce != NULL && ce[0] != '\0' && strcmp(ce, "0") != 0) {
            census_done = true;
            census_bMat_sparse_SDP(inputData, stdout);
            // The census describes STRUCTURE, which is known before any arithmetic. Sweeping
            // a corpus for it should not require solving each problem.
            if (strcmp(ce, "exit") == 0) {
                exit(0);
            }
        }
    }

    for (int l = 0; l < SDP_nBlock; ++l) {
        DenseMatrix &xMat = currentPt.xMat.SDP_block[l];
        DenseMatrix &invzMat = currentPt.invzMat.SDP_block[l];
        DenseMatrix &work1 = work.DLS1.SDP_block[l];
        DenseMatrix &work2 = work.DLS2.SDP_block[l];

        // ------------------------------------------------------------------ group boundaries
        // A GROUP is a maximal run of pairs sharing the same owning constraint i. The pairs
        // are already contiguous in i (make_aggrigateIndex_SDP emits them that way), so the
        // groups are found by one scan. A group is the unit of parallel work because the
        // per-i precompute is shared by all its pairs.
        std::vector<int> gstart;
        {
            int previous_i = -1;
            for (int iter = 0; iter < SDP_number[l]; ++iter) {
                const int i = SDP_constraint1[l][iter];
                if (i != previous_i) {
                    gstart.push_back(iter);
                    previous_i = i;
                }
            }
            gstart.push_back(SDP_number[l]); // sentinel
        }
        const int ngroups = (int)gstart.size() - 1;
        if (ngroups <= 0) {
            continue;
        }

        // One group's whole contribution: its precompute, then each of its pairs.
        //
        // WRITE OWNERSHIP, which is what makes this safe to run concurrently: within a block
        // each pair owns a DISTINCT sparse_bMat location, so two groups never touch the same
        // destination. Across blocks the same location does accumulate once per block, and
        // THAT order is what bit-identity depends on -- which is why the block loop stays
        // serial. See review/MAKE-BMAT-THREADING-PLAN.md.
        //
        // It is a lambda so the SERIAL path can run without entering any OpenMP construct.
        // `#pragma omp parallel if(false)` still creates a team of one, which makes every
        // inner Rgemm nested; nested parallelism is off by default, so Rgemm's own threading
        // would be silently lost. The dense assembly measured that at 7.7x on gpp124-1.
        auto run_group = [&](int g, DenseMatrix &w1, DenseMatrix &w2, double *pre_secs,
                             double *lazy_secs) {
            const int begin = gstart[g], end = gstart[g + 1];
            const int i = SDP_constraint1[l][begin];
            const int ib = SDP_blockIndex1[l][begin];
            SparseMatrix &Ai = inputData.A[i].SDP_sp_block[ib];
            const FormulaType formula = useFormula[i * SDP_nBlock + l];

            // Group state: has G = X*F been computed for THIS i yet. Per group, never per
            // pair, and never shared between groups. Upstream declared this UNINITIALISED per
            // pair, assigning it only on a group's first pair -- an indeterminate read on
            // every later F2 pair with a dense Aj. At the DEFAULT Kappa that consuming pair
            // is structurally impossible (a dense Aj's owner is dense, and a dense owner
            // takes F1 whenever n^2 < Kappa*up, automatic for Kappa >= 1 since up >= n*n;
            // measured zero across all 93 SDPLIB problems). But Kappa is a command-line flag
            // (-k), and below 1/(tied dense constraints) the highest-index dense constraint
            // takes F2 with the others as dense-Aj pairs in its group: tests/f2dense.dat-s
            // reaches exactly that under -k 0.3, and CI runs it. The old read was therefore
            // reachable under a supported configuration; this is a live-path fix, not
            // hygiene. (An earlier comment here claimed unconditional unreachability --
            // review supplied the -k counterexample.)
            bool hasF2Gcal = false;

            if (formula == F1 || formula == F2) {
                const double t0 = (pre_secs != NULL) ? Time::rGetUseTime() : 0.0;
                if (formula == F1) {
                    Lal::let(w1, '=', Ai, '*', invzMat);
                    Lal::let(w2, '=', xMat, '*', w1);
                } else {
                    Lal::let(w1, '=', Ai, '*', invzMat);
                }
                if (pre_secs != NULL) {
                    *pre_secs += Time::rGetUseTime() - t0;
                }
            }

            for (int iter = begin; iter < end; ++iter) {
                const int j = SDP_constraint2[l][iter];
                const int jb = SDP_blockIndex2[l][iter];
                SparseMatrix &Aj = inputData.A[j].SDP_sp_block[jb];

                mpf_class value;
                switch (formula) {
                case F1:
                    calF1(value, w2, Aj);
                    break;
                case F2:
                    calF2(value, w1, w2, xMat, Aj, hasF2Gcal, lazy_secs);
                    break;
                case F3:
                    calF3(value, w1, w2, xMat, invzMat, Ai, Aj);
                    break;
                }
                sparse_bMat.sp_ele[SDP_location_sparse_bMat[l][iter]] += value;
            }
        };

        // ------------------------------------------------------------------ admission
        bool want_parallel = false;
        int team = 1;
#ifdef _OPENMP
        const BmatAsmCfg &cfg = asm_cfg;
        const char *why = NULL;
        if (cfg.mode == BMAT_ASM_SERIAL) {
            why = "SDPA_BMAT_ASM_MODE=serial";
        } else if (omp_get_level() != 0) {
            why = "nested inside another OpenMP region";
        } else if (ngroups < 2) {
            why = "fewer than two groups in this block";
        } else {
            team = omp_get_max_threads();
            const int tl = omp_get_thread_limit();
            if (tl > 0 && tl < team) {
                team = tl;
            }
            if (team > ngroups) {
                team = ngroups; // never more workers than groups
            }
            // Scratch budget: each EXTRA worker needs two block-sized dense matrices. Bound
            // the team by that rather than discovering the cost as RSS.
            // NO double-to-int conversion here. The first version computed
            // `1 + (int)(budget / per_worker)`, and a large but valid SDPA_BMAT_ASM_SCRATCH_MB
            // put that cast outside int's range -- undefined behaviour on a published
            // configuration path, which in practice collapsed the team to one and silently
            // serialised every block. Review predicted it; the CI control then demonstrated
            // it. The team is already small, so shrink it while the extra workers' need
            // exceeds the budget.
            if (cfg.scratch_mb > 0) {
                const double per_worker_mb = 2.0 * (double)xMat.nRow * (double)xMat.nCol *
                                             bmat_asm_bytes_per_elem() / 1048576.0;
                if (per_worker_mb > 0.0) {
                    while (team > 1 &&
                           (double)(team - 1) * per_worker_mb > (double)cfg.scratch_mb) {
                        team--;
                    }
                }
            }
            if (team < 2) {
                why = "team would be one (threads, thread limit, groups or scratch budget)";
            } else if (cfg.mode != BMAT_ASM_PARALLEL &&
                       (uint64_t)SDP_number[l] < cfg.min_pairs) {
                why = "block below SDPA_BMAT_ASM_MIN_PAIRS";
            } else {
                want_parallel = true;
            }
        }
        if (bmat_asm_log()) {
            printf("bmat asm block %d: %d groups, %d pairs -> %s%s%s\n", l, ngroups,
                   SDP_number[l], want_parallel ? "parallel" : "serial", why ? " (" : "",
                   why ? why : "");
        }
#endif

        double pre_secs = 0.0, lazy_secs = 0.0;
        const bool prof = asm_prof;

#ifdef _OPENMP
        if (want_parallel) {
            bool one_thread = false;
            int actual_team = 1;
            // The test override applies HERE, after admission -- forcing the REQUEST the
            // region is created with. Applied before admission it just trips the team<2
            // rejection, which is a different branch: that is pre-admission fallback, and
            // what needs deterministic coverage is the region that was admitted and then
            // finds itself with one member. (Its first placement made the control test the
            // wrong branch, and the control caught it.)
            if (cfg.team_override > 0) {
                team = cfg.team_override;
            }
            std::vector<double> t_pre(team, 0.0), t_lazy(team, 0.0);
            // Per-thread WORK counters, because a route line printed at admission proves a
            // decision, not execution: the runtime can still hand the region one thread, and
            // "-> parallel" would read as concurrency that never happened.
            std::vector<long long> t_groups(team, 0), t_pairs(team, 0);
#pragma omp parallel num_threads(team)
            {
                // THE team, read inside the region that owns it. num_threads is a REQUEST;
                // if the runtime returns one thread we do NO work here and fall out to the
                // untouched serial loop OUTSIDE every construct -- running the serial body
                // inside a team of one would make each inner Rgemm nested and silently lose
                // its threading, which is the failure this design exists to avoid.
                if (omp_get_num_threads() < 2) {
#pragma omp single
                    {
                        one_thread = true;
                        actual_team = omp_get_num_threads();
                    }
                } else {
                    const int tid = omp_get_thread_num();
#pragma omp single
                    { actual_team = omp_get_num_threads(); }
                    // Worker 0 reuses the block's existing scratch; only the extra workers
                    // allocate, and only the two matrices they actually need.
                    DenseMatrix priv1, priv2;
                    bool owns_priv = false;
                    DenseMatrix *w1 = &work1, *w2 = &work2;
                    if (tid != 0) {
                        priv1.initialize(xMat.nRow, xMat.nCol, DenseMatrix::DENSE);
                        priv2.initialize(xMat.nRow, xMat.nCol, DenseMatrix::DENSE);
                        owns_priv = true;
                        w1 = &priv1;
                        w2 = &priv2;
                    }
#pragma omp for schedule(dynamic)
                    for (int g = 0; g < ngroups; ++g) {
                        run_group(g, *w1, *w2, prof ? &t_pre[tid] : NULL,
                                  prof ? &t_lazy[tid] : NULL);
                        t_groups[tid]++;
                        t_pairs[tid] += (long long)(gstart[g + 1] - gstart[g]);
                    }
                    if (owns_priv) {
                        priv1.terminate();
                        priv2.terminate();
                    }
                }
            }
            if (one_thread) {
                want_parallel = false; // fall through to the serial loop below
                if (bmat_asm_log()) {
                    printf("bmat asm block %d: runtime gave one thread; ran serially\n", l);
                }
            } else {
                int workers = 0;
                long long gsum = 0, psum = 0;
                for (int t = 0; t < team; ++t) {
                    pre_secs += t_pre[t];
                    lazy_secs += t_lazy[t];
                    gsum += t_groups[t];
                    psum += t_pairs[t];
                    if (t_pairs[t] > 0) {
                        workers++;
                    }
                }
                // An actual cross-check, not a printout: every group and every pair must
                // have executed exactly once. A worksharing bug that dropped or doubled a
                // group would otherwise surface only as a wrong answer downstream.
                if (gsum != (long long)ngroups || psum != (long long)SDP_number[l]) {
                    rError("compute_bMat_sparse_SDP: block " << l << " executed " << gsum
                           << " of " << ngroups << " groups and " << psum << " of "
                           << SDP_number[l] << " pairs; the parallel region lost or doubled"
                           << " work");
                }
                if (bmat_asm_log()) {
                    printf("bmat asm done  %d: requested %d actual %d workers %d groups"
                           " %lld/%d pairs %lld/%d\n",
                           l, team, actual_team, workers, gsum, ngroups, psum, SDP_number[l]);
                }
            }
        }
#endif

        if (!want_parallel) {
            for (int g = 0; g < ngroups; ++g) {
                run_group(g, work1, work2, prof ? &pre_secs : NULL, prof ? &lazy_secs : NULL);
            }
        }

        com.B_PRE += pre_secs;
        asm_pre_total += pre_secs;
        asm_lazy_total += lazy_secs;
    } // end of 'for (int l)'

    if (asm_prof) {
        // Per solve, printed here rather than accumulated in a file-static: a cumulative
        // static survives across solves and would misattribute one run's cost to another.
        printf("bmat asm profile : explicit pre %.6f s, lazy F2 gemm %.6f s\n", asm_pre_total,
               asm_lazy_total);
    }
}

#if 0
void Newton::compute_bMat_dense_SCOP(InputData& inputData,
				     Solutions& currentPt,
				     WorkVariables& work,
				     ComputeTime& com)
{
    rError("current version does not support SOCP");
}

void Newton::compute_bMat_sparse_SOCP(InputData& inputData,
				      Solutions& currentPt,
				      WorkVariables& work,
				      ComputeTime& com)
{
    rError("current version does not support SOCP");
}
#endif

void Newton::compute_bMat_dense_LP(InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    int m = currentPt.mDim;
    int LP_nBlock = inputData.LP_nBlock;

    TimeEnd(B_DIAG_START1);
    for (int l = 0; l < LP_nBlock; ++l) {
        mpf_class xMat = currentPt.xMat.LP_block[l];
        mpf_class invzMat = currentPt.invzMat.LP_block[l];

        for (int k1 = 0; k1 < inputData.LP_nConstraint[l]; k1++) {
            int i = inputData.LP_constraint[l][k1];
            int ib = inputData.LP_blockIndex[l][k1];
            //	int inz = inputData.A[i].LP_sp_block[ib].NonZeroEffect;
            mpf_class Ai = inputData.A[i].LP_sp_block[ib];

            for (int k2 = k1; k2 < inputData.LP_nConstraint[l]; k2++) {
                int j = inputData.LP_constraint[l][k2];
                int jb = inputData.LP_blockIndex[l][k2];
                //	  int jnz = inputData.A[j].LP_sp_block[jb].NonZeroEffect;
                mpf_class Aj = inputData.A[j].LP_sp_block[jb];

                mpf_class value;
                value = xMat * invzMat * Ai * Aj;

                // Lower triangle only -- see compute_bMat_dense_SDP above.
                const int brow = (i > j) ? i : j;
                const int bcol = (i > j) ? j : i;
                bMat.de_ele[brow + m * bcol] += value;
            } // end of 'for (int j)'
        }     // end of 'for (int i)'
    }         // end of 'for (int l)'
    TimeEnd(B_DIAG_END1);
    com.B_DIAG += TimeCal(B_DIAG_START1, B_DIAG_END1);
}

void Newton::compute_bMat_sparse_LP(InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    TimeEnd(B_DIAG_START1);
    for (int l = 0; l < LP_nBlock; ++l) {
        mpf_class xMat = currentPt.xMat.LP_block[l];
        mpf_class invzMat = currentPt.invzMat.LP_block[l];

        for (int iter = 0; iter < LP_number[l]; iter++) {
            int i = LP_constraint1[l][iter];
            int ib = LP_blockIndex1[l][iter];
            mpf_class Ai = inputData.A[i].LP_sp_block[ib];

            int j = LP_constraint2[l][iter];
            int jb = LP_blockIndex2[l][iter];
            mpf_class Aj = inputData.A[j].LP_sp_block[jb];

            mpf_class value;
            value = xMat * invzMat * Ai * Aj;
            sparse_bMat.sp_ele[LP_location_sparse_bMat[l][iter]] += value;
        } // end of 'for (int iter)
    }     // end of 'for (int l)'
    TimeEnd(B_DIAG_END1);
    com.B_DIAG += TimeCal(B_DIAG_START1, B_DIAG_END1);
}

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: added. See Make_bMat. */
void Newton::emit_bMat_stream() {
    const char *dumpf = getenv("SDPA_BMAT_ASM_DUMP");
    const char *want = getenv("SDPA_BMAT_ASM_DIGEST");
    const bool digest = (want != NULL && want[0] != '\0' && strcmp(want, "0") != 0);
    if (dumpf != NULL && dumpf[0] == '\0')
        dumpf = NULL;
    if (!digest && dumpf == NULL) {
        return;
    }
    FILE *dump = NULL;
    if (dumpf != NULL) {
        dump = fopen(dumpf, "ab"); // appends: one file holds a whole solve, in order
        if (dump == NULL) {
            rError("SDPA_BMAT_ASM_DUMP: cannot open \"" << dumpf << "\" for append");
        }
    }
    // Same serialiser as the factor, different tag -- an assembly stream can never compare
    // equal to a factor stream by accident.
    CanonicalStream c =
        canonicalSparseStream(sparse_bMat, diagonalIndex, sparse_bMat.nRow, "BMATASMv1", dump);
    if (dump != NULL) {
        if (fflush(dump) != 0 || ferror(dump) != 0)
            c.io_error = true;
        if (fclose(dump) != 0)
            c.io_error = true;
    }
    if (c.io_error) {
        rError("SDPA_BMAT_ASM_DUMP: writing \"" << dumpf << "\" failed after "
               << (unsigned long long)c.bytes << " bytes. The dump is truncated, so comparing"
               << " against it would be meaningless; failing rather than leaving a file that"
               << " looks complete.");
    }
    if (digest) {
        printf("bmat assembled   : %d rows, %llu records, %llu stream bytes,"
               " fingerprint %016llx\n",
               sparse_bMat.nRow, (unsigned long long)c.records, (unsigned long long)c.bytes,
               (unsigned long long)c.fnv);
    }
}

void Newton::Make_bMat(InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    TimeStart(START3);
    if (bMat_type == SPARSE) {
        // set sparse_bMat zero
        for (int iter = 0; iter < sparse_bMat.NonZeroCount; ++iter) {
            sparse_bMat.sp_ele[iter] = 0.0;
        }
        compute_bMat_sparse_SDP(inputData, currentPt, work, com);
        //   compute_bMat_sparse_SOCP(inputData,currentPt,work,com);
        compute_bMat_sparse_LP(inputData, currentPt, work, com);
        /* MODIFIED from upstream (GPLv2 2a notice), 2026-08-14: emit the ASSEMBLED matrix,
           here -- after every SDP and LP contribution and BEFORE it is factored.
           This is the primary oracle for any change to the assembly. The factor stream is
           not a substitute: finite-precision Cholesky is not injective, so two different
           assembled matrices can round to the same factor, and "the factor matched" would
           then hide a real difference. Off unless asked for. See git log and
           review/MAKE-BMAT-THREADING-PLAN.md. */
        emit_bMat_stream();
    } else {
        // Keep this a FULL-matrix zero. Only the lower triangle is written below,
        // but leaving the strict upper half uninitialised would put indeterminate
        // values in a live allocation for no measurable gain.
        bMat.setZero();
        compute_bMat_dense_SDP(inputData, currentPt, work, com);
        //    compute_bMat_dense_SOCP(inputData,currentPt,work,com);
        compute_bMat_dense_LP(inputData, currentPt, work, com);
    }
    // rMessage("bMat =  ");
    // bMat.display();
    // sparse_bMat.display();
    TimeEnd(END3);
    com.makebMat += TimeCal(START3, END3);
}

// nakata 2004/12/01
// WARNING: permuteMat is NOT CALLED anywhere in this tree. A whole-tree grep finds
// only this definition and the declaration in sdpa_newton.h, and the method is not
// virtual, so it cannot be reached indirectly either. It is also the only routine that
// would read the dense bMat's strict UPPER triangle: it copies arbitrary (i, j) chosen
// by ordering[]. Since 2026-08-05 the dense bMat is accumulated in its lower triangle
// only, so the strict upper half holds whatever bMat.setZero() left there, i.e. zero.
// A future caller must either mirror the lower half up first, or index with row >= col.
void Newton::permuteMat(DenseMatrix &bMat, SparseMatrix &sparse_bMat) {
    int i, j, k;
    int mDIM = bMat.nRow;

    for (k = 0; k < sparse_bMat.NonZeroCount; k++) {
        i = ordering[sparse_bMat.row_index[k]];
        j = ordering[sparse_bMat.column_index[k]];
        sparse_bMat.sp_ele[k] = bMat.de_ele[i + j * mDIM];
    }
}

// nakata 2004/12/01
void Newton::permuteVec(Vector &gVec, Vector &gVec2) {
    int i, k;
    int mDIM = gVec2.nDim;

    for (k = 0; k < mDIM; k++) {
        i = ordering[k];
        gVec2.ele[k] = gVec.ele[i];
    }
}

// nakata 2004/12/01
void Newton::reverse_permuteVec(Vector &DyVec2, Vector &DyVec) {
    int i, k;
    int mDIM = DyVec.nDim;

    for (k = 0; k < mDIM; k++) {
        i = ordering[k];
        DyVec.ele[i] = DyVec2.ele[k];
    }
}

bool Newton::compute_DyVec(Newton::WHICH_DIRECTION direction, InputData &inputData, Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    if (direction == PREDICTOR) {
        TimeStart(START3_2);

        if (bMat_type == SPARSE) {
            bool ret = Lal::getCholesky(sparse_bMat, diagonalIndex);
            if (ret == FAILURE) {
                return FAILURE;
            }
        } else {
            bool ret = Lal::choleskyFactorWithAdjust(bMat);
            if (ret == FAILURE) {
                return FAILURE;
            }
        }
        // rMessage("Cholesky of bMat =  ");
        // bMat.display();
        // sparse_bMat.display();
        TimeEnd(END3_2);
        com.choleskybMat += TimeCal(START3_2, END3_2);
    }
    // bMat is already cholesky factorized.

    TimeStart(START4);
    if (bMat_type == SPARSE) {
        permuteVec(gVec, work.DV1);
        Lal::let(work.DV2, '=', sparse_bMat, '/', work.DV1);
        reverse_permuteVec(work.DV2, DyVec);
    } else {
        Lal::let(DyVec, '=', bMat, '/', gVec);
    }
    TimeEnd(END4);
    com.solve += TimeCal(START4, END4);
    // rMessage("DyVec =  ");
    // DyVec.display();
    return _SUCCESS;
}

void Newton::compute_DzMat(InputData &inputData, Residuals &currentRes, Phase &phase, ComputeTime &com) {
    TimeStart(START_SUMDZ);
    mpf_class MMONE = -1.0;
    inputData.multi_plusToA(DyVec, DzMat);
    Lal::let(DzMat, '=', DzMat, '*', &MMONE);
    if (phase.value == SolveInfo::pFEAS || phase.value == SolveInfo::noINFO) {
        Lal::let(DzMat, '=', DzMat, '+', currentRes.dualMat);
    }
    TimeEnd(END_SUMDZ);
    com.sumDz += TimeCal(START_SUMDZ, END_SUMDZ);
}

void Newton::compute_DxMat(Solutions &currentPt, WorkVariables &work, ComputeTime &com) {
    TimeStart(START_DX);
    mpf_class MMONE = -1.0;
    // work.DLS1 = dX dZ Z^{-1}
    Jal::ns_jordan_triple_product(work.DLS1, currentPt.xMat, DzMat, currentPt.invzMat, work.DLS2);
    // dX = R Z^{-1} - dX dZ Z^{-1}
    Lal::let(DxMat, '=', r_zinvMat, '+', work.DLS1, &MMONE);
    TimeEnd(END_DX);
    TimeStart(START_SYMM);
    Lal::getSymmetrize(DxMat);
    TimeEnd(END_SYMM);
    // rMessage("DxMat =  ");
    // DxMat.display();
    com.makedX += TimeCal(START_DX, END_DX);
    com.symmetriseDx += TimeCal(START_SYMM, END_SYMM);
}

bool Newton::Mehrotra(Newton::WHICH_DIRECTION direction, InputData &inputData, Solutions &currentPt, Residuals &currentRes, AverageComplementarity &mu, DirectionParameter &beta, Switch &reduction, Phase &phase, WorkVariables &work, ComputeTime &com) {
    //   rMessage("xMat, yVec, zMat =  ");
    //   currentPt.xMat.display();
    //   currentPt.yVec.display();
    //   currentPt.zMat.display();

    Make_gVec(direction, inputData, currentPt, currentRes, mu, beta, phase, work, com);

    if (direction == PREDICTOR) {
        Make_bMat(inputData, currentPt, work, com);
    }

    // rMessage("gVec, bMat =  ");
    //   gVec.display();
    //   bMat.display();
    //   sparse_bMat.display();  //
    //   display_sparse_bMat();  // with reverse ordering

    bool ret = compute_DyVec(direction, inputData, currentPt, work, com);
    if (ret == FAILURE) {
        return FAILURE;
    }
    //  rMessage("cholesky factorization =  ");
    //  sparse_bMat.display();

    TimeStart(START5);

    compute_DzMat(inputData, currentRes, phase, com);
    compute_DxMat(currentPt, work, com);

    TimeEnd(END5);
    com.makedXdZ += TimeCal(START5, END5);

    // rMessage("DxMat, DyVec, DzMat =  ");
    //   DxMat.display();
    //   DyVec.display();
    //   DzMat.display();

    return true;
}

void Newton::display(FILE *fpout) {
    if (fpout == NULL) {
        return;
    }

    fprintf(fpout, "rNewton.DxMat = \n");
    DxMat.display(fpout);
    fprintf(fpout, "rNewton.DyVec = \n");
    DyVec.display(fpout);
    fprintf(fpout, "rNewton.DzMat = \n");
    DzMat.display(fpout);
}

void Newton::display_index(FILE *fpout) {
    if (fpout == NULL) {
        return;
    }
    printf("display_index: %d %d %d\n", SDP_nBlock, SOCP_nBlock, LP_nBlock);

    for (int b = 0; b < SDP_nBlock; b++) {
        printf("SDP:%dth block\n", b);
        for (int i = 0; i < SDP_number[b]; i++) {
            printf("cons1:%d const2:%d block1:%d block2:%d sp_bMat:%d \n", SDP_constraint1[b][i], SDP_constraint2[b][i], SDP_blockIndex1[b][i], SDP_blockIndex2[b][i], SDP_location_sparse_bMat[b][i]);
        }
    }

    for (int b = 0; b < SOCP_nBlock; b++) {
        printf("SOCP:%dth block\n", b);
        for (int i = 0; i < SOCP_number[b]; i++) {
            printf("cons1:%d const2:%d block1:%d block2:%d sp_bMat:%d \n", SOCP_constraint1[b][i], SOCP_constraint2[b][i], SOCP_blockIndex1[b][i], SOCP_blockIndex2[b][i], SOCP_location_sparse_bMat[b][i]);
        }
    }

    for (int b = 0; b < LP_nBlock; b++) {
        printf("LP:%dth block\n", b);
        for (int i = 0; i < LP_number[b]; i++) {
            printf("cons1:%d const2:%d block1:%d block2:%d sp_bMat:%d \n", LP_constraint1[b][i], LP_constraint2[b][i], LP_blockIndex1[b][i], LP_blockIndex2[b][i], LP_location_sparse_bMat[b][i]);
        }
    }
}

void Newton::display_sparse_bMat(FILE *fpout) {
    if (fpout == NULL) {
        return;
    }
    fprintf(fpout, "{");
    for (int index = 0; index < sparse_bMat.NonZeroCount; ++index) {
        int i = sparse_bMat.row_index[index];
        int j = sparse_bMat.column_index[index];
        mpf_class value = sparse_bMat.sp_ele[index];
        int ii = ordering[i];
        int jj = ordering[j];
        gmp_fprintf(fpout, "val[%d,%d] = %Fe\n", ii, jj, value.get_mpf_t());
    }
    fprintf(fpout, "}\n");
}

} // namespace sdpa
