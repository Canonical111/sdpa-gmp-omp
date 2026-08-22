#!/usr/bin/env python3
# MODIFIED from upstream (GPLv2 2a notice), 2026-08-18: NEW FILE.
#
# Generator for the bMat gate-boundary fixtures used by tests/bmat_gate_fixtures.sh.
#
# Each fixture is a block-diagonal SDP whose aggregate sparsity pattern -- which (i,j)
# constraint pairs share a block -- is EXACTLY computable by hand: constraint i touches only
# its own block, so the aggregate count is sum(k_b^2) over block sizes k_b (symmetric pattern
# including the diagonal), and because a block-diagonal pattern is already chordal, the
# ordered fill EQUALS the aggregate. That makes these fixtures sharp on both sides of every
# gate: the expected ratios below are arithmetic, not measurements, and the fill==aggregate
# equality also exercises the boundary case of the fill>=aggregate invariant.
#
# Fixture map (m = 120 constraints unless stated; density = sum(k^2)/m^2):
#   g1_smallblocks : 4 blocks of 30            -> nBlock=4 <= 5: gate 1 DENSE, both policies
#   g2_bigblock    : sizes 70,10,10,10,10,10   -> max 70 > 0.5m=60: gate 2 DENSE, both
#   low_sparse     : sizes 30,30,15,15,15,15   -> density 0.1875 < 0.25: SPARSE, both
#   switch         : sizes 46,46,7,7,7,7       -> density 0.3075: auto gate-3 DENSE,
#                                                 fill gate-4 SPARSE  (the population the
#                                                 recalibration is about)
#   high_dense     : sizes 60,48,3,3,3,3       -> density 0.4125 > 0.40: DENSE both -- auto
#                                                 via gate 3 at 0.25, fill via the gate-3'
#                                                 skip at F; also puts gate 2 EXACTLY at its
#                                                 cutoff (60 > 60 is false), pinning the
#                                                 strict inequality
#
# The SDPs are trivial (diagonal constraints, identity objective); the tests run them with
# maxIteration=1 and read only the chooser's decision log, which prints before iteration 1.
import os
import sys

FIXTURES = {
    "g1_smallblocks": [30, 30, 30, 30],
    "g2_bigblock": [70, 10, 10, 10, 10, 10],
    "low_sparse": [30, 30, 15, 15, 15, 15],
    "switch": [46, 46, 7, 7, 7, 7],
    "high_dense": [60, 48, 3, 3, 3, 3],
    # Exact-boundary controls (review of 2026-08-18, point 2):
    #   boundary_025: sum(k^2) = 3600 = 0.25 * 120^2 EXACTLY -- auto's gate 3 is a strict >,
    #                 so auto must pass and route SPARSE; pins the strict inequality at 0.25.
    "boundary_025": [40, 40, 10, 10, 10, 10],
    #   boundary_040: sum(k^2) = 5760 = 0.40 * 120^2 EXACTLY (nBlock=7) -- pins BOTH strict
    #                 inequalities at F on the fill path (gate-3' skip must not fire, gate 4
    #                 must not fire) while auto's gate 3 at 0.25 still says DENSE: the two
    #                 policies diverge exactly on the boundary.
    "boundary_040": [60, 46, 4, 4, 2, 2, 2],
    #   gate 1 edges: m = 100 is <= and must be DENSE; m = 101 with the same shape must pass.
    #   nBlock = 5 is <= and must be DENSE even though the density (0.2) is below every
    #   threshold.
    "g1_m100": [20, 16, 16, 16, 16, 16],
    "g1_m101": [21, 16, 16, 16, 16, 16],
    "g1_nb5": [24, 24, 24, 24, 24],
}

# The nonchordal RING fixture: k groups of g constraints in a cycle; block j (dim 2g) joins
# groups j and j+1 (mod k). Aggregate: each constraint shares a block with groups j-1, j, j+1,
# so its row holds exactly 3g entries -> aggregate count = 3g*m, here 3*15*120 = 5400 = 0.375.
# Fill: eliminating a cycle of cliques must add chords -- at group level a cycle C_k needs at
# least k-3 chords whatever the ordering, each chord connecting two g-groups = 2*g*g counted
# entries, so fill count >= 5400 + (8-3)*2*225 = 7650 = 0.531 > 0.40. Both bounds are
# arithmetic: aggregate exactly 0.375 (< F, so the gate-3' skip must NOT fire) and fill
# provably > F for EVERY ordering -- this is the fixture that reaches the fill policy's
# gate-4 DENSE branch, and its fill > aggregate strictly (the block-diagonal fixtures pin the
# equality case of the invariant).
RING_K, RING_G = 8, 15


def write_fixture(name, sizes, outdir):
    m = sum(sizes)
    n_block = len(sizes)
    lines = []
    lines.append(f"{m} = mDIM")
    lines.append(f"{n_block} = nBLOCK")
    lines.append(" ".join(str(k) for k in sizes) + " = bLOCKsTRUCT")
    lines.append(" ".join(["1.0"] * m))
    # F0 = C: identity in every block (row "0 blk r r 1.0")
    for b, k in enumerate(sizes, start=1):
        for r in range(1, k + 1):
            lines.append(f"0 {b} {r} {r} 1.0")
    # F_i: constraint i is E_rr in its own block -- constraint/block incidence is exactly
    # the block-diagonal pattern the docstring's arithmetic assumes.
    i = 1
    for b, k in enumerate(sizes, start=1):
        for r in range(1, k + 1):
            lines.append(f"{i} {b} {r} {r} 1.0")
            i += 1
    path = f"{outdir}/bmat_gate_{name}.dat-s"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    agg = sum(k * k for k in sizes)
    print(f"{name}: m={m} nBlock={n_block} agg={agg} density={agg / (m * m):.6f} -> {path}")


def write_ring(outdir, k=RING_K, g=RING_G):
    m = k * g
    lines = []
    lines.append(f"{m} = mDIM")
    lines.append(f"{k} = nBLOCK")
    lines.append(" ".join(str(2 * g) for _ in range(k)) + " = bLOCKsTRUCT")
    lines.append(" ".join(["1.0"] * m))
    for b in range(1, k + 1):
        for r in range(1, 2 * g + 1):
            lines.append(f"0 {b} {r} {r} 1.0")
    # constraint i of group j (0-based) sits in block j+1 (slots 1..g) and block j (slots
    # g+1..2g), blocks 1-based, group k-1 wrapping into block k and block k-1 accordingly.
    for j in range(k):
        for t in range(g):
            i = j * g + t + 1
            b_lead = j + 1                      # block covering groups j (first half) ...
            b_trail = ((j - 1) % k) + 1         # ... and the block where group j is second half
            lines.append(f"{i} {b_lead} {t + 1} {t + 1} 1.0")
            lines.append(f"{i} {b_trail} {g + t + 1} {g + t + 1} 1.0")
    path = f"{outdir}/bmat_gate_ring.dat-s"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    agg = 3 * g * m
    print(f"ring: m={m} nBlock={k} agg={agg} density={agg / (m * m):.6f} "
          f"fill>=+{(k - 3) * 2 * g * g} -> {path}")


def write_lp_low_fixture(outdir):
    # gate 2's LP loop at EXACTLY its cutoff: one scalar LP variable touched by exactly 60 of
    # 120 constraints (60 > 60 is false -- the strict inequality is the assertion), the other
    # 60 in six 10-dim SDP blocks. The 60-clique makes aggregate = 60^2 + 6*10^2 = 4200 =
    # 0.291667: auto trips gate 3 (DENSE) while fill passes to gate 4 (clique + singletons is
    # chordal, fill == aggregate = 4200 < 5760) and routes SPARSE -- so this fixture pins the
    # gate-2 boundary AND is one more auto/fill divergence point.
    # Each toucher also owns a PRIVATE 1x1 SDP block: with only the shared scalar, the 60
    # touchers' bMat rows are a rank-one block and the SPARSE route dies before iteration 1
    # (rc=2) while the dense route's Cholesky-with-adjust shrugs it off -- the first version
    # of this fixture failed exactly there. The private blocks make bMat nonsingular and add
    # NOTHING to the aggregate count: a toucher's row is still its 60-clique (its own
    # diagonal included), so agg stays 60^2 + 6*10^2 = 4200.
    sdp = [10] * 6
    lp_touchers = 60
    m = sum(sdp) + lp_touchers
    nblock = len(sdp) + lp_touchers + 1
    lines = [f"{m} = mDIM", f"{nblock} = nBLOCK",
             " ".join(str(k) for k in sdp) + " " + " ".join(["1"] * lp_touchers) + " -1 = bLOCKsTRUCT",
             " ".join(["1.0"] * m)]
    for b, k in enumerate(sdp, start=1):
        for r in range(1, k + 1):
            lines.append(f"0 {b} {r} {r} 1.0")
    for t in range(lp_touchers):
        lines.append(f"0 {len(sdp) + 1 + t} 1 1 1.0")
    lines.append(f"0 {nblock} 1 1 1.0")
    for i in range(1, lp_touchers + 1):
        lines.append(f"{i} {nblock} 1 1 {float(i)}")
        lines.append(f"{i} {len(sdp) + i} 1 1 1.0")
    i = lp_touchers + 1
    for b, k in enumerate(sdp, start=1):
        for r in range(1, k + 1):
            lines.append(f"{i} {b} {r} {r} 1.0")
            i += 1
    path = f"{outdir}/bmat_gate_g2_lp_low.dat-s"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    agg = lp_touchers * lp_touchers + sum(k * k for k in sdp)
    print(f"g2_lp_low: m={m} nBlock={nblock} agg={agg} density={agg / (m * m):.6f} -> {path}")


def write_lp_fixture(outdir):
    # gate 2's LP loop. The first version of this fixture gave 70 constraints their own
    # diagonal slot in one dim-70 LP block and expected gate 2 to fire -- it routed SPARSE,
    # because SDPA stores an LP block of dimension d as d SCALAR blocks internally, each
    # touched here by exactly one constraint (LP_nConstraint = 1 per scalar). The lesson is
    # kept in this comment because the fixture exists to pin exactly this kind of
    # representation detail. To make the LP loop's count exceed 0.5*m, ONE scalar LP variable
    # must be touched by more than 60 constraints: here constraints 1..70 all carry an entry
    # on the same dim-1 LP block (distinct coefficients, so no duplicate rows), and the other
    # 50 live in five 10-dim SDP blocks. Expected: DENSE (gate2 LP block 0), both policies.
    # SOCP remains unreachable from the dat-s format (SDPA 7 input has no SOCP blocks) --
    # documented rather than pretended away.
    sdp = [10, 10, 10, 10, 10]
    lp_touchers = 70
    m = sum(sdp) + lp_touchers
    lines = [f"{m} = mDIM", "6 = nBLOCK",
             " ".join(str(k) for k in sdp) + " -1 = bLOCKsTRUCT",
             " ".join(["1.0"] * m)]
    for b, k in enumerate(sdp, start=1):
        for r in range(1, k + 1):
            lines.append(f"0 {b} {r} {r} 1.0")
    lines.append("0 6 1 1 1.0")
    for i in range(1, lp_touchers + 1):
        lines.append(f"{i} 6 1 1 {float(i)}")
    i = lp_touchers + 1
    for b, k in enumerate(sdp, start=1):
        for r in range(1, k + 1):
            lines.append(f"{i} {b} {r} {r} 1.0")
            i += 1
    path = f"{outdir}/bmat_gate_g2_lp.dat-s"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"g2_lp: m={m} nBlock=6 (5 SDP + 1 scalar LP touched by {lp_touchers}) -> {path}")


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, sizes in FIXTURES.items():
        write_fixture(name, sizes, outdir)
    write_ring(outdir)
    write_lp_fixture(outdir)
    write_lp_low_fixture(outdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
