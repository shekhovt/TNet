# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
#%%

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def optimal_reads_linear_ref(m,n,k,S):
    """
    Computes the theoretical optimal number of memory reads for matrix multiplication of sizes (m,k) and (k,n) and the on-chip (registers amd shared) memory size S.
    The theoreitcally optimal tiling is computed asssuming continuous approximation
    We assume the result need ont be read, i.e. we accumulate starting from zero and only write out resutls once per data point
    """
    # assume tile size is (b_m, b_n, b_k)
    # Total reads is R = (m*n*k)/(b_m*b_n*b_k) * (b_m*b_k + b_k*b_n)
    # Simplify:
    # R = m*n*k * (1/b_n + 1/b_m)
    # subject to b_m*b_n + b_n*b_k + b_m*b_k <= S
    # and b_m < m, b_n < n, b_k < k

    if min(m, n, k, S) <= 0:
        raise ValueError("m, n, k, S must be positive")

    # Continuous optimization sketch:
    #   min R = m*n*k * (1/b_n + 1/b_m)
    #   s.t. b_m*b_n + b_k*(b_m + b_n) <= S, b_m,b_n,b_k >= 1.
    #
    # 1) R does not depend on b_k directly.
    # 2) Increasing b_k tightens the memory constraint, so feasible (b_m, b_n)
    #    can only get smaller.
    # 3) Since R decreases when either b_m or b_n increases,
    #    the best choice is the smallest feasible b_k, i.e. b_k = 1.
    #
    # With b_k=1, constraint becomes:
    #   b_m*b_n + b_m + b_n <= S
    #   (b_m + 1)(b_n + 1) <= S + 1.
    #
    # Let P = S+1 and x=b_m, y=b_n. Then:
    #   maximize x,y (to reduce R) under (x+1)(y+1) <= P,
    #   with box constraints 1 <= x <= m, 1 <= y <= n.
    # Since R decreases in x,y, optimum lies on active boundary
    # (x+1)(y+1)=P unless capped by x=m and/or y=n.
    #
    # KKT gives 1/x^2 : 1/y^2 = lambda*(y+1) : lambda*(x+1),
    # and interior symmetric solution x=y=sqrt(P)-1.
    # But if that violates x<=m or y<=n, optimum is at a box edge:
    #   x=m, y=P/(m+1)-1   (clipped to [1,n])
    #   y=n, x=P/(n+1)-1   (clipped to [1,m])
    # We evaluate all feasible candidates and choose minimal R.
    b_k = 1
    P = float(S) + 1.0

    # Feasibility with b_k=1 requires at least x=y=1 => 3 <= S.
    if S < 3:
        raise ValueError("No feasible tiling with b_k >= 1 when S < 3")

    candidates = []

    # Interior/symmetric candidate.
    t = max(1.0, math.sqrt(P) - 1.0)
    x = min(float(m), t)
    y = min(float(n), t)
    if (x + 1.0) * (y + 1.0) <= P + 1e-12:
        candidates.append((x, y))

    # Edge candidate x = m.
    x = float(m)
    y = P / (x + 1.0) - 1.0
    y = min(float(n), max(1.0, y))
    if (x + 1.0) * (y + 1.0) <= P + 1e-12:
        candidates.append((x, y))

    # Edge candidate y = n.
    y = float(n)
    x = P / (y + 1.0) - 1.0
    x = min(float(m), max(1.0, x))
    if (x + 1.0) * (y + 1.0) <= P + 1e-12:
        candidates.append((x, y))

    # Full-tile candidate when memory allows both dimensions.
    if (float(m) + 1.0) * (float(n) + 1.0) <= P + 1e-12:
        candidates.append((float(m), float(n)))

    if not candidates:
        raise ValueError("No feasible candidate found under provided constraints")

    b_m, b_n = min(candidates, key=lambda pair: (1.0 / pair[0] + 1.0 / pair[1]))
    R_optimal = float(m) * float(n) * float(k) * (1.0 / b_n + 1.0 / b_m)
    return R_optimal

def optimal_reads_linear_bits(m,n,k, S_bits, b1, b2, accumulator_bits = 16):
    """
    Same as above, but assymetric: matrix A has b1 bits per entry and matrix B has b2 bits per entry. We assume the output is accumulated in higher precision and does not need to be read until the end, so it does not contribute to memory but does contribute to the memory constraint.
    """
    if min(m, n, k, S_bits, b1, b2, accumulator_bits) <= 0:
        raise ValueError("m, n, k, S_bits, b1, b2, accumulator_bits must be positive")

    m = float(m)
    n = float(n)
    k = float(k)
    S = float(S_bits)
    b1 = float(b1)
    b2 = float(b2)
    acc = float(accumulator_bits)

    # Tile model (b_m, b_n, b_k):
    #   Reads-per-tile (bits) = b_m*b_k*b1 + b_k*b_n*b2
    #   Number of tiles       = m*n*k/(b_m*b_n*b_k)
    # so total input read bits:
    #   R_bits = m*n*k * (b1/b_n + b2/b_m)
    #
    # Memory constraint (bits):
    #   acc*b_m*b_n + b1*b_m*b_k + b2*b_n*b_k <= S
    # Objective does not depend on b_k directly, and larger b_k only tightens
    # memory, therefore optimal b_k is the smallest feasible value: b_k = 1.
    # Then:
    #   acc*b_m*b_n + b1*b_m + b2*b_n <= S
    #
    # Interior derivation (no m/n saturation): active constraint and
    # b_n = (S - b1*b_m)/(acc*b_m + b2), giving 1D objective
    #   f(b_m) = b2/b_m + b1*(acc*b_m + b2)/(S - b1*b_m).
    # f'(b_m)=0 simplifies to
    #   b2/b_m^2 = b1*(acc*S + b1*b2)/(S - b1*b_m)^2,
    # hence closed-form
    #   b_m* = S / (b1 + sqrt(b1*(acc*S + b1*b2)/b2)).
    # We evaluate this interior candidate plus boundary-saturated candidates
    # (b_m=m and b_n=n) and pick the minimal objective.

    # Feasibility with b_k=1 and smallest tile b_m=b_n=1.
    if S < (acc + b1 + b2):
        raise ValueError("No feasible tiling with b_k >= 1 under provided S_bits")

    candidates = []

    def add_candidate(x, y):
        # Clip to box — interior formula can land just outside due to float rounding.
        x = min(x, m)
        y = min(y, n)
        if x < 1.0 or y < 1.0:
            return
        # Use a relative tolerance so that floating-point rounding in the
        # closed-form interior expression never spuriously rejects the candidate.
        if acc * x * y + b1 * x + b2 * y <= S * (1.0 + 1e-9):
            candidates.append((x, y))

    # Interior candidate from stationary point.
    C = math.sqrt(b1 * (acc * S + b1 * b2) / b2)
    x = S / (b1 + C)
    y = (S - b1 * x) / (acc * x + b2)
    add_candidate(x, y)

    # Boundary candidate b_m = m.
    # Compute the maximum feasible b_n given b_m=m, then clip to n.
    x = m
    y = (S - b1 * x) / (acc * x + b2)
    add_candidate(x, y)  # add_candidate clips y to n if needed

    # Boundary candidate b_n = n.
    # Compute the maximum feasible b_m given b_n=n, then clip to m.
    y = n
    x = (S - b2 * y) / (acc * y + b1)
    add_candidate(x, y)  # add_candidate clips x to m if needed

    # Full-tile candidate if both dimensions fit.
    add_candidate(m, n)

    if not candidates:
        raise ValueError("No feasible candidate found under provided constraints")

    b_m, b_n = min(candidates, key=lambda pair: (b2 / pair[0] + b1 / pair[1]))
    R_bits_optimal = m * n * k * (b1 / b_n + b2 / b_m)
    return R_bits_optimal

#%%

def plot_optimal_reads_examples(out_dir="energy"):
    """
    Generate requested plots:
      1) Square matrices m=n=k=1000, b1=b2=8 bits, accumulator_bits=16,
         chip memory S from 1000 to 100000 elements.
      2) Fixed chip memory S=16KB and b1=b2 ranging from 1 to 16 bits.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    m = n = k = 1000
    accumulator_bits = 16

    # Plot 1: sweep S in elements, with equal input bitwidth.
    # Overread factor = R_bits / total_input_bits
    # where total_input_bits = k*(m*b1 + n*b2).
    bw_equal = 8
    S_elements = np.linspace(1000, 1000000, 200)
    total_input_bits = k * (m * bw_equal + n * bw_equal)
    overread_vs_S = []
    for S_el in S_elements:
        S_bits = float(S_el) * bw_equal
        R = optimal_reads_linear_bits(m, n, k, S_bits, bw_equal, bw_equal, accumulator_bits)
        overread_vs_S.append(R / total_input_bits)
    overread_vs_S = np.array(overread_vs_S)

    plt.figure(figsize=(7, 4.5))
    plt.plot(S_elements, overread_vs_S, linewidth=2)
    plt.xlabel("On-chip memory S (input elements)")
    plt.ylabel("Overread factor (R / total input size)")
    plt.title("Overread factor vs chip memory (m=n=k=1000, b1=b2=8)")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(bottom = 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1 = out_path / "mem_model_plot1_overread_vs_S.png"
    plt.show()
    # plt.savefig(fig1, dpi=160)
    plt.close()

    # Plot 2: fixed S=16KB, sweep b1=b2 in [1,16].
    # Overread factor = R_bits / total_input_bits; note total_input_bits
    # scales with bw, so the factor is R_bits / (k*(m+n)*bw).
    S_bits_fixed = 8 * 1024 * 8 # kilobytes
    bitwidths = np.arange(1, 17)
    overread_vs_bw = []
    for bw in bitwidths:
        R = optimal_reads_linear_bits(m, n, k, S_bits_fixed, bw, bw, accumulator_bits)
        total_input_bits_bw = k * (m + n) * bw
        overread_vs_bw.append(R / total_input_bits_bw)
    overread_vs_bw = np.array(overread_vs_bw)

    plt.figure(figsize=(7, 4.5))
    plt.plot(bitwidths, overread_vs_bw, marker="o", linewidth=2)
    plt.xlabel("Input bitwidth b1=b2 (bits)")
    plt.ylabel("Overread factor (R / total input size)")
    plt.title("Overread factor vs bitwidth (S=16KB, m=n=k=1000)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2 = out_path / "mem_model_plot2_overread_vs_bitwidth.png"
    # plt.savefig(fig2, dpi=160)
    plt.show()
    plt.close()
    

    return str(fig1), str(fig2)


if __name__ == "__main__":
    p1, p2 = plot_optimal_reads_examples()
    print("Saved:")
    print(p1)
    print(p2)
# %%
