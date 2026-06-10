"""
Visualizations for the tree-pricer project.

Produces:
    1. exercise_boundary.png — the American put early-exercise boundary
    2. convergence.png       — binomial tree convergence to Black-Scholes

Run with: python3 plot_tree.py
"""

import numpy as np
import matplotlib.pyplot as plt

from binomial_tree import OptionType, OptionParams, binomial_european, black_scholes_price
from american import binomial_american


# Colors
COLOR_BOUNDARY = "#C44E52"
COLOR_HOLD = "#4C72B0"
COLOR_TREE = "#C44E52"
COLOR_BS = "#4C72B0"


# =============================================================================
# Plot 1: Exercise boundary
# =============================================================================

# Load the boundary saved by greeks_boundary.py (or recompute)
try:
    data = np.load("exercise_boundary.npz")
    times, boundary = data["times"], data["boundary"]
except FileNotFoundError:
    from greeks_boundary import extract_exercise_boundary
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    times, boundary = extract_exercise_boundary(params, n_steps=200)

K = 100.0

fig, ax = plt.subplots(figsize=(10, 6.5))

# Shade the exercise region (below the boundary) and hold region (above)
valid = ~np.isnan(boundary)
ax.fill_between(times[valid], 0, boundary[valid],
                alpha=0.18, color=COLOR_BOUNDARY, label="Exercise region (exercise immediately)")
ax.fill_between(times[valid], boundary[valid], K * 1.5,
                alpha=0.10, color=COLOR_HOLD, label="Hold region (continue)")

# The boundary curve itself
ax.plot(times[valid], boundary[valid], color=COLOR_BOUNDARY, linewidth=2.5,
        label="Early-exercise boundary S*(t)")

# Strike reference
ax.axhline(K, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Strike K = {K:.0f}")

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Stock price", fontsize=12)
ax.set_title("American Put Early-Exercise Boundary\n"
             "S=K=100, r=5%, $\\sigma$=20%, T=1y — exercise when stock falls below the red curve",
             fontsize=13)
ax.set_ylim(60, 130)
ax.set_xlim(0, 1)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

ax.text(0.97, 0.05,
        "The boundary is the 'free boundary' that makes American\n"
        "options analytically intractable. It rises to the strike at\n"
        "expiry (no time value left) and falls earlier in life (more\n"
        "time value to preserve before exercising).",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

plt.tight_layout()
plt.savefig("exercise_boundary.png", dpi=150, bbox_inches="tight")
print("Saved exercise_boundary.png")
plt.close()


# =============================================================================
# Plot 2: Convergence to Black-Scholes
# =============================================================================

params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
bs_call = black_scholes_price(params, OptionType.CALL)

n_values = np.arange(10, 151, 1)  # every N — reveals the odd/even oscillation
tree_prices = [binomial_european(params, OptionType.CALL, int(n)) for n in n_values]
errors = np.array(tree_prices) - bs_call

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1.4, 1])

# Top: price vs N
ax1.plot(n_values, tree_prices, color=COLOR_TREE, linewidth=1, alpha=0.8,
         label="Binomial tree price")
ax1.axhline(bs_call, color=COLOR_BS, linestyle="--", linewidth=2,
            label=f"Black-Scholes = {bs_call:.4f}")
ax1.set_ylabel("Call price", fontsize=12)
ax1.set_title("Binomial Tree Convergence to Black-Scholes\n"
              "ATM 1y call — note the oscillation between odd and even N",
              fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(10, 150)

# Bottom: error vs N (shows oscillation and 1/N decay)
ax2.plot(n_values, errors, color=COLOR_TREE, linewidth=1, alpha=0.8, label="Pricing error")
ax2.axhline(0, color="gray", linewidth=0.5)
# 1/N envelope
envelope = 0.5 / n_values
ax2.plot(n_values, envelope, color="gray", linestyle=":", alpha=0.6, label="$\\pm$ O(1/N) envelope")
ax2.plot(n_values, -envelope, color="gray", linestyle=":", alpha=0.6)
ax2.set_xlabel("Number of steps N", fontsize=12)
ax2.set_ylabel("Error (tree - BS)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(10, 150)

plt.tight_layout()
plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
print("Saved convergence.png")
plt.close()

print("\nBoth plots generated.")
