"""
Day 4 Part A: Multi-Asset American Basket Option via LSM.

Prices a 3-asset American basket put using Longstaff-Schwartz with
Cholesky-correlated GBM paths — and quantifies why the binomial tree
cannot do this (the curse of dimensionality).

Run with: python3 basket_option.py
"""

import numpy as np


# ----------------------------- CONFIG -----------------------------
# Three assets, each starting at 100, equal basket weights
N_ASSETS = 3
S0 = np.array([100.0, 100.0, 100.0])
WEIGHTS = np.array([1/3, 1/3, 1/3])
K = 100.0
T = 1.0
R = 0.05
SIGMA = np.array([0.20, 0.25, 0.30])   # different vol per asset

# Correlation matrix (moderate positive correlation)
CORR = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 1.0, 0.4],
    [0.2, 0.4, 1.0],
])


# ----------------------------- CORRELATED PATH SIMULATION -----------------------------
def simulate_correlated_paths(
    S0, sigma, corr, r, T, n_paths, n_steps, seed=None,
):
    """Simulate correlated GBM paths for multiple assets.

    Returns array of shape (n_paths, n_steps + 1, n_assets).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(S0)
    dt = T / n_steps

    # Cholesky factor of the correlation matrix.
    # We need L such that L @ L.T = corr. Then L @ z (for independent normals z)
    # produces correlated normals.
    
    L = np.linalg.cholesky(corr)

    # Initialize the price array
    paths = np.zeros((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0

    for step in range(1, n_steps + 1):
        # Independent standard normals: shape (n_paths, n_assets)
        Z = rng.standard_normal((n_paths, n_assets))

        # Impose correlation: each row z becomes (L @ z). Vectorized: Z @ L.T
        correlated_Z = Z @ L.T

        # GBM step for each asset
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * correlated_Z
        paths[:, step, :] = paths[:, step - 1, :] * np.exp(drift + diffusion)

    return paths


# ----------------------------- BASKET LSM -----------------------------
def basket_american_put_lsm(
    S0, weights, sigma, corr, K, r, T,
    n_paths=100_000, n_steps=50, seed=None,
):
    """Price an American basket put via Longstaff-Schwartz."""
    dt = T / n_steps
    discount = np.exp(-r * dt)

    paths = simulate_correlated_paths(S0, sigma, corr, r, T, n_paths, n_steps, seed)

    # Basket value on each path at each time: weighted sum across assets
    # shape (n_paths, n_steps + 1)
    basket = paths @ weights

    def payoff(basket_values):
        return np.maximum(K - basket_values, 0.0)

    # Initialize cashflows at expiry
    cashflow = payoff(basket[:, -1])
    exercise_time = np.full(n_paths, n_steps)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        basket_t = basket[:, t]
        immediate = payoff(basket_t)

        itm_mask = immediate > 0
        if itm_mask.sum() == 0:
            continue

        B_itm = basket_t[itm_mask]
        steps_to_exercise = exercise_time[itm_mask] - t
        discounted_future = cashflow[itm_mask] * (discount ** steps_to_exercise)



        # Regression on basket value [1, B, B^2].
        # Same structure as Day 3, but the regressor is the basket value B_itm.

        X = np.column_stack([
            np.ones_like(B_itm),
            B_itm,
            B_itm**2,
        ])
        beta, *_ = np.linalg.lstsq(
            X,
            discounted_future,
            rcond=None,
        )
        continuation_estimate = X @ beta



        # Exercise decision (same as Day 3).
        
        
        immediate_itm = immediate[itm_mask]

        exercise_now = immediate_itm > continuation_estimate

        itm_indices = np.where(itm_mask)[0]

        exercising_paths = itm_indices[exercise_now]

        cashflow[exercising_paths] = immediate_itm[exercise_now]

        exercise_time[exercising_paths] = t


        

    final_discounted = cashflow * (discount ** exercise_time)
    return final_discounted.mean()


# ----------------------------- DIMENSIONALITY ARGUMENT -----------------------------
def tree_node_count(n_assets, n_steps):
    """How many nodes a recombining lattice would need for n_assets."""
    # A recombining tree on 1 asset has (N+1) terminal nodes.
    # On d assets (independent up/down per asset), roughly (N+1)^d.
    return (n_steps + 1) ** n_assets


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("MULTI-ASSET AMERICAN BASKET OPTION via LSM")
    print("=" * 70)
    print(f"3-asset basket put, K={K}, T={T}, r={R:.0%}")
    print(f"Asset vols: {SIGMA}")
    print(f"Correlation matrix:\n{CORR}")
    print()

    price = basket_american_put_lsm(
        S0, WEIGHTS, SIGMA, CORR, K, R, T,
        n_paths=100_000, n_steps=50, seed=42,
    )
    print(f"American basket put price (LSM): {price:.4f}")
    print()

    # Dimensionality argument
    print("=" * 70)
    print("Why the binomial tree CANNOT price this:")
    print("=" * 70)
    n_steps_tree = 1000
    for d in [1, 2, 3, 5]:
        nodes = tree_node_count(d, n_steps_tree)
        print(f"  {d}-asset tree at N={n_steps_tree}: ~{nodes:.2e} nodes")
    print()
    print(f"  LSM cost scales LINEARLY in assets — a 5-asset basket is")
    print(f"  barely more expensive than a 1-asset option in Monte Carlo.")
    print(f"  This is the entire reason Longstaff-Schwartz exists.")

    # Diversification check: basket vol is lower than average asset vol
    print()
    print("=" * 70)
    print("Diversification effect:")
    print("=" * 70)
    # Effective basket variance = w' Σ w where Σ is the covariance matrix
    cov = np.outer(SIGMA, SIGMA) * CORR
    basket_var = WEIGHTS @ cov @ WEIGHTS
    basket_vol = np.sqrt(basket_var)
    avg_vol = (WEIGHTS * SIGMA).sum()
    print(f"  Weighted-average asset vol: {avg_vol:.4f}")
    print(f"  Effective basket vol:       {basket_vol:.4f}")
    print(f"  Diversification reduces effective vol by "
          f"{(1 - basket_vol/avg_vol)*100:.1f}%")
    print(f"  -> Lower effective vol means the basket option is cheaper")
    print(f"     than an equivalent single-asset option at the average vol.")
