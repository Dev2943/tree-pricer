"""
Day 3: Longstaff-Schwartz Least-Squares Monte Carlo for American Options.

Prices American options by simulation using regression to estimate the
continuation value at each exercise date. Validates against the binomial tree.

The algorithm (backward induction with regression):
    1. Simulate M GBM paths, storing prices at each exercise date.
    2. At expiry, option value = immediate payoff.
    3. Working backward, at each date:
       - find in-the-money paths
       - regress discounted future cashflows on a polynomial basis of S
       - the fitted values estimate the continuation value
       - exercise where immediate payoff > estimated continuation
    4. Average discounted cashflows across paths.


Run with: python3 longstaff_schwartz.py
"""

import numpy as np

from binomial_tree import OptionType, OptionParams
from american import binomial_american


# ----------------------------- PATH SIMULATION -----------------------------
def simulate_gbm_paths(
    params: OptionParams,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure.

    Returns array of shape (n_paths, n_steps + 1): each row is one path,
    columns are the stock price at t=0, t=dt, ..., t=T.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps
    S, r, sigma = params.S, params.r, params.sigma

    # Generate log-returns: shape (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum along time axis, prepend zeros for t=0
    cumulative = np.cumsum(log_returns, axis=1)
    cumulative = np.hstack([np.zeros((n_paths, 1)), cumulative])

    paths = S * np.exp(cumulative)
    return paths  # shape (n_paths, n_steps + 1)


# ----------------------------- LONGSTAFF-SCHWARTZ -----------------------------
def longstaff_schwartz(
    params: OptionParams,
    opt_type: OptionType,
    n_paths: int = 100_000,
    n_steps: int = 50,
    seed: int | None = None,
) -> float:
    """Price an American option via Least-Squares Monte Carlo.

    n_steps here is the number of exercise dates (Bermudan approximation to
    American; converges to American as n_steps grows).
    """
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    dt = T / n_steps
    discount = np.exp(-r * dt)  # one-step discount factor

    # Simulate paths
    paths = simulate_gbm_paths(params, n_paths, n_steps, seed)

    # Immediate exercise payoff function
    def payoff(stock_prices):
        if opt_type == OptionType.CALL:
            return np.maximum(stock_prices - K, 0.0)
        else:
            return np.maximum(K - stock_prices, 0.0)

    # Initialize cashflows at expiration (last column)
    # cashflow[i] = the payoff that path i ultimately delivers, discounted to
    # the time it is realized. We track the value at each path's exercise time.
    cashflow = payoff(paths[:, -1])  # value at expiry on each path
    # exercise_time[i] = the step index at which path i exercises (starts at n_steps = expiry)
    exercise_time = np.full(n_paths, n_steps)

    # Backward induction over exercise dates (from n_steps-1 down to 1; skip t=0)
    for t in range(n_steps - 1, 0, -1):
        stock_t = paths[:, t]
        immediate = payoff(stock_t)

        # Only consider in-the-money paths for the regression
        itm_mask = immediate > 0
        if itm_mask.sum() == 0:
            # No ITM paths at this date — nothing to decide, just keep discounting
            continue

        S_itm = stock_t[itm_mask]

        # The "Y" for regression: the discounted value of what each ITM path
        # currently delivers in the future. We discount each path's cashflow
        # from its exercise time back to the current time t.

        steps_to_exercise = exercise_time[itm_mask] - t
        discounted_future = cashflow[itm_mask] * (discount ** steps_to_exercise)

        # Build the polynomial basis matrix for the regression.
        # Use basis [1, S, S^2] evaluated at the ITM stock prices S_itm.
        # This is the design matrix X with shape (n_itm, 3).
       
        X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm**2])
        beta, *_ = np.linalg.lstsq(X, discounted_future, rcond=None)
        continuation_estimate = X @ beta

        # The exercise decision.
        # For each ITM path, exercise if immediate payoff > estimated continuation.
        # Where we exercise, update that path's cashflow to the immediate payoff
        # and set its exercise_time to the current step t.
        

        immediate_itm = immediate[itm_mask]
        exercise_now = immediate_itm > continuation_estimate
        itm_indices = np.where(itm_mask)[0]
        exercising_paths = itm_indices[exercise_now]
        cashflow[exercising_paths] = immediate_itm[exercise_now]
        exercise_time[exercising_paths] = t

        
    # Discount each path's cashflow from its exercise time back to t=0
    final_discounted = cashflow * (discount ** exercise_time)
    price = final_discounted.mean()

    return price


# ----------------------------- ANALYSIS -----------------------------
if __name__ == "__main__":
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    print("LONGSTAFF-SCHWARTZ MONTE CARLO vs BINOMIAL TREE")
    print("=" * 70)
    print(f"ATM 1y put: S=K={params.S}, r={params.r:.0%}, sigma={params.sigma:.0%}")
    print()

    # Binomial tree reference (high N)
    tree_price = binomial_american(params, OptionType.PUT, n_steps=2000)
    print(f"Binomial tree American put (N=2000): {tree_price:.4f}")
    print()

    # LSM at increasing path counts
    print(f"{'Paths':>10}  {'LSM Price':>12}  {'vs Tree':>10}")
    print("-" * 70)
    for n_paths in [5_000, 10_000, 50_000, 100_000, 200_000]:
        lsm_price = longstaff_schwartz(params, OptionType.PUT,
                                       n_paths=n_paths, n_steps=50, seed=42)
        diff = lsm_price - tree_price
        print(f"{n_paths:>10,}  {lsm_price:>12.4f}  {diff:>+10.4f}")

    print()
    print("Notes:")
    print("  - LSM should converge toward the tree value as paths increase")
    print("  - LSM tends to be slightly BELOW the tree (suboptimal-policy low-bias)")
    print("  - Remaining gap is Monte Carlo sampling error + the LSM bias")

    # Validation: American call should ~equal European (no early exercise)
    print("\n" + "=" * 70)
    print("Validation: American CALL via LSM should show no early-exercise benefit")
    print("=" * 70)
    lsm_call = longstaff_schwartz(params, OptionType.CALL,
                                  n_paths=100_000, n_steps=50, seed=42)
    tree_call = binomial_american(params, OptionType.CALL, n_steps=2000)
    print(f"  LSM American call:   {lsm_call:.4f}")
    print(f"  Tree American call:  {tree_call:.4f}")
    print(f"  (Both should be ~10.45 — calls aren't exercised early)")
