"""
Day 1: CRR Binomial Tree Pricer for European Options.

Implements the Cox-Ross-Rubinstein (1979) binomial lattice and validates
its convergence to Black-Scholes as N → ∞.

The tree is recombining (u × d = 1), so an N-step tree has only N+1 endpoints
instead of 2^N. We work vectorized: compute all terminal payoffs at once,
then sweep backward by collapsing the array one element per step.


Run with: python3 binomial_tree.py
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats


# ----------------------------- TYPES -----------------------------
class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionParams:
    S: float        # spot price
    K: float        # strike
    T: float        # time to expiry (years)
    r: float        # risk-free rate
    sigma: float    # volatility


# ----------------------------- INLINE BLACK-SCHOLES (for comparison) -----------------------------
def black_scholes_price(params: OptionParams, opt_type: OptionType) -> float:
    """Standard closed-form Black-Scholes. Used here only as a convergence target."""
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == OptionType.CALL:
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


# ----------------------------- BINOMIAL TREE (European) -----------------------------
def binomial_european(
    params: OptionParams,
    opt_type: OptionType,
    n_steps: int,
) -> float:
    """Price a European option using a CRR binomial tree with N steps.

    Algorithm:
        1. Compute per-step parameters: u, d, p
        2. Build the terminal stock price vector (N+1 values)
        3. Compute terminal payoffs from those stock prices
        4. Sweep backward: at each step, collapse the value vector by one element
           using the risk-neutral discounted expectation
    """
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    dt = T / n_steps

    # CRR calibration parameters.
    # u = e^(sigma × √dt)      — up factor
    # d = 1 / u                — down factor (so u*d = 1, recombining tree)
    # p = (e^(r×dt) - d) / (u - d)   — risk-neutral probability of an up move
   
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)



    # Sanity check: 0 ≤ p ≤ 1 is required for no-arbitrage
    if not (0 <= p <= 1):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside [0,1] — "
            f"check that d ≤ e^(r×dt) ≤ u"
        )

    discount = np.exp(-r * dt)

    # terminal stock prices.
    # At step N, there are N+1 possible prices indexed by j = 0..N (number of up-moves):
    #   S_N(j) = S * u^j * d^(N-j)
   
    j = np.arange(n_steps + 1)
    terminal_prices = S * (u ** j) * (d ** (n_steps - j))

    # terminal payoffs.
    # Call: max(S_T - K, 0)
    # Put:  max(K - S_T, 0)
    
    if opt_type == OptionType.CALL:
        option_values = np.maximum(terminal_prices - K, 0.0)
    else:
        option_values = np.maximum(K - terminal_prices, 0.0)


    # backward induction.
    # At each step (working from N back to 1), the new value vector has one
    # fewer element. The relationship at each interior node:
    #
    #   V_new[k] = discount × (p × V_old[k+1] + (1-p) × V_old[k])
    #
    # In vectorized numpy:
    #   option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
    #
    # Each iteration reduces the length by 1. After n_steps iterations, length is 1.
   
    for _ in range(n_steps):
        option_values = discount * (
            p * option_values[1:]
            + (1 - p) * option_values[:-1]
        )

    # After the loop, option_values is a single-element array; return as a float
    return float(option_values[0])


# ----------------------------- CONVERGENCE COMPARISON -----------------------------
if __name__ == "__main__":
    # Test case: ATM 1-year call (same as your Project 1 reference)
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    print("Convergence of CRR Binomial Tree to Black-Scholes")
    print("=" * 75)
    print(f"ATM 1y call: S=K={params.S}, r={params.r:.0%}, sigma={params.sigma:.0%}")
    print()

    # Reference: closed-form BS
    bs_call = black_scholes_price(params, OptionType.CALL)
    bs_put = black_scholes_price(params, OptionType.PUT)
    print(f"Black-Scholes call: {bs_call:.6f}")
    print(f"Black-Scholes put:  {bs_put:.6f}")
    print()

    # Binomial at increasing N
    n_values = [5, 10, 25, 50, 100, 250, 500, 1000, 2000]
    print(f"{'N':>6}  {'Call Price':>12}  {'BS Error':>12}  {'Put Price':>12}  {'BS Error':>12}")
    print("-" * 75)

    for n in n_values:
        call_price = binomial_european(params, OptionType.CALL, n_steps=n)
        put_price = binomial_european(params, OptionType.PUT, n_steps=n)
        call_err = call_price - bs_call
        put_err = put_price - bs_put
        print(f"{n:>6}  {call_price:>12.6f}  {call_err:>+12.6f}  "
              f"{put_price:>12.6f}  {put_err:>+12.6f}")

    print()
    print("Notes:")
    print("  - Error should decrease roughly as 1/N (with oscillation between odd/even)")
    print("  - At N=2000, errors should be < 0.001 — production-grade convergence")
    print("  - Put-call parity holds: C - P = S - K*e^(-rT)")
    print(f"    Check: C - P = {bs_call - bs_put:.4f}, "
          f"S - K*e^(-rT) = {params.S - params.K * np.exp(-params.r * params.T):.4f}")
