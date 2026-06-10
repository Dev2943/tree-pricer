"""
Day 2: American Options via Binomial Tree.

Extends the European pricer with early-exercise: at every node, compare the
continuation value (hold) against the immediate exercise value, take the max.

This single comparison captures the entire concept of American optionality —
and prices something (the American put early-exercise premium) that has no
closed-form Black-Scholes solution.

Add this to your existing binomial_tree.py (it imports OptionType, OptionParams,
binomial_european, black_scholes_price from Day 1).


Run with: python3 american.py
"""

import numpy as np

from binomial_tree import (
    OptionType, OptionParams,
    binomial_european, black_scholes_price,
)


# ----------------------------- AMERICAN BINOMIAL TREE -----------------------------
def binomial_american(
    params: OptionParams,
    opt_type: OptionType,
    n_steps: int,
) -> float:
    """Price an American option using a CRR binomial tree.

    Identical to the European pricer EXCEPT: at every interior node we compare
    the continuation value against immediate exercise and take the larger.

    This requires tracking the stock price at every node (not just the leaves),
    because the exercise value depends on the node's stock price.
    """
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    dt = T / n_steps

    # CRR parameters (same as Day 1)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    if not (0 <= p <= 1):
        raise ValueError(f"Risk-neutral probability p={p:.4f} outside [0,1]")

    # Terminal stock prices and payoffs (same as Day 1)
    j = np.arange(n_steps + 1)
    terminal_prices = S * (u ** j) * (d ** (n_steps - j))

    if opt_type == OptionType.CALL:
        option_values = np.maximum(terminal_prices - K, 0.0)
    else:
        option_values = np.maximum(K - terminal_prices, 0.0)

    # Backward induction WITH early-exercise check
    # Backward induction WITH early-exercise check
    for step in range(n_steps - 1, -1, -1):
        i = np.arange(step + 1)
        stock_prices = S * (u ** i) * (d ** (step - i))

        # Continuation value (the "hold" value)
        continuation = discount * (
            p * option_values[1:]
            + (1 - p) * option_values[:-1]
        )

        # Early-exercise check — MUST be inside the loop
        if opt_type == OptionType.CALL:
            exercise = np.maximum(stock_prices - K, 0.0)
        else:
            exercise = np.maximum(K - stock_prices, 0.0)

        option_values = np.maximum(continuation, exercise)

    return float(option_values[0])


# ----------------------------- ANALYSIS -----------------------------
if __name__ == "__main__":
    N = 1000  # high step count for accuracy

    print("AMERICAN vs EUROPEAN OPTIONS — Early-Exercise Premium")
    print("=" * 75)

    # ---- Part 1: ATM options, no dividends ----
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    print(f"\nATM 1y options: S=K={params.S}, r={params.r:.0%}, sigma={params.sigma:.0%}, N={N}")
    print("-" * 75)

    # Calls
    am_call = binomial_american(params, OptionType.CALL, N)
    eu_call = binomial_european(params, OptionType.CALL, N)
    bs_call = black_scholes_price(params, OptionType.CALL)
    print(f"\nCALL:")
    print(f"  American:      {am_call:.4f}")
    print(f"  European:      {eu_call:.4f}")
    print(f"  Black-Scholes: {bs_call:.4f}")
    print(f"  Early-exercise premium: {am_call - eu_call:.4f}")
    print(f"  -> Should be ~0: never optimal to exercise a non-dividend call early")

    # Puts
    am_put = binomial_american(params, OptionType.PUT, N)
    eu_put = binomial_european(params, OptionType.PUT, N)
    bs_put = black_scholes_price(params, OptionType.PUT)
    print(f"\nPUT:")
    print(f"  American:      {am_put:.4f}")
    print(f"  European:      {eu_put:.4f}")
    print(f"  Black-Scholes: {bs_put:.4f}")
    print(f"  Early-exercise premium: {am_put - eu_put:.4f}")
    print(f"  -> Should be POSITIVE: deep ITM puts benefit from early exercise")

    # ---- Part 2: Put premium grows as option goes deeper in-the-money ----
    print("\n" + "=" * 75)
    print("Put early-exercise premium vs moneyness (K=100 fixed, varying S)")
    print("=" * 75)
    print(f"{'Spot S':>8}  {'Moneyness':>12}  {'Am Put':>10}  {'Eu Put':>10}  {'Premium':>10}  {'Prem %':>8}")
    print("-" * 75)

    for spot in [70, 80, 90, 100, 110, 120, 130]:
        p_params = OptionParams(S=spot, K=100, T=1.0, r=0.05, sigma=0.20)
        am = binomial_american(p_params, OptionType.PUT, N)
        eu = binomial_european(p_params, OptionType.PUT, N)
        premium = am - eu
        prem_pct = (premium / eu * 100) if eu > 0.01 else 0.0
        moneyness = "ITM" if spot < 100 else ("ATM" if spot == 100 else "OTM")
        print(f"{spot:>8}  {moneyness:>12}  {am:>10.4f}  {eu:>10.4f}  "
              f"{premium:>10.4f}  {prem_pct:>7.1f}%")

    print("\nObservation: the premium is largest (in % terms) for deep ITM puts,")
    print("where capturing the time value of money on the cash proceeds matters most.")
