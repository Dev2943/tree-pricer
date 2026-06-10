"""
Day 4 Part B: Early-Exercise Boundary & American Greeks.

Two analyses on the American put:
    1. Extract the early-exercise boundary S*(t) from the binomial tree
    2. Compute all five Greeks via finite differences (no closed form exists)


Run with: python3 greeks_boundary.py
"""

import numpy as np

from binomial_tree import OptionType, OptionParams
from american import binomial_american


# ----------------------------- EXERCISE BOUNDARY EXTRACTION -----------------------------
def extract_exercise_boundary(
    params: OptionParams,
    n_steps: int = 200,
):
    """Extract the early-exercise boundary S*(t) for an American put.

    At each time step, the boundary is the HIGHEST stock price at which
    immediate exercise is still optimal (exercise >= continuation).

    Returns
    -------
    times : array of time points
    boundary : array of S*(t) at each time (NaN where no exercise region exists)
    """
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal values (put)
    j = np.arange(n_steps + 1)
    terminal_prices = S * (u ** j) * (d ** (n_steps - j))
    option_values = np.maximum(K - terminal_prices, 0.0)

    boundary = np.full(n_steps + 1, np.nan)
    # At expiry, the boundary is the strike (you'd exercise any ITM put at expiry)
    boundary[n_steps] = K

    for step in range(n_steps - 1, -1, -1):
        i = np.arange(step + 1)
        stock_prices = S * (u ** i) * (d ** (step - i))

        continuation = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
        exercise = np.maximum(K - stock_prices, 0.0)

        # Identify which nodes exercise (exercise >= continuation AND in-the-money).
        # Then the boundary at this step is the highest stock price among exercising nodes.
        
        exercises = (exercise >= continuation) & (exercise > 0)

        if exercises is not None and exercises.any():
            boundary[step] = stock_prices[exercises].max()

        # Update option values for next iteration (standard American step)
        option_values = np.maximum(continuation, exercise)

    times = np.linspace(0, T, n_steps + 1)
    return times, boundary


# ----------------------------- FINITE-DIFFERENCE GREEKS -----------------------------
def american_greeks(
    params: OptionParams,
    opt_type: OptionType,
    n_steps: int = 1000,
):
    """Compute American option Greeks via finite differences."""
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma

    # Base price
    base = binomial_american(params, opt_type, n_steps)

    # Bump sizes
    dS = S * 0.01      # 1% of spot
    dsigma = 0.01      # 1 vol point
    dr = 0.0001        # 1 basis point
    dT = 1/365         # one day

    # Delta and Gamma via central differences on spot.
    # Delta = (V(S+dS) - V(S-dS)) / (2*dS)
    # Gamma = (V(S+dS) - 2*V(S) + V(S-dS)) / dS^2
   
    params_up = OptionParams(
        S=S + dS,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    params_dn = OptionParams(
        S=S - dS,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    v_up = binomial_american(
        params_up,
        opt_type,
        n_steps,
    )

    v_dn = binomial_american(
        params_dn,
        opt_type,
        n_steps,
    )

    delta = (v_up - v_dn) / (2 * dS)
    gamma = (v_up - 2 * base + v_dn) / (dS ** 2)

    # Vega, Theta, Rho via one-sided differences.
    # Vega  = (V(sigma+dsigma) - V(sigma)) / dsigma     [per 1.00 vol; divide by 100 for per-1%]
    # Theta = (V(T-dT) - V(T)) / dT  ... but reported per day, so just V(T-dT) - V(T)
    # Rho   = (V(r+dr) - V(r)) / dr                      [per 1.00 rate; divide by 100 for per-1%]
    

    v_vega = binomial_american(
        OptionParams(S, K, T, r, sigma + dsigma),
        opt_type,
        n_steps,
    )

    vega = (v_vega - base) / dsigma / 100


    v_theta = binomial_american(
        OptionParams(S, K, T - dT, r, sigma),
        opt_type,
        n_steps,
    )

    theta = v_theta - base


    v_rho = binomial_american(
        OptionParams(S, K, T, r + dr, sigma),
        opt_type,
        n_steps,
    )

    rho = (v_rho - base) / dr / 100

    return {
        "price": base,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    # ---- Exercise boundary ----
    print("EARLY-EXERCISE BOUNDARY (American Put)")
    print("=" * 70)
    times, boundary = extract_exercise_boundary(params, n_steps=200)
    print(f"{'Time':>8}  {'Boundary S*(t)':>16}  {'Exercise if S below'}")
    print("-" * 70)
    # Print a sample of boundary points
    for idx in range(0, len(times), 20):
        t = times[idx]
        b = boundary[idx]
        if not np.isnan(b):
            print(f"{t:>8.3f}  {b:>16.2f}  S ≤ {b:.2f}")
    print()
    print("The boundary starts at K=100 at expiry and curves DOWNWARD earlier in")
    print("time: you need the put deeper in-the-money to justify early exercise")
    print("when more time value remains.")

    # Save boundary for plotting on Day 5
    np.savez("exercise_boundary.npz", times=times, boundary=boundary)
    print("\nSaved boundary to exercise_boundary.npz")

    # ---- Greeks ----
    print("\n" + "=" * 70)
    print("AMERICAN PUT GREEKS (finite differences)")
    print("=" * 70)
    greeks = american_greeks(params, OptionType.PUT, n_steps=1000)
    print(f"  Price:  {greeks['price']:>10.4f}")
    print(f"  Delta:  {greeks['delta']:>10.4f}   (put delta is negative)")
    print(f"  Gamma:  {greeks['gamma']:>10.4f}")
    print(f"  Vega:   {greeks['vega']:>10.4f}   (per 1% vol move)")
    print(f"  Theta:  {greeks['theta']:>10.4f}   (per day)")
    print(f"  Rho:    {greeks['rho']:>10.4f}   (per 1% rate move)")

    # ---- American vs European delta comparison ----
    print("\n" + "=" * 70)
    print("American vs European put delta across moneyness")
    print("=" * 70)
    from binomial_tree import binomial_european

    def fd_delta(pricer, params, opt_type, n_steps):
        S = params.S
        dS = S * 0.01
        up = pricer(OptionParams(S+dS, params.K, params.T, params.r, params.sigma), opt_type, n_steps)
        dn = pricer(OptionParams(S-dS, params.K, params.T, params.r, params.sigma), opt_type, n_steps)
        return (up - dn) / (2 * dS)

    print(f"{'Spot':>6}  {'Am Delta':>10}  {'Eu Delta':>10}")
    print("-" * 70)
    for spot in [80, 90, 100, 110, 120]:
        p = OptionParams(spot, 100, 1.0, 0.05, 0.20)
        am_d = fd_delta(binomial_american, p, OptionType.PUT, 500)
        eu_d = fd_delta(binomial_european, p, OptionType.PUT, 500)
        print(f"{spot:>6}  {am_d:>10.4f}  {eu_d:>10.4f}")

    print("\nAmerican put delta is more negative (closer to -1) when ITM —")
    print("the early-exercise feature makes it behave more like short stock.")
