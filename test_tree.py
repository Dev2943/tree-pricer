"""
Test suite for the tree-pricer project.

Six categories:
    1. EUROPEAN CONVERGENCE — binomial tree converges to Black-Scholes
    2. PUT-CALL PARITY — holds exactly in the tree
    3. AMERICAN PREMIUM — American >= European; calls have ~0 early-exercise premium
    4. TREE/LSM AGREEMENT — the two American methods agree
    5. GREEKS — finite-difference Greeks have correct signs
    6. BASKET — diversification reduces effective vol; price is sensible

Run with: pytest test_tree.py -v
"""

import numpy as np
import pytest

from binomial_tree import (
    OptionType, OptionParams,
    binomial_european, black_scholes_price,
)
from american import binomial_american
from longstaff_schwartz import longstaff_schwartz
from greeks_boundary import extract_exercise_boundary, american_greeks
from basket_option import basket_american_put_lsm, S0, WEIGHTS, SIGMA, CORR, K, R, T


# ----- Shared fixtures -----
@pytest.fixture
def atm_params():
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)


# ========== 1. EUROPEAN CONVERGENCE ==========

def test_european_converges_to_bs_call(atm_params):
    """At high N, the binomial call price matches Black-Scholes."""
    bs = black_scholes_price(atm_params, OptionType.CALL)
    tree = binomial_european(atm_params, OptionType.CALL, n_steps=2000)
    assert abs(tree - bs) < 0.01, f"Tree {tree} vs BS {bs}"


def test_european_converges_to_bs_put(atm_params):
    """At high N, the binomial put price matches Black-Scholes."""
    bs = black_scholes_price(atm_params, OptionType.PUT)
    tree = binomial_european(atm_params, OptionType.PUT, n_steps=2000)
    assert abs(tree - bs) < 0.01, f"Tree {tree} vs BS {bs}"


def test_convergence_improves_with_n(atm_params):
    """Error at N=1000 should be smaller than error at N=50."""
    bs = black_scholes_price(atm_params, OptionType.CALL)
    err_50 = abs(binomial_european(atm_params, OptionType.CALL, 50) - bs)
    err_1000 = abs(binomial_european(atm_params, OptionType.CALL, 1000) - bs)
    assert err_1000 < err_50, "Convergence should improve with N"


# ========== 2. PUT-CALL PARITY ==========

def test_put_call_parity_in_tree(atm_params):
    """C - P = S - K*e^(-rT) must hold in the binomial tree."""
    call = binomial_european(atm_params, OptionType.CALL, 1000)
    put = binomial_european(atm_params, OptionType.PUT, 1000)
    S, K, T, r = atm_params.S, atm_params.K, atm_params.T, atm_params.r
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 0.01, f"Parity violated: {lhs} vs {rhs}"


# ========== 3. AMERICAN PREMIUM ==========

def test_american_put_geq_european(atm_params):
    """American put must be worth at least as much as European."""
    am = binomial_american(atm_params, OptionType.PUT, 1000)
    eu = binomial_european(atm_params, OptionType.PUT, 1000)
    assert am >= eu - 1e-6, f"American put {am} < European {eu}"


def test_american_put_premium_positive(atm_params):
    """ATM American put should have a positive early-exercise premium."""
    am = binomial_american(atm_params, OptionType.PUT, 1000)
    eu = binomial_european(atm_params, OptionType.PUT, 1000)
    assert (am - eu) > 0.1, f"Premium {am-eu} too small — expected ~0.5"


def test_american_call_no_early_exercise(atm_params):
    """Non-dividend American call equals European call (never exercise early)."""
    am = binomial_american(atm_params, OptionType.CALL, 1000)
    eu = binomial_european(atm_params, OptionType.CALL, 1000)
    assert abs(am - eu) < 0.001, f"Call premium {am-eu} should be ~0"


def test_deep_itm_put_pins_to_intrinsic():
    """A deep ITM American put should equal its intrinsic value K - S."""
    params = OptionParams(S=70, K=100, T=1.0, r=0.05, sigma=0.20)
    am = binomial_american(params, OptionType.PUT, 1000)
    intrinsic = 100 - 70
    assert abs(am - intrinsic) < 0.01, f"Deep ITM put {am} should pin to {intrinsic}"


# ========== 4. TREE/LSM AGREEMENT ==========

def test_lsm_agrees_with_tree_put(atm_params):
    """Longstaff-Schwartz American put should match the tree within MC error."""
    tree = binomial_american(atm_params, OptionType.PUT, 2000)
    lsm = longstaff_schwartz(atm_params, OptionType.PUT,
                             n_paths=100_000, n_steps=50, seed=42)
    assert abs(lsm - tree) < 0.10, f"LSM {lsm} vs tree {tree} — gap too large"


def test_lsm_recovers_american_premium(atm_params):
    """LSM should price ABOVE the European value (it captures early exercise)."""
    eu = binomial_european(atm_params, OptionType.PUT, 2000)
    lsm = longstaff_schwartz(atm_params, OptionType.PUT,
                             n_paths=100_000, n_steps=50, seed=42)
    assert lsm > eu, f"LSM {lsm} should exceed European {eu}"


# ========== 5. GREEKS ==========

def test_put_greek_signs(atm_params):
    """American put Greeks should have the correct signs."""
    g = american_greeks(atm_params, OptionType.PUT, n_steps=500)
    assert g["delta"] < 0, f"Put delta should be negative, got {g['delta']}"
    assert g["gamma"] > 0, f"Gamma should be positive, got {g['gamma']}"
    assert g["vega"] > 0, f"Vega should be positive, got {g['vega']}"
    assert g["theta"] < 0, f"Theta should be negative, got {g['theta']}"
    assert g["rho"] < 0, f"Put rho should be negative, got {g['rho']}"


def test_deep_itm_american_put_delta_near_minus_one():
    """Deep ITM American put in the exercise region behaves like short stock (delta ~ -1)."""
    params = OptionParams(S=80, K=100, T=1.0, r=0.05, sigma=0.20)
    g = american_greeks(params, OptionType.PUT, n_steps=500)
    assert g["delta"] < -0.95, f"Deep ITM American put delta {g['delta']} should be ~-1"


# ========== 6. EXERCISE BOUNDARY ==========

def test_boundary_starts_at_strike(atm_params):
    """At expiry, the exercise boundary equals the strike."""
    times, boundary = extract_exercise_boundary(atm_params, n_steps=200)
    assert abs(boundary[-1] - atm_params.K) < 0.01, "Boundary at expiry should be K"


def test_boundary_below_strike_early(atm_params):
    """Early in the option's life, the boundary should be below the strike."""
    times, boundary = extract_exercise_boundary(atm_params, n_steps=200)
    # Find a boundary value near t=0.2
    early_idx = np.argmin(np.abs(times - 0.2))
    assert boundary[early_idx] < atm_params.K, "Early boundary should be below strike"


# ========== 7. BASKET ==========

def test_basket_diversification_reduces_vol():
    """Effective basket vol should be below the weighted-average asset vol."""
    cov = np.outer(SIGMA, SIGMA) * CORR
    basket_vol = np.sqrt(WEIGHTS @ cov @ WEIGHTS)
    avg_vol = (WEIGHTS * SIGMA).sum()
    assert basket_vol < avg_vol, "Diversification should reduce effective vol"


def test_basket_price_sensible():
    """Basket American put should produce a positive, reasonable price."""
    price = basket_american_put_lsm(
        S0, WEIGHTS, SIGMA, CORR, K, R, T,
        n_paths=50_000, n_steps=50, seed=42,
    )
    assert 0 < price < 20, f"Basket put price {price} outside sensible range"
