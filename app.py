"""
streamlit_app.py — Interactive American Options Pricer
Live demo wrapper around the binomial tree + Longstaff-Schwartz engines.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from binomial_tree import OptionParams, OptionType, black_scholes_price, binomial_european
from american import binomial_american
from longstaff_schwartz import longstaff_schwartz

st.set_page_config(page_title="American Options Pricer", page_icon="🌳", layout="wide")

st.title("🌳 American Options Pricer")
st.caption("Binomial trees (Cox-Ross-Rubinstein) + Longstaff-Schwartz Least-Squares Monte Carlo")

# ── Sidebar inputs ──
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Spot price (S)", 1.0, 1000.0, 100.0, 1.0)
K = st.sidebar.number_input("Strike price (K)", 1.0, 1000.0, 100.0, 1.0)
T = st.sidebar.slider("Time to expiry (years)", 0.05, 5.0, 1.0, 0.05)
r = st.sidebar.slider("Risk-free rate (r)", 0.0, 0.20, 0.05, 0.005)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 1.0, 0.20, 0.01)
opt_type_str = st.sidebar.radio("Option type", ["Put", "Call"])
n_steps = st.sidebar.slider("Binomial steps (N)", 10, 2000, 500, 10)

params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma)
opt_type = OptionType.PUT if opt_type_str == "Put" else OptionType.CALL

# ── Pricing ──
bs = black_scholes_price(params, opt_type)
euro = binomial_european(params, opt_type, n_steps)
amer = binomial_american(params, opt_type, n_steps)
premium = amer - euro

col1, col2, col3, col4 = st.columns(4)
col1.metric("Black-Scholes (European)", f"${bs:.4f}")
col2.metric("Binomial European", f"${euro:.4f}")
col3.metric("Binomial American", f"${amer:.4f}")
col4.metric("Early-Exercise Premium", f"${premium:.4f}")

st.divider()

# ── Convergence chart ──
left, right = st.columns(2)

with left:
    st.subheader("Convergence to Black-Scholes")
    steps_range = [10, 25, 50, 100, 200, 350, 500, 750, 1000]
    euro_prices = [binomial_european(params, opt_type, n) for n in steps_range]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(bs, color="red", linestyle="--", label=f"Black-Scholes = {bs:.4f}")
    ax.plot(steps_range, euro_prices, "o-", color="#1a73e8", label="Binomial European")
    ax.set_xlabel("Number of steps (N)")
    ax.set_ylabel("Option price")
    ax.set_title("Binomial → Black-Scholes (note oscillation)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("Cross-Validation: LSM Monte Carlo")
    st.write("Longstaff-Schwartz prices the American option by simulation — a completely different method that should agree with the tree.")
    if st.button("Run Longstaff-Schwartz (100k paths)"):
        with st.spinner("Simulating 100,000 paths…"):
            lsm = longstaff_schwartz(params, opt_type, n_paths=100_000, n_steps=50, seed=42)
        st.metric("LSM American price", f"${lsm:.4f}")
        st.metric("Binomial American price", f"${amer:.4f}")
        diff = abs(lsm - amer)
        st.write(f"**Difference:** ${diff:.4f} — two independent methods agreeing validates both.")
        st.caption("LSM tends to be slightly below the tree due to the suboptimal-policy low-bias.")

st.divider()
with st.expander("📖 About this project"):
    st.markdown("""
    **American options** can be exercised any time before expiry, unlike European options.
    This creates a *free-boundary problem* with **no closed-form Black-Scholes solution**.

    - **Binomial tree (CRR):** handles early exercise by checking `max(continuation, exercise)` at each node
    - **Longstaff-Schwartz LSM:** prices by Monte Carlo + regression, scaling to high dimensions where trees fail
    - **Key insight:** American calls on non-dividend stocks are *never* exercised early (premium ≈ 0), but American puts *can* be (premium > 0)

    Built with NumPy, SciPy, and Matplotlib.
    """)
