import numpy as np
import time
import os
from pathlib import Path
import pandas as pd
import cvxpy as cp

def simulate_regime_vol_with_jumps(
    N,
    p_LL=0.995,
    p_HH=0.99,
    vol_low_annual=0.12,
    vol_high_annual=0.30,
    rho=0.95,
    std_lns=0.05,
    jump_prob=0.03,
    jump_size=0.002,
    periods=252,
):
    """
    Simulates regime-switching log-volatility with mean reversion and occasional positive jumps.

    Args:
        N (int)                 : Total number of time steps to simulate.
        p_LL (float)            : Probability of staying in the low-volatility regime.
        p_HH (float)            : Probability of staying in the high-volatility regime.
        vol_low_annual (float)  : Annualized volatility in the low regime.
        vol_high_annual (float) : Annualized volatility in the high regime.
        rho (float)             : Mean reversion coefficient.
        std_lns (float)         : Std. deviation of Gaussian noise in log-vol process.
        jump_prob (float)       : Probability of a volatility jump at any time step.
        jump_size (float)       : Jump size is sampled uniformly from [0, jump_size].
        periods (int)           : Number of periods per year.

    Returns:
        lns (np.ndarray): Simulated log-volatility series of length N.
    """
    regime = np.zeros(N, dtype=int)
    lns = np.zeros(N)
    jumps = np.zeros(N)

    mu_low = np.log(vol_low_annual / np.sqrt(periods))
    mu_high = np.log(vol_high_annual / np.sqrt(periods))

    regime[0] = 0
    lns[0] = mu_low

    for t in range(1, N):
        if regime[t - 1] == 0:
            regime[t] = 0 if np.random.rand() < p_LL else 1
        else:
            regime[t] = 1 if np.random.rand() < p_HH else 0

        mu = mu_low if regime[t] == 0 else mu_high
        lns[t] = mu + rho * (lns[t - 1] - mu) + std_lns * np.random.randn()

        if np.random.rand() < jump_prob:
            magnitude = np.random.rand() * jump_size
            if jumps[t - 1] > 0:
                magnitude *= 1.5
            if np.random.rand() < 0.1:
                magnitude += 1.0
            lns[t] += magnitude

    return lns



def drawdown(P):
    """
    Computes drawdown from a price or wealth time series.

    Args:
        P (np.ndarray): Array of portfolio values (e.g., prices or wealth levels) over time.

    Returns:
        np.ndarray: Array of drawdown values, same length as P.
    """
    peak = np.maximum.accumulate(P)   # Running maximum up to each time step
    dd = P / peak - 1.0               # Relative drop from peak
    return dd



def stats(R_ts, Rebalancing):
    """
    Computes key performance metrics for a return series:
    - Arithmetic Sharpe ratio
    - Geometric Sharpe ratio
    - Final wealth (from compounded log returns)
    - Maximum drawdown (based on wealth path)

    Args:
        R_ts (array-like): Periodic returns (can include NaNs).
        Rebalancing (int): Number of return periods per year (e.g., 252 for daily, 12 for monthly).

    Returns:
        tuple:
            - sr (float)            : Annualized arithmetic Sharpe ratio (mean / std).
            - geo_sr (float)        : Annualized geometric Sharpe ratio (mean log return / std log return).
            - final_wealth (float)  : Terminal wealth level from compounded log returns.
            - max_drawdown (float)  : Worst drawdown (as a negative percentage) from wealth path.
    """
    R_ts = np.asarray(R_ts)
    R_ts = R_ts[~np.isnan(R_ts)]  # Remove NaN values

    # Arithmetic Sharpe Ratio: mean / std of raw returns
    mean = R_ts.mean() * Rebalancing
    sd = R_ts.std(ddof=1) * np.sqrt(Rebalancing)
    sr = mean / sd if sd > 0 else np.nan

    # Geometric Sharpe Ratio: mean / std of log returns
    log_returns = np.log1p(R_ts)
    geo_mean = log_returns.mean() * Rebalancing
    geo_std = log_returns.std(ddof=1) * np.sqrt(Rebalancing)
    geo_sr = geo_mean / geo_std if geo_std > 0 else np.nan

    # Final wealth and max drawdown from exponential of cumulative log returns
    wealth_path = np.exp(np.cumsum(log_returns))
    dd = drawdown(wealth_path).min()  # Worst drawdown

    return sr, geo_sr, wealth_path[-1], dd



def kelly(
    n_years,
    periods,
    burn_in,
    p_LL,
    p_HH,
    vol_low_annual,
    vol_high_annual,
    rho,
    std_lns,
    jump_prob,
    jump_size,
    c,
    annual_ret,
    sharpe,
    f,
    min_leverage,
    max_leverage,
    Rebalancing,
    return_paths, 
    tc, 
    estimation_period, 
    expected_return = "known", 
    expected_vol = "known"
):
    """
    Simulates asset returns using a regime-switching stochastic volatility model with jumps,
    then evaluates three strategies under these returns:
    - Buy-and-Hold
    - (Frational) Kelly allocation
    - Volatility-Targeting allocation

    The simulation includes both known and estimated expectations/variances depending on user input,
    and allows for transaction costs and dynamic rebalancing.

    Args:
        n_years (int)           : Number of years to simulate (excluding burn-in).
        periods (int)           : Number of time periods per year
        burn_in (int)           : Initial number of steps to discard for stationarity.
        c (float)               : Scaling coefficient when using constant Sharpe assumption.
        annual_ret (float)      : Annual expected return when Sharpe ratio is "variable"
        sharpe (str)            : Whether to use "constant" or "variable" Sharpe ratio.
        f (float)               : Kelly fraction
        min_leverage (float)    : Minimum allowed leverage.
        max_leverage (float)    : Maximum allowed leverage.
        Rebalancing (int)       : Rebalancing frequency 
        return_paths (bool)     : If True, return full time-series data in results.
        tc (float)              : Transaction cost per unit traded.
        estimation_period (int) : Lookback window (in years) for expected return estimation (if unknown).
        expected_return (str)   : "known" or "unknown" — whether expected return is observable.
        expected_vol (str)      : "known" or "unknown" — whether expected volatility is observable.

        All other args: Parameters for the `simulate_regime_vol_with_jumps` function.

    Returns:
        dict: A dictionary of performance metrics and (optionally) simulated paths:
            - 'sharpe_*': Arithmetic Sharpe ratios
            - 'geomSR_*': Geometric Sharpe ratios
            - 'final_wealth_*': Final wealth levels
            - 'max_dd_*': Maximum drawdowns
            - 'kelly_weights', 'vol_target_weights' (if return_paths=True)
            - 'volatility_path', 'returns', and wealth paths by strategy (if return_paths=True)
    """

    # 1) Simulate
    N = n_years * periods
    T = N + burn_in

    p = np.zeros(T)
    r = np.zeros(T)
    E_r = np.zeros(T)

    lns = simulate_regime_vol_with_jumps(
        T,
        p_LL,
        p_HH,
        vol_low_annual,
        vol_high_annual,
        rho,
        std_lns,
        jump_prob,
        jump_size,
        periods,
    )

    # 2) Build path
    for i in range(periods + 1, T):
        sigma = np.exp(lns[i])

        if sharpe == "constant":
            E_r[i] = c * sigma - 0.5*sigma**2
        elif sharpe == "variable":
            E_r[i] = annual_ret / periods - 0.5*sigma**2

        r[i] = E_r[i] +  sigma * np.random.randn()
        p[i] = p[i - 1] + r[i]
    
    R = np.exp(r) - 1

    step = periods // Rebalancing

    vol = np.exp(lns)

    if expected_vol == "known":
        E_R2 = vol**2
        pred_E_R2 = "n.a."
    elif expected_vol == "unknown":
        R = pd.Series(R)
        R_sq = R**2
        EMW = R_sq.ewm(halflife=20, adjust=False).mean()
        pred_E_R2 = EMW.shift(1)

        # pred_E_R2 = har_new(r, step, periods * estimation_period)
        E_R2 = pred_E_R2
    

    if expected_return == "known":
        E_R = E_r + 0.5 * E_R2
    elif expected_return == "unknown":
        pred_E_R = pd.Series(r).shift(1).rolling(window=estimation_period * periods).mean().to_numpy()
        E_R = pred_E_R + 0.5 * E_R2


    # 3) Discard burn-in
    p = p[burn_in:] - p[burn_in]
    r = r[burn_in:]
    E_r = E_r[burn_in:]
    lns = lns[burn_in:]
    E_R = E_R[burn_in:]
    E_R2 = E_R2[burn_in:]
    vol = vol[burn_in:]
    pred_E_R2 = pred_E_R2[burn_in:]
    R = R[burn_in:]

    # 4) Levels 
    P = np.exp(p)
    
    # (Fractional) Kelly Weights
    w = np.clip(f * (E_R / E_R2), min_leverage, max_leverage)

    # Volatility Targeting weights
    vol_pred = np.sqrt(E_R2)

    # Step 1: Compute raw volatility-targeting weights
    mean_vol = np.mean(vol)
    w_volTarget = mean_vol / vol_pred

    vol_xs_returns = R * w_volTarget

    benchmark_std = np.std(R, ddof=1)
    strategy_std = np.std(vol_xs_returns, ddof=1)
    scaling = benchmark_std / strategy_std
    w_volTarget *= scaling

    w_volTarget = np.clip(w_volTarget, min_leverage, max_leverage)

    w = np.asarray(w)
    w_volTarget = np.asarray(w_volTarget)

    # 5) Rebalancing
    rebalance_points = list(range(step, len(R) - step, step))
    steps = len(rebalance_points)

    Rp, Rp_tc = rebalanced_returns(R, w, Rebalancing, periods, tc)
    RvolT, RvolT_tc_ = rebalanced_returns(R, w_volTarget, Rebalancing, periods, tc)

    # 6) Buy-and-hold monthly
    monthly_long = P[step::step] / P[:-step:step] - 1

    # 7) Stats
    sl, gsl, wl, dl = stats(monthly_long, Rebalancing)
    sk, gsk, wk, dk = stats(Rp, Rebalancing)
    sv, gsv, wv, dv = stats(RvolT, Rebalancing)


    results = {
    "sharpe_long": sl,
    "sharpe_kelly": sk,
    "sharpe_volTarget": sv,
    "geomSR_long": gsl, 
    "geomSR_kelly": gsk, 
    "geomSR_volTarget": gsv,
    "final_wealth_long": wl,
    "final_wealth_kelly": wk,
    "final_wealth_volTarget": wv,
    "max_dd_volTarget": dv,
    "max_dd_long": dl,
    "max_dd_kelly": dk,
}

    if return_paths:
        results.update({
        "volatility_path": vol * np.sqrt(periods),  # Annualized
        "log_prices": p,
        "kelly_weights": w,
        "vol_target_weights": w_volTarget,
        "Vol": vol, 
        "wealth_paths": {
            "buy_and_hold": np.cumprod(1 + monthly_long),
            "kelly": np.cumprod(1 + Rp),
            "vol_target": np.cumprod(1 + RvolT),
        },
        "returns": {
            "Actual expectation": (E_r + 0.5 * vol**2), 
            "Used expectation": E_R, 
            "Realized return": R
        }
    })
    return results


def rebalanced_returns(
    R: np.ndarray,
    w: np.ndarray,
    rebalancing: int,
    periods: int = 252,
    tc: float = 0.000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rebalanced returns (net of transaction costs) and trading costs
    for a single-strategy weight series, rebalancing a given number of times per year.

    Args:
        R            : array of log returns, length T
        w            : array of target weights, length T
        rebalancing  : times per year to rebalance
        periods      : number of periods in one year (default 252 for daily data)
        tc           : transaction cost per unit traded (default 0.0%)

    Returns:
        Rp           : np.ndarray of net returns for each rebalance interval
        trade_costs  : np.ndarray of transaction costs at each rebalance
    """
    T = len(R)
    step = periods // rebalancing
    if step < 1:
        raise ValueError(
            f"Rebalancing frequency ({rebalancing}) must be ≤ periods ({periods})."
        )

    rebalance_points = np.arange(step, T, step)
    n = len(rebalance_points)

    Rp = np.zeros(n)
    trade_costs = np.zeros(n)

    for idx, i in enumerate(rebalance_points):
        prev_ret = max((np.prod(1 + R[i - step : i]) - 1), -1)
        this_ret = max((np.prod(1 + R[i : min(i + step, T)]) - 1), -1)

        w_eff = w[i - 1] * (1 + prev_ret) / (1 + w[i - 1] * prev_ret)

        # Full rebalance
        trade = abs(w[i] - w_eff)
        cost = trade * tc
        trade_costs[idx] = cost

        # Update weight (fully rebalanced)
        w[i] = w[i]

        # Compute net return
        Rp[idx] = max(w[i] * this_ret - cost, -1)

    return Rp, trade_costs

def run_simulation(
    sim,
    n_years,
    periods,
    burn_in,
    p_LL,
    p_HH,
    vol_low_annual,
    vol_high_annual,
    rho,
    std_lns,
    jump_prob,
    jump_size,
    c,
    annual_ret,
    sharpe,
    f,
    min_leverage,
    max_leverage,
    Rebalancing,
    return_paths,
    tc,
    estimation_period,
    expected_return,
    expected_vol
):
    """
    Runs a single simulation of asset return strategies using the Kelly model.

    Args:
        sim (int): Simulation index for tracking.
        All other args: Parameters for the `kelly` function.

    Returns:
        tuple:
            - metrics (dict): Dictionary of summary statistics for this simulation.
            - kelly_log_wealth (float): Log of final Kelly wealth.
            - bh_log_wealth (float): Log of final Buy-and-Hold wealth.
            - vol_log_wealth (float): Log of final Volatility Targeting wealth.
    """
    res = kelly(
        n_years, periods, burn_in, p_LL, p_HH, vol_low_annual, vol_high_annual,
        rho, std_lns, jump_prob, jump_size, c, annual_ret, sharpe, f, min_leverage,
        max_leverage, Rebalancing, return_paths, tc, estimation_period, 
        expected_return, expected_vol
    )

    return {
        'sim': sim,
        'half-Kelly Final Log Wealth':              np.log(res['final_wealth_kelly']),
        'Volatility Targeting Final Log Wealth':    np.log(res['final_wealth_volTarget']),
        'Buy-and-Hold Final Log Wealth':            np.log(res['final_wealth_long']),
        'Half-kelly Max Drawdown':                  res['max_dd_kelly'],
        'Volatility Targeting Max Drawdown':        res['max_dd_volTarget'],
        'Buy-and-Hold Max Drawdown':                res['max_dd_long'],
        'half-kelly SR':                            res['sharpe_kelly'],
        'Volatility Targeting SR':                  res['sharpe_volTarget'],
        'Buy-and-Hold SR':                          res['sharpe_long'],
        'half-Kelly geometric SR':                  res['geomSR_kelly'],
        'Buy-and-Hold  geometric SR':               res['geomSR_long'],
        'Volatility Targeting  geometric SR':       res['geomSR_volTarget'],
        'half-Kelly weight':                        np.mean(res['kelly_weights']),
        'Volatility Targeting weight':              np.mean(res['vol_target_weights']),
    }, np.log(res['final_wealth_kelly']), np.log(res['final_wealth_long']), np.log(res['final_wealth_volTarget'])