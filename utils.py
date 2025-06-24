import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import statsmodels.api as sm

def get_first_and_last_day_in_period(date_list, period="M"):
    """
    Given a Series of datetime64, return the index positions of the first and last
    observation in each period.

    Parameters:
    - date_list: pd.Series of datetime64[ns]
    - period: string ('M' for month, 'Y' for year, etc.)

    Returns:
    - first_idx: integer positions of first date in each period
    - last_idx: integer positions of last date in each period
    """
    # Create a dummy DataFrame for grouping
    df = pd.DataFrame({"date": date_list})
    periods = df["date"].dt.to_period(period)

    # Use integer positions for return values
    first_idx = df.groupby(periods).head(1).index.to_numpy()
    last_idx = df.groupby(periods).tail(1).index.to_numpy()

    return first_idx, last_idx

def aggregate_returns(original_returns, date_list, period="M"):
    """
    Aggregate returns over time by compounding: ∏(1 + r) - 1

    Parameters:
    - original_returns: pd.DataFrame or pd.Series of returns (daily, monthly, etc.)
    - date_list: pd.Series of datetime64[ns], same length as original_returns
    - period: 'M' for monthly, 'Y' for yearly, etc.

    Returns:
    - aggregated_returns: np.ndarray of shape (nPeriods, nAssets)
    """
    # Ensure 2D structure for original_returns
    if isinstance(original_returns, pd.Series):
        original_returns = original_returns.to_frame()

    n_assets = original_returns.shape[1]

    # Get first and last day index per period
    first_idx, last_idx = get_first_and_last_day_in_period(date_list, period)

    n_periods = len(first_idx)
    aggregated_returns = np.zeros((n_periods, n_assets))

    for i in range(n_periods):
        first = first_idx[i]
        last = last_idx[i]

        window = original_returns.iloc[first : last + 1]  # include last row
        aggregated_returns[i, :] = (1 + window).prod(axis=0).values - 1
    
    dates = date_list.iloc[last_idx].reset_index(drop=True)

    return aggregated_returns, dates


def drawdown(P):
    """
    Compute drawdown (as negative percentage) from a price or wealth series.
    Returns an array of same length as P.
    """
    peak = np.maximum.accumulate(P)
    dd = P / peak - 1.0
    return dd

def modified_sharpe_ratio(R_ts, Rebalancing, target=0.0):
    """
    Computes annualized modified Sharpe ratio using downside deviation from 0,
    assuming returns are already excess and periodic (e.g., monthly).

    Args:
        returns (np.array): Array of periodic excess returns.
        rebalancing (int): Number of periods per year (e.g., 12 for monthly data).

    Returns:
        float: Annualized modified Sharpe ratio.
    """
    mean = np.mean(R_ts) * Rebalancing

    downside = np.minimum(R_ts - target, 0.0)
    dd = np.sqrt(np.sum(downside**2) / (len(R_ts) - 1))

    if dd == 0:
        return np.nan

    return mean / (np.sqrt(2) * dd * np.sqrt(Rebalancing))

def stats(R_ts, Rf, weights, Rebalancing):
    """
    Compute performance statistics using consistent return scaling (decimal).
    All values are computed in raw decimal scale and only scaled to % when returned.
    """
    # Total return stream
    total_returns = R_ts + Rf
    n_periods = len(R_ts)

    # Geometric returns (compounded growth)
    final_pt_val_rf = np.prod(1 + Rf)
    final_pt_val_total = np.prod(1 + total_returns)
    final_pt_val_xs = np.prod(1 + R_ts)

    geom_avg_rf = final_pt_val_rf**(1 / n_periods) - 1
    geom_avg_total_return = final_pt_val_total**(1 / n_periods) - 1
    geom_avg_xs_return =  geom_avg_total_return - geom_avg_rf # final_pt_val_xs**(Rebalancing / n_periods) - 1
    

    # Arithmetic returns
    arithm_avg_total_return = np.mean(total_returns) * Rebalancing
    arithm_avg_xs_return = np.mean(R_ts) * Rebalancing

    # Std and Sharpe ratios
    std_xs_returns = R_ts.std() * np.sqrt(Rebalancing)
    std_xs_geo = R_ts.std()
    sharpe_arithmetic = arithm_avg_xs_return / std_xs_returns
    sharpe_geometric = geom_avg_xs_return / std_xs_geo
    sharpe_geometric = sharpe_geometric * np.sqrt(Rebalancing)

    # SDR Sharpe
    sharpe_sdr = modified_sharpe_ratio(R_ts, Rebalancing)

    # Risk stats
    min_xs_return = R_ts.min()
    max_xs_return = R_ts.max()
    skew_xs_returns = skew(R_ts)
    kurt_xs_returns = kurtosis(R_ts, fisher=False)

    # Wealth path and log wealth
    wealth_path = np.cumprod(1 + total_returns)
    terminal_wealth = wealth_path.iloc[-1]
    log_terminal_wealth = np.log(terminal_wealth)

    geom_avg_total_return = geom_avg_total_return * Rebalancing
    geom_avg_xs_return = geom_avg_xs_return * Rebalancing

    # Average allocation
    avg_allocation = weights.mean(axis=0)

    # Scale key stats to percent
    return (
        log_terminal_wealth,
        avg_allocation,
        arithm_avg_total_return * 100,
        arithm_avg_xs_return * 100,
        sharpe_arithmetic,
        geom_avg_total_return * 100,
        geom_avg_xs_return * 100,
        sharpe_geometric,
        drawdown(wealth_path).min(),
        terminal_wealth,
        sharpe_sdr,
        min_xs_return * 100,
        max_xs_return * 100,
        skew_xs_returns,
        kurt_xs_returns,
    )


def summarize_strategies(returns_dict, rf_dict, weights_dict, rebalancing_freq=12):
    """
    Summarize performance metrics for multiple strategies.

    Parameters:
    - returns_dict: dict[strategy_name -> pd.Series] of excess returns
    - rf_dict: dict[strategy_name -> pd.Series] of risk-free rates
    - rebalancing_freq: number of periods per year (e.g., 12 for monthly)

    Returns:
    - pd.DataFrame of summary statistics
    """
    summary = []

    for name, R_ts in returns_dict.items():
        Rf = rf_dict[name]
        weights = weights_dict[name]
        results = stats(R_ts, Rf, weights, rebalancing_freq)

        summary.append(
            {
                "Strategy": name,
                "Final Log Wealth": results[0],
                "Average Allocation": results[1],
                "Arithm. Total Return (%)": results[2],
                "Arithm. Excess Return (%)": results[3],
                "SR": results[4],
                "Geom. Total Return (%)": results[5],
                "Geom. Excess Return (%)": results[6],
                "Geom. SR": results[7],
                "Max Drawdown": results[8],
                "Terminal Wealth": results[9],
                "SDR SR": results[10],
                "Min Return (%)": results[11],
                "Max Return (%)": results[12],
                "Skewness": results[13],
                "Kurtosis": results[14],
            }
        )

    return pd.DataFrame(summary)

def compute_turnover(previous_weights, new_weights, asset_returns, rf_return):
    """
    Python version of the MATLAB computeTurnover function.

    Inputs:
    - previous_weights: 1D array of weights at time t
    - new_weights: 1D array of weights at time t+1
    - asset_returns: 1D array of total returns of risky assets at t
    - rf_return: scalar, risk-free return at t

    Outputs:
    - turnover: total rebalancing turnover (sum of absolute differences)
    - Rp: portfolio return (gross of costs)
    """

    Rp = np.dot(previous_weights, asset_returns) + (1 - np.sum(previous_weights)) * rf_return
    value_per_asset = previous_weights * (1 + asset_returns)
    current_weights = value_per_asset / (1 + Rp)
    turnover = np.sum(np.abs(new_weights - current_weights))

    return turnover, Rp

def aggregate_fixed_window_returns(original_returns, window_size=21):
    """
    Aggregate returns over fixed non-overlapping windows (e.g., 21 trading days).

    Parameters:
    - original_returns: pd.DataFrame or pd.Series of returns (daily)
    - window_size: int, number of periods per aggregation (default=21 for ~monthly)

    Returns:
    - aggregated_returns: np.ndarray of shape (n_blocks, n_assets)
    """

    # Ensure 2D structure
    if isinstance(original_returns, pd.Series):
        original_returns = original_returns.to_frame()

    original_returns = original_returns.dropna()
    n_obs = len(original_returns)
    n_assets = original_returns.shape[1]

    # Number of full non-overlapping blocks
    n_blocks = n_obs // window_size
    aggregated_returns = np.zeros((n_blocks, n_assets))

    for i in range(n_blocks):
        start = i * window_size
        end = start + window_size
        window = original_returns.iloc[start:end]
        aggregated_returns[i, :] = (1 + window).prod(axis=0).values - 1

    return aggregated_returns


def get_fixed_month_end_dates(df, window_size=21):
    """
    Extracts the end dates of fixed-length trading months (e.g., 21-day blocks).

    Parameters:
    - df: pd.DataFrame with a DatetimeIndex
    - window_size: int, length of each trading month (default = 21)

    Returns:
    - List of pd.Timestamp corresponding to the end of each block
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex")

    n_obs = len(df)
    n_blocks = n_obs // window_size
    block_end_indices = [(i + 1) * window_size - 1 for i in range(n_blocks)]

    # Make sure we don’t go out of bounds
    block_end_indices = [i for i in block_end_indices if i < n_obs]

    return df.index[block_end_indices]





## Not in use

# def stats(R_ts, Rf, Rebalancing):
#     total_returns_simple = R_ts + Rf
#     n_periods = len(R_ts)

#     final_pt_val_rf = np.prod(1 + Rf)
#     final_pt_val_total = np.prod(1 + total_returns_simple)

#     geom_avg_rf = 100 * (final_pt_val_rf**(Rebalancing / n_periods) - 1)
#     geom_avg_total_return = 100 * (final_pt_val_total**(Rebalancing / n_periods) - 1)
#     geom_avg_xs_return = geom_avg_total_return - geom_avg_rf

#     xs_returns = R_ts * 100
#     total_returns = total_returns_simple * 100

#     arithm_avg_total_return = np.mean(total_returns) * Rebalancing
#     arithm_avg_xs_return = np.mean(xs_returns) * Rebalancing

#     std_xs_returns = xs_returns.std() * np.sqrt(Rebalancing)
#     sharpe_arithmetic = arithm_avg_xs_return / std_xs_returns

#     sharpe_geometric = geom_avg_xs_return / std_xs_returns

#     sharpe_sdr = modified_sharpe_ratio(R_ts, Rebalancing)

#     min_xs_return = xs_returns.min()
#     max_xs_return = xs_returns.max()
#     skew_xs_returns = skew(xs_returns)
#     kurt_xs_returns = kurtosis(xs_returns, fisher=False)

#     wealth_path = np.cumprod(1 + total_returns_simple)
#     terminal_wealth = wealth_path.iloc[-1]
#     log_terminal_wealth = np.log(terminal_wealth)

#     return (
#         arithm_avg_total_return,
#         arithm_avg_xs_return,
#         sharpe_arithmetic,
#         geom_avg_total_return,
#         geom_avg_xs_return,
#         sharpe_geometric,
#         sharpe_sdr,
#         terminal_wealth,
#         drawdown(wealth_path).min(),
#         min_xs_return,
#         max_xs_return,
#         skew_xs_returns,
#         kurt_xs_returns,
#         log_terminal_wealth,
#     )


# def compute_turnover_single_asset(prev_weight, new_weight, asset_return, rf_return):
#     """
#     Compute turnover for a single risky asset strategy.
    
#     Parameters:
#     - prev_weight: previous weight in risky asset (scalar)
#     - new_weight: new target weight in risky asset (scalar)
#     - asset_return: realized total return in current period (scalar)
#     - rf_return: risk-free return in current period (scalar)
    
#     Returns:
#     - turnover: absolute change in effective weight
#     - portfolio_return: realized total portfolio return
#     """
#     # Step 1: Compute realized portfolio return
#     portfolio_return = prev_weight * asset_return + (1 - prev_weight) * rf_return

#     # Step 2: Compute value of risky asset after return
#     value_in_risky = prev_weight * (1 + asset_return)

#     # Step 3: Compute new current weight after drift
#     current_weight = value_in_risky / (1 + portfolio_return)

#     # Step 4: Compute turnover as absolute difference
#     turnover = abs(new_weight - current_weight)

#     return turnover, current_weight, portfolio_return



# def fit_har_model(df, rv_col="RV"):
#     df = df.copy()

#     # Step 1: Compute log RV and lags (local variables only)
#     log_RV = np.log(df[rv_col])
#     # log_RV = df[rv_col]
#     log_RV_d = log_RV.shift(1)
#     log_RV_w = log_RV.rolling(window=5, min_periods=5).mean().shift(1)
#     log_RV_m = log_RV.rolling(window=22, min_periods=22).mean().shift(1)

#     # Step 2: Prepare regression dataset
#     har_df = pd.DataFrame(
#         {
#             "log_RV": log_RV,
#             "log_RV_d": log_RV_d,
#             "log_RV_w": log_RV_w,
#             "log_RV_m": log_RV_m,
#         }
#     ).dropna()

#     X = sm.add_constant(har_df[["log_RV_d", "log_RV_w", "log_RV_m"]])
#     y = har_df["log_RV"]
#     model = sm.OLS(y, X).fit()  # cov_type='HC0'

#     # Step 3: Predict log RV and convert back to variance scale
#     full_X = sm.add_constant(
#         pd.DataFrame({"log_RV_d": log_RV_d, "log_RV_w": log_RV_w, "log_RV_m": log_RV_m})
#     )
#     log_RV_pred = model.predict(full_X).shift(1)
#     df["RV_pred"] = np.exp(log_RV_pred)
#     # df['RV_pred'] = log_RV_pred

#     return df, model


# def har(returns, N=252):
#     """
#     Forecast variance using the HAR (Heterogeneous Autoregressive) model.

#     Args:
#         returns (array-like): Sequence of log returns.
#         N (int): Number of return observations per year (e.g., 252 for daily).

#     Returns:
#         np.ndarray: HAR-forecasted variance for each period (NaNs for initial lags).
#     """
#     returns = np.asarray(returns)
#     RV = returns**2  # Realized variance

#     # Rolling averages: 1-day (lagged), 5-day (weekly), 22-day (monthly)
#     RV_series = pd.Series(RV)
#     daily_lag = RV_series.shift(1)
#     weekly_avg = RV_series.rolling(window=5).mean().shift(1)
#     monthly_avg = RV_series.rolling(window=22).mean().shift(1)

#     # Drop initial NaNs
#     X = pd.concat([daily_lag, weekly_avg, monthly_avg], axis=1).dropna()
#     X.columns = ["daily", "weekly", "monthly"]
#     y = RV_series.loc[X.index]  # Align targets with predictors

#     # Add intercept
#     X.insert(0, "intercept", 1.0)

#     # Estimate HAR coefficients via OLS
#     beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]

#     # Forecast next-period variance
#     X_full = pd.DataFrame(
#         {
#             "intercept": 1.0,
#             "daily": daily_lag,
#             "weekly": weekly_avg,
#             "monthly": monthly_avg,
#         }
#     )
#     forecasts = X_full.dot(beta)

#     # Ensure non-negative variances
#     return np.maximum(forecasts.to_numpy(), 1e-10)


# def aggregate_variance(original_var, date_list, period="M"):
#     """
#     Aggregate variance or volatility proxy over time by summing.

#     Parameters:
#     - original_var: pd.DataFrame or pd.Series of daily variances or RV
#     - date_list: pd.Series of datetime64[ns], same length as original_var
#     - period: 'M' for monthly, 'Y' for yearly, etc.

#     Returns:
#     - aggregated_var: np.ndarray of shape (nPeriods, nAssets)
#     """
#     if isinstance(original_var, pd.Series):
#         original_var = original_var.to_frame()

#     n_assets = original_var.shape[1]
#     first_idx, last_idx = get_first_and_last_day_in_period(date_list, period)

#     n_periods = len(first_idx)
#     aggregated_var = np.zeros((n_periods, n_assets))

#     for i in range(n_periods):
#         first = first_idx[i]
#         last = last_idx[i]

#         window = original_var.iloc[first : last + 1]
#         aggregated_var[i, :] = window.sum(axis=0).values

#     return aggregated_var


# def scale_last_variance(original_var, date_list, period="M"):
#     """
#     Create time-scaled variance by using the last daily value in each period
#     and multiplying by the number of trading days in that period.

#     Parameters:
#     - original_var: pd.DataFrame or pd.Series of daily variances
#     - date_list: pd.Series of datetime64[ns], same length as original_var
#     - period: 'M' for monthly, 'Y' for yearly, etc.

#     Returns:
#     - scaled_var: np.ndarray of shape (nPeriods, nAssets)
#     """
#     if isinstance(original_var, pd.Series):
#         original_var = original_var.to_frame()

#     n_assets = original_var.shape[1]
#     first_idx, last_idx = get_first_and_last_day_in_period(date_list, period)
#     n_periods = len(first_idx)

#     scaled_var = np.zeros((n_periods, n_assets))

#     for i in range(n_periods):
#         first = first_idx[i]
#         last = last_idx[i]
#         window = original_var.iloc[first : last + 1]

#         # Get the last value in the window
#         last_value = window.iloc[-1].values

#         # Scale by number of trading days in the window
#         num_days = last - first + 1
#         scaled_var[i, :] = num_days * last_value

#     return scaled_var


import time
import os
from pathlib import Path
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



def stats_sim(R_ts, Rebalancing):
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
    sl, gsl, wl, dl = stats_sim(monthly_long, Rebalancing)
    sk, gsk, wk, dk = stats_sim(Rp, Rebalancing)
    sv, gsv, wv, dv = stats_sim(RvolT, Rebalancing)


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