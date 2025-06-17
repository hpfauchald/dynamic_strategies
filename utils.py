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