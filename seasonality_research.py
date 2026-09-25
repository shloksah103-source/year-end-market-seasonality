# Year-end seasonality study
# Shlok Sah
#
# I started this because I was curious if the market usually behaves
# differently around Christmas and New Year. The main window I test is
# Dec. 21 to Jan. 5. SPY is the main benchmark, then I compare a few
# other ETFs and leveraged ETFs too.

from pathlib import Path
import warnings
import subprocess
import sys
import webbrowser

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")



# 1. Project settings

# keeps files in the folder where I run the notebook/script
ROOT = Path.cwd()
PROJECT_DIR = ROOT / "year_end_seasonality_project"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

for folder in [PROJECT_DIR, DATA_DIR, OUTPUT_DIR, CHART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

START_DATE = "2008-01-01"

# yfinance end date is EXCLUSIVE.
# To include the full Dec 2025 -> early Jan 2026 seasonal window,
# the end date must extend slightly into January 2026.
END_DATE = "2026-01-10"

PRIMARY_TICKER = "SPY"  # main benchmark

ALL_TICKERS = [
    "SPY", "SSO", "UPRO",
    "QQQ", "QLD", "TQQQ",
    "IWM", "IWB", "XLP", "WMT"
]

RESEARCH_GROUPS = {
    "PRIMARY_SPY": ["SPY"],
    "UNLEVERED_REPLICATION": ["SPY", "QQQ", "IWM", "IWB", "XLP", "WMT"],
    "SP500_LEVERAGE_FAMILY": ["SPY", "SSO", "UPRO"],
    "NASDAQ_LEVERAGE_FAMILY": ["QQQ", "QLD", "TQQQ"],
}

ENTRY_MONTH = 12
ENTRY_DAY = 21

EXIT_MONTH = 1
EXIT_DAY = 5

COST_BPS_PER_SIDE = 5.0  # small trading-cost assumption
BOOTSTRAP_SIMULATIONS = 10_000
RANDOM_WINDOW_SIMULATIONS = 10_000
BOOTSTRAP_BLOCK = 2
RANDOM_SEED = 42

TRAIN_END_YEAR = 2018  # later sample starts after 2018

# If False, previously downloaded CSVs are reused.
REFRESH_DATA = False

print("Project folder:", PROJECT_DIR)


# 2. Download and save price data

def clean_yfinance_frame(raw, ticker):
    """Clean the downloaded prices."""
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        raise ValueError(f"{ticker}: Close column missing")

    close = raw["Close"].copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()

    if close.empty:
        raise ValueError(f"{ticker}: no valid close prices")
    if close.index.has_duplicates:
        raise ValueError(f"{ticker}: duplicate dates found")
    if (close <= 0).any():
        raise ValueError(f"{ticker}: non-positive prices found")

    close.name = ticker
    return close


def get_price_series(ticker, refresh=False):
    """Download prices or use the saved CSV if it is already there."""
    path = DATA_DIR / f"{ticker}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        series = pd.to_numeric(df[ticker], errors="coerce").dropna()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        return series.sort_index()

    print(f"Downloading {ticker}...")

    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
    )

    series = clean_yfinance_frame(raw, ticker)
    series.to_frame().to_csv(path, index_label="Date")

    return series


def load_all_prices():
    prices = {}
    for ticker in ALL_TICKERS:
        prices[ticker] = get_price_series(ticker, refresh=REFRESH_DATA)

    manifest = []
    for ticker, series in prices.items():
        manifest.append({
            "Ticker": ticker,
            "Rows": len(series),
            "First_Date": series.index.min(),
            "Last_Date": series.index.max(),
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUTPUT_DIR / "data_manifest.csv", index=False)

    print("\nDATA COVERAGE")
    print(manifest_df.to_string(index=False))

    return prices


# 3. Build the Dec. 21 to Jan. 5 trade dates

def build_schedule(master_calendar, entry_day=ENTRY_DAY, exit_day=EXIT_DAY):
    """Make the Dec. 21 to Jan. 5 trade dates for each year."""
    rows = []

    start_year = pd.Timestamp(START_DATE).year
    end_year = pd.Timestamp(END_DATE).year

    for year in range(start_year, end_year):
        entry_lower = pd.Timestamp(year, ENTRY_MONTH, entry_day)
        entry_upper = pd.Timestamp(year, 12, 31)

        exit_lower = pd.Timestamp(year + 1, 1, 1)
        exit_upper = pd.Timestamp(year + 1, EXIT_MONTH, exit_day)

        entry_candidates = master_calendar[
            (master_calendar >= entry_lower) &
            (master_calendar <= entry_upper)
        ]

        exit_candidates = master_calendar[
            (master_calendar >= exit_lower) &
            (master_calendar <= exit_upper)
        ]

        if len(entry_candidates) == 0 or len(exit_candidates) == 0:
            continue

        entry = entry_candidates[0]
        exit_ = exit_candidates[-1]

        if exit_ <= entry:
            continue

        trade_dates = master_calendar[
            (master_calendar >= entry) &
            (master_calendar <= exit_)
        ]

        rows.append({
            "Season": year,
            "Entry": entry,
            "Exit": exit_,
            "Holding_Intervals": len(trade_dates) - 1,
            "Trading_Days": len(trade_dates),
        })

    schedule = pd.DataFrame(rows).set_index("Season")

    if schedule.empty:
        raise ValueError("No complete year-end windows found.")

    return schedule


def eligible_schedule_for_group(schedule, master_calendar, prices, tickers):
    """Keep years where every ticker in the group has enough data."""
    good = []

    for season, row in schedule.iterrows():
        cycle_start = pd.Timestamp(season, 1, 6)
        cycle_end = pd.Timestamp(season + 1, 1, 5)

        cycle_dates = master_calendar[
            (master_calendar >= cycle_start) &
            (master_calendar <= cycle_end)
        ]

        if len(cycle_dates) == 0:
            continue

        passed = True

        for ticker in tickers:
            aligned = prices[ticker].reindex(cycle_dates)
            if aligned.isna().any():
                passed = False
                break

        if passed:
            good.append(season)

    return schedule.loc[good].copy()


# 4. Calculate each yearly trade

def net_return_from_prices(entry_price, exit_price, cost_bps=COST_BPS_PER_SIDE):
    """Return after entry and exit costs."""
    c = cost_bps / 10_000.0
    gross_multiplier = exit_price / entry_price
    net_multiplier = gross_multiplier * (1 - c) / (1 + c)
    return net_multiplier - 1


def build_trade_table(prices, schedule, tickers, cost_bps=COST_BPS_PER_SIDE):
    rows = []

    for season, row in schedule.iterrows():
        for ticker in tickers:
            p = prices[ticker].loc[row.Entry:row.Exit]

            if len(p) < 2:
                raise ValueError(f"{ticker}: incomplete trade window for {season}")

            entry_price = float(p.iloc[0])
            exit_price = float(p.iloc[-1])

            gross_return = exit_price / entry_price - 1
            net_return = net_return_from_prices(
                entry_price,
                exit_price,
                cost_bps=cost_bps,
            )

            # In-trade drawdown including entry cost approximately.
            c = cost_bps / 10_000.0
            path = p.to_numpy() / entry_price / (1 + c)
            path[-1] *= (1 - c)
            path = np.r_[1.0, path]

            running_peak = np.maximum.accumulate(path)
            drawdown = path / running_peak - 1

            rows.append({
                "Season": season,
                "Ticker": ticker,
                "Entry_Date": row.Entry,
                "Exit_Date": row.Exit,
                "Trading_Days": len(p),
                "Gross_Return": gross_return,
                "Net_Return": net_return,
                "Max_In_Trade_Drawdown": float(drawdown.min()),
            })

    return pd.DataFrame(rows)


# 5. Basic return and risk statistics

def summarize_trade_returns(trades):
    rows = []

    for ticker, g in trades.groupby("Ticker"):
        r = g["Net_Return"].dropna()

        winners = r[r > 0]
        losers = r[r < 0]

        vol = r.std(ddof=1)

        downside = np.sqrt(
            np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)
        )

        gross_loss = abs(losers.sum())

        rows.append({
            "Ticker": ticker,
            "N_Seasons": len(r),
            "Mean_Return": r.mean(),
            "Median_Return": r.median(),
            "Seasonal_Volatility": vol,
            "Mean_to_Volatility": r.mean() / vol if vol > 0 else np.nan,
            "Downside_Deviation": downside,
            "Sortino_Zero_Target": r.mean() / downside if downside > 0 else np.nan,
            "Profit_Factor": winners.sum() / gross_loss if gross_loss > 0 else np.nan,
            "Win_Rate": (r > 0).mean(),
            "Best_Return": r.max(),
            "Worst_Return": r.min(),
            "Average_Winner": winners.mean() if len(winners) else np.nan,
            "Average_Loser": losers.mean() if len(losers) else np.nan,
            "Worst_In_Trade_Drawdown": g["Max_In_Trade_Drawdown"].min(),
        })

    return pd.DataFrame(rows)


# 6. Bootstrap confidence intervals

def circular_block_bootstrap_indices(n, simulations, block_size, seed):
    """
    Bootstrap the yearly returns in small blocks.
    I use a block size of 2 so neighboring years are not treated as
    completely unrelated.
    """
    rng = np.random.default_rng(seed)

    blocks_needed = int(np.ceil(n / block_size))
    starts = rng.integers(
        0,
        n,
        size=(simulations, blocks_needed),
    )

    blocks = (
        starts[..., None] + np.arange(block_size)
    ) % n

    return blocks.reshape(simulations, -1)[:, :n]


def bootstrap_mean_ci(values, simulations=BOOTSTRAP_SIMULATIONS,
                      block_size=BOOTSTRAP_BLOCK, seed=RANDOM_SEED):
    x = np.asarray(values, dtype=float)

    idx = circular_block_bootstrap_indices(
        len(x),
        simulations,
        block_size,
        seed,
    )

    boot_means = x[idx].mean(axis=1)

    low, high = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return low, high


# 7. Adjust p-values when testing several assets

def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)

    result = np.full(len(p), np.nan)

    valid = np.flatnonzero(np.isfinite(p))

    if len(valid) == 0:
        return result

    ordered = valid[np.argsort(p[valid])]

    adjusted = (
        p[ordered]
        * len(ordered)
        / np.arange(1, len(ordered) + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    result[ordered] = np.minimum(adjusted, 1.0)

    return result


# 8. Compare the seasonal window with random windows

def matched_random_window_test(
    prices,
    master_calendar,
    schedule,
    tickers,
    simulations=RANDOM_WINDOW_SIMULATIONS,
    seed=RANDOM_SEED,
    cost_bps=COST_BPS_PER_SIDE,
):
    """Compare the year-end window with random windows of the same length."""
    rng = np.random.default_rng(seed)

    expected_by_season = []
    simulated_mean = np.zeros((simulations, len(tickers)))

    for season, row in schedule.iterrows():
        cycle_start = pd.Timestamp(season, 1, 6)
        cycle_end = pd.Timestamp(season + 1, 1, 5)

        dates = master_calendar[
            (master_calendar >= cycle_start) &
            (master_calendar <= cycle_end)
        ]

        L = int(row.Holding_Intervals)

        starts = np.arange(0, len(dates) - L)

        entry_dates = dates[starts]
        exit_dates = dates[starts + L]

        # Candidate must not overlap the seasonal trade.
        keep = (
            (exit_dates < row.Entry) |
            (entry_dates > row.Exit)
        )

        starts = starts[keep]

        if len(starts) == 0:
            raise ValueError(
                f"No eligible random windows for season {season}"
            )

        matrix = np.column_stack([
            prices[ticker].reindex(dates).to_numpy()
            for ticker in tickers
        ])

        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Missing data in matched-window cycle {season}"
            )

        candidate_returns = []

        for s in starts:
            entry_prices = matrix[s]
            exit_prices = matrix[s + L]

            c = cost_bps / 10_000.0

            returns = (
                (exit_prices / entry_prices)
                * (1 - c)
                / (1 + c)
                - 1
            )

            candidate_returns.append(returns)

        candidate_returns = np.asarray(candidate_returns)

        expected_by_season.append(
            candidate_returns.mean(axis=0)
        )

        chosen = rng.integers(
            0,
            len(candidate_returns),
            size=simulations,
        )

        simulated_mean += (
            candidate_returns[chosen]
            / len(schedule)
        )

    expected_by_season = np.asarray(expected_by_season)

    return expected_by_season, simulated_mean


# 9. Run the statistical tests for each group

def run_group_inference(
    group_name,
    tickers,
    prices,
    master_calendar,
    base_schedule,
):
    schedule = eligible_schedule_for_group(
        base_schedule,
        master_calendar,
        prices,
        tickers,
    )

    if len(schedule) < 6:
        raise ValueError(
            f"{group_name}: fewer than 6 complete seasons."
        )

    trades = build_trade_table(
        prices,
        schedule,
        tickers,
        cost_bps=COST_BPS_PER_SIDE,
    )

    returns = (
        trades
        .pivot(
            index="Season",
            columns="Ticker",
            values="Net_Return",
        )
        .loc[schedule.index, tickers]
    )

    expected, random_null = matched_random_window_test(
        prices,
        master_calendar,
        schedule,
        tickers,
    )

    expected_df = pd.DataFrame(
        expected,
        index=schedule.index,
        columns=tickers,
    )

    rows = []

    for j, ticker in enumerate(tickers):
        actual = returns[ticker].to_numpy()
        random_expected = expected_df[ticker].to_numpy()

        advantage = actual - random_expected

        mean_low, mean_high = bootstrap_mean_ci(
            actual,
            seed=RANDOM_SEED + 10 + j,
        )

        adv_low, adv_high = bootstrap_mean_ci(
            advantage,
            seed=RANDOM_SEED + 100 + j,
        )

        actual_mean = actual.mean()
        null_distribution = random_null[:, j]

        # One-sided: probability random timing is at least as good.
        timing_p = (
            1
            + np.sum(null_distribution >= actual_mean)
        ) / (
            len(null_distribution) + 1
        )

        t_result = stats.ttest_1samp(
            actual,
            popmean=0,
            nan_policy="omit",
        )

        rows.append({
            "Group": group_name,
            "Ticker": ticker,
            "N_Seasons": len(actual),
            "First_Season": schedule.index.min(),
            "Last_Season": schedule.index.max(),
            "Mean_Net_Return": actual_mean,
            "Median_Net_Return": np.median(actual),
            "Win_Rate": np.mean(actual > 0),
            "Mean_CI_Low": mean_low,
            "Mean_CI_High": mean_high,
            "Matched_Random_Mean": random_expected.mean(),
            "Mean_Advantage": advantage.mean(),
            "Advantage_CI_Low": adv_low,
            "Advantage_CI_High": adv_high,
            "Timing_P_One_Sided": timing_p,
            "Zero_Mean_TTest_P": t_result.pvalue,
        })

    result = pd.DataFrame(rows)

    result["Timing_Q_BH"] = benjamini_hochberg(
        result["Timing_P_One_Sided"]
    )

    result["Zero_Mean_TTest_Q_BH"] = benjamini_hochberg(
        result["Zero_Mean_TTest_P"]
    )

    return {
        "group": group_name,
        "schedule": schedule,
        "trades": trades,
        "returns": returns,
        "expected": expected_df,
        "inference": result,
        "summary": summarize_trade_returns(trades),
    }


# 10. Compare earlier years with later years

def train_test_stability(returns, expected):
    rows = []

    for ticker in returns.columns:
        for label, mask in [
            ("TRAIN_<=2018", returns.index <= TRAIN_END_YEAR),
            ("TEST_>=2019", returns.index > TRAIN_END_YEAR),
        ]:
            r = returns.loc[mask, ticker]
            e = expected.loc[mask, ticker]

            if len(r) == 0:
                continue

            rows.append({
                "Ticker": ticker,
                "Sample": label,
                "N": len(r),
                "Mean_Return": r.mean(),
                "Win_Rate": (r > 0).mean(),
                "Mean_Advantage_vs_Random": (r - e).mean(),
            })

    return pd.DataFrame(rows)


# 11. Check whether one year is driving the result

def leave_one_season_out(returns, expected):
    rows = []

    for ticker in returns.columns:
        for season in returns.index:
            r = returns[ticker].drop(index=season)
            e = expected.loc[r.index, ticker]

            rows.append({
                "Ticker": ticker,
                "Omitted_Season": season,
                "Remaining_Mean_Return": r.mean(),
                "Remaining_Mean_Advantage": (r - e).mean(),
            })

    return pd.DataFrame(rows)


# 12. Compare regular and leveraged ETFs

def leverage_comparison(group_result):
    returns = group_result["returns"]

    pairs = []

    candidate_pairs = [
        ("SPY", "SSO"),
        ("SPY", "UPRO"),
        ("QQQ", "QLD"),
        ("QQQ", "TQQQ"),
    ]

    for base, levered in candidate_pairs:
        if base not in returns.columns or levered not in returns.columns:
            continue

        diff = (
            returns[levered]
            - returns[base]
        )

        low, high = bootstrap_mean_ci(
            diff.to_numpy(),
            seed=RANDOM_SEED + 500,
        )

        pairs.append({
            "Base": base,
            "Levered": levered,
            "N": len(diff),
            "Mean_Extra_Return": diff.mean(),
            "CI_Low": low,
            "CI_High": high,
            "Probability_Levered_Beats_Base": (diff > 0).mean(),
            "Base_Worst_Season": returns[base].min(),
            "Levered_Worst_Season": returns[levered].min(),
        })

    return pd.DataFrame(pairs)


# 13. Test nearby date windows

def date_window_sensitivity(
    prices,
    master_calendar,
    tickers,
    common_seasons,
):
    alternative_windows = {
        "Dec18-Jan5": (18, 5),
        "Dec20-Jan5": (20, 5),
        "Dec21-Jan3": (21, 3),
        "Dec21-Jan5": (21, 5),
        "Dec21-Jan7": (21, 7),
        "Dec22-Jan5": (22, 5),
    }

    rows = []

    for label, (start_day, end_day) in alternative_windows.items():
        alt_schedule = build_schedule(
            master_calendar,
            entry_day=start_day,
            exit_day=end_day,
        )

        available = [
            season
            for season in common_seasons
            if season in alt_schedule.index
        ]

        if not available:
            continue

        alt_schedule = alt_schedule.loc[available]

        trades = build_trade_table(
            prices,
            alt_schedule,
            tickers,
            cost_bps=COST_BPS_PER_SIDE,
        )

        for ticker, g in trades.groupby("Ticker"):
            r = g["Net_Return"]

            rows.append({
                "Ticker": ticker,
                "Window": label,
                "N": len(r),
                "Mean_Return": r.mean(),
                "Win_Rate": (r > 0).mean(),
                "Worst_Return": r.min(),
            })

    return pd.DataFrame(rows)


# 14. Check whether the market trend matters

def regime_analysis(spy_prices, primary_schedule, primary_trades):
    spy = spy_prices.copy()

    ma200 = spy.rolling(
        200,
        min_periods=200,
    ).mean()

    rows = []

    for season, row in primary_schedule.iterrows():
        earlier_dates = spy.index[
            spy.index < row.Entry
        ]

        if len(earlier_dates) == 0:
            continue

        signal_date = earlier_dates[-1]

        price = spy.loc[signal_date]
        ma = ma200.loc[signal_date]

        if pd.isna(ma):
            regime = "Unknown"
        elif price > ma:
            regime = "Above 200DMA"
        else:
            regime = "Below 200DMA"

        rows.append({
            "Season": season,
            "Signal_Date": signal_date,
            "SPY_Close": price,
            "MA200": ma,
            "Regime": regime,
        })

    regimes = pd.DataFrame(rows)

    joined = primary_trades.merge(
        regimes[["Season", "Regime"]],
        on="Season",
        how="left",
    )

    summary = (
        joined
        .groupby(["Ticker", "Regime"])["Net_Return"]
        .agg(["count", "mean", "median", "min"])
        .reset_index()
    )

    return regimes, summary


# 15. Rolling 5-year stability

def rolling_stability(returns, window=5):
    rows = []

    for ticker in returns.columns:
        series = returns[ticker].sort_index()

        rolling_mean = series.rolling(window).mean()
        rolling_win = (
            (series > 0)
            .astype(float)
            .rolling(window)
            .mean()
        )

        for season in series.index:
            rows.append({
                "Ticker": ticker,
                "Season": season,
                f"Rolling_{window}Y_Mean": rolling_mean.loc[season],
                f"Rolling_{window}Y_Win_Rate": rolling_win.loc[season],
            })

    return pd.DataFrame(rows)


# 16. Plotly charts

def save_chart(fig, filename):
    """
    Save each Plotly chart as interactive HTML and open it in a browser.

    On macOS, this explicitly opens the HTML file in Google Chrome.
    In Jupyter, the chart is also displayed inline.
    """
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Arial", size=15),
        title=dict(x=0.5, xanchor="center"),
        width=1250,
        height=720,
        margin=dict(l=85, r=60, t=95, b=85),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(font_size=14),
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False)

    # Save interactive HTML first.
    html_path = (CHART_DIR / f"{filename}.html").resolve()

    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    # Also show inline when running in Jupyter.
    try:
        fig.show(renderer="notebook_connected")
    except Exception:
        pass

    # Explicitly open the saved HTML in Google Chrome on macOS.
    opened = False

    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["open", "-a", "Google Chrome", str(html_path)]
            )
            opened = True
        except Exception as exc:
            print(f"Could not open Google Chrome directly: {exc}")

    # Fallback for other systems or if Chrome did not open.
    if not opened:
        try:
            webbrowser.open_new_tab(html_path.as_uri())
            opened = True
        except Exception as exc:
            print(f"Browser fallback failed: {exc}")

    print(f"Interactive chart saved to: {html_path}")

    # Optional PNG image for GitHub / LinkedIn.
    png_path = (CHART_DIR / f"{filename}.png").resolve()

    try:
        fig.write_image(
            png_path,
            width=1800,
            height=1000,
            scale=2,
        )
        print(f"PNG saved to: {png_path}")
    except Exception:
        print(
            "PNG not created. If you want static PNG files, run:\n"
            "    pip install -U kaleido"
        )


def make_charts(primary, unlevered):
    # --------------------------------------------------------
    # Chart 1: Primary SPY mean with bootstrap CI
    # --------------------------------------------------------
    row = primary["inference"].iloc[0]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=["SPY"],
            y=[row["Mean_Net_Return"] * 100],
            mode="markers+text",
            marker=dict(size=18, symbol="diamond"),
            text=[f"{row['Mean_Net_Return']*100:.2f}%"],
            textposition="top center",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[
                    (
                        row["Mean_CI_High"]
                        - row["Mean_Net_Return"]
                    ) * 100
                ],
                arrayminus=[
                    (
                        row["Mean_Net_Return"]
                        - row["Mean_CI_Low"]
                    ) * 100
                ],
            ),
            name="SPY",
            hovertemplate=(
                "<b>SPY</b><br>"
                "Mean Return: %{y:.2f}%<br>"
                f"95% CI: [{row['Mean_CI_Low']*100:.2f}%, {row['Mean_CI_High']*100:.2f}%]<br>"
                f"Matched Random Mean: {row['Matched_Random_Mean']*100:.2f}%<br>"
                f"Timing p-value: {row['Timing_P_One_Sided']:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash")
    fig.add_annotation(
        x="SPY",
        y=row["Matched_Random_Mean"] * 100,
        text=f"Matched Random Mean: {row['Matched_Random_Mean']*100:.2f}%",
        showarrow=True,
        arrowhead=2,
        ax=120,
        ay=-50,
    )

    fig.update_layout(
        title="Primary Test: SPY Year-End Mean Net Return with 95% Bootstrap CI",
        xaxis_title="Asset",
        yaxis_title="Net Return (%)",
    )

    save_chart(fig, "01_primary_spy_bootstrap_ci")

    # --------------------------------------------------------
    # Chart 2: Mean advantage vs matched random windows
    # --------------------------------------------------------
    inf = unlevered["inference"].copy().sort_values(
        "Mean_Advantage",
        ascending=False,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=inf["Ticker"],
            y=inf["Mean_Advantage"] * 100,
            mode="markers+text",
            marker=dict(size=15),
            text=(inf["Mean_Advantage"] * 100).round(2).astype(str) + "%",
            textposition="top center",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(
                    inf["Advantage_CI_High"]
                    - inf["Mean_Advantage"]
                ) * 100,
                arrayminus=(
                    inf["Mean_Advantage"]
                    - inf["Advantage_CI_Low"]
                ) * 100,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean Advantage: %{y:.2f}%<br>"
                "Win Rate: %{customdata[0]:.1%}<br>"
                "Matched Random Mean: %{customdata[1]:.2%}<br>"
                "Timing p-value: %{customdata[2]:.4f}<extra></extra>"
            ),
            customdata=np.column_stack([
                inf["Win_Rate"],
                inf["Matched_Random_Mean"],
                inf["Timing_P_One_Sided"],
            ]),
        )
    )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title="Seasonal Advantage vs Matched Random Windows",
        xaxis_title="Ticker",
        yaxis_title="Advantage (percentage points)",
    )

    save_chart(fig, "02_advantage_vs_random")

    # --------------------------------------------------------
    # Chart 3: Year-by-year heatmap
    # --------------------------------------------------------
    r = unlevered["returns"] * 100

    limit = np.nanmax(np.abs(r.to_numpy()))

    fig = px.imshow(
        r.T,
        text_auto=".1f",
        aspect="auto",
        zmin=-limit,
        zmax=limit,
        color_continuous_scale="RdBu",
        title="Year-by-Year Net Returns: Dec 21 to Jan 5 Strategy (%)",
        labels={
            "x": "December Entry Year",
            "y": "Ticker",
            "color": "Return (%)",
        },
    )

    fig.update_layout(coloraxis_colorbar_title="Return (%)")
    save_chart(fig, "03_year_by_year_heatmap")

    # --------------------------------------------------------
    # Chart 4: Rolling 5-year SPY mean
    # --------------------------------------------------------
    roll = rolling_stability(
        primary["returns"],
        window=5,
    )

    fig = px.line(
        roll,
        x="Season",
        y="Rolling_5Y_Mean",
        color="Ticker",
        markers=True,
        title="Rolling 5-Season SPY Mean Return",
        labels={
            "Season": "December Entry Year",
            "Rolling_5Y_Mean": "Rolling Mean Return",
        },
    )

    fig.update_yaxes(tickformat=".1%")
    save_chart(fig, "04_rolling_5y_mean")

    # --------------------------------------------------------
    # Chart 5: Mean return by unlevered asset
    # --------------------------------------------------------
    summary = unlevered["summary"].copy().sort_values(
        "Mean_Return",
        ascending=False,
    )

    fig = px.bar(
        summary,
        x="Ticker",
        y=summary["Mean_Return"] * 100,
        text=summary["Mean_Return"] * 100,
        title="Average Dec. 21–Jan. 5 Net Return",
        labels={"y": "Mean return (%)", "x": "Ticker"},
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean Return: %{y:.2f}%<br>"
            "Win Rate: %{customdata[0]:.1%}<br>"
            "Volatility: %{customdata[1]:.2%}<extra></extra>"
        ),
        customdata=np.column_stack([
            summary["Win_Rate"],
            summary["Seasonal_Volatility"],
        ]),
    )

    fig.add_hline(y=0, line_dash="dash")
    save_chart(fig, "05_average_returns")

    # --------------------------------------------------------
    # Chart 6: Seasonal volatility vs mean return
    # --------------------------------------------------------
    fig = px.scatter(
        summary,
        x=summary["Seasonal_Volatility"] * 100,
        y=summary["Mean_Return"] * 100,
        text="Ticker",
        size=summary["Win_Rate"] * 100,
        hover_data=["Worst_Return", "Sortino_Zero_Target", "Profit_Factor"],
        title="Seasonality Risk vs Return",
        labels={
            "x": "Seasonal volatility (%)",
            "y": "Mean return (%)",
        },
    )

    fig.update_traces(
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Seasonal Volatility: %{x:.2f}%<br>"
            "Mean Return: %{y:.2f}%<br>"
            "Bubble Size = Win Rate<extra></extra>"
        ),
    )

    save_chart(fig, "06_risk_vs_return")

    # --------------------------------------------------------
    # Chart 7: Win rate
    # --------------------------------------------------------
    win_df = summary.sort_values(
        "Win_Rate",
        ascending=False,
    ).copy()

    fig = px.bar(
        win_df,
        x="Ticker",
        y=win_df["Win_Rate"] * 100,
        text=win_df["Win_Rate"] * 100,
        title="Year-End Strategy Win Rate",
        labels={"y": "Win rate (%)", "x": "Ticker"},
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
    )

    fig.update_yaxes(range=[0, 100])
    save_chart(fig, "07_win_rate")

    # --------------------------------------------------------
    # Chart 8: Worst in-trade drawdown
    # --------------------------------------------------------
    dd_df = summary.sort_values(
        "Worst_In_Trade_Drawdown"
    ).copy()

    fig = px.bar(
        dd_df,
        x="Ticker",
        y=dd_df["Worst_In_Trade_Drawdown"] * 100,
        text=dd_df["Worst_In_Trade_Drawdown"] * 100,
        title="Worst In-Trade Drawdown",
        labels={"y": "Drawdown (%)", "x": "Ticker"},
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Worst In-Trade Drawdown: %{y:.2f}%<extra></extra>",
    )

    save_chart(fig, "08_worst_drawdown")



def make_leverage_charts(sp500_group, nasdaq_group):
    # I kept the leverage charts separate because comparing a 3x ETF
    # directly with an unlevered ETF can be misleading if the extra risk
    # is not shown at the same time.

    sp = sp500_group["summary"].copy()
    nq = nasdaq_group["summary"].copy()

    # S&P 500 family: SPY vs SSO vs UPRO
    fig = px.bar(
        sp.sort_values("Mean_Return", ascending=False),
        x="Ticker",
        y=sp.sort_values("Mean_Return", ascending=False)["Mean_Return"] * 100,
        text=sp.sort_values("Mean_Return", ascending=False)["Mean_Return"] * 100,
        title="S&P 500 Year-End Return: SPY vs SSO vs UPRO",
        labels={"y": "Average net return (%)", "x": "ETF"},
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )
    fig.add_hline(y=0, line_dash="dash")
    save_chart(fig, "09_sp500_leverage_average_return")

    # Nasdaq family: QQQ vs QLD vs TQQQ
    fig = px.bar(
        nq.sort_values("Mean_Return", ascending=False),
        x="Ticker",
        y=nq.sort_values("Mean_Return", ascending=False)["Mean_Return"] * 100,
        text=nq.sort_values("Mean_Return", ascending=False)["Mean_Return"] * 100,
        title="Nasdaq-100 Year-End Return: QQQ vs QLD vs TQQQ",
        labels={"y": "Average net return (%)", "x": "ETF"},
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )
    fig.add_hline(y=0, line_dash="dash")
    save_chart(fig, "10_nasdaq_leverage_average_return")

    # Put both families together for risk vs return.
    lev = pd.concat(
        [
            sp.assign(Family="S&P 500"),
            nq.assign(Family="Nasdaq-100"),
        ],
        ignore_index=True,
    )

    fig = px.scatter(
        lev,
        x=lev["Seasonal_Volatility"] * 100,
        y=lev["Mean_Return"] * 100,
        text="Ticker",
        symbol="Family",
        size=lev["Win_Rate"] * 100,
        hover_data=[
            "Win_Rate",
            "Worst_Return",
            "Worst_In_Trade_Drawdown",
            "Sortino_Zero_Target",
        ],
        title="Leverage Trade-Off: Return vs Risk",
        labels={
            "x": "Seasonal volatility (%)",
            "y": "Average net return (%)",
        },
    )
    fig.update_traces(textposition="top center")
    save_chart(fig, "11_leverage_risk_vs_return")

    # Year-by-year leveraged ETF heatmap.
    sp_r = sp500_group["returns"].copy()
    nq_r = nasdaq_group["returns"].copy()

    common_years = sp_r.index.intersection(nq_r.index)
    leverage_returns = pd.concat(
        [
            sp_r.loc[common_years, ["SPY", "SSO", "UPRO"]],
            nq_r.loc[common_years, ["QQQ", "QLD", "TQQQ"]],
        ],
        axis=1,
    ) * 100

    limit = np.nanmax(np.abs(leverage_returns.to_numpy()))

    fig = px.imshow(
        leverage_returns.T,
        text_auto=".1f",
        aspect="auto",
        zmin=-limit,
        zmax=limit,
        color_continuous_scale="RdBu",
        title="Year-by-Year Returns: 1x vs 2x vs 3x ETFs",
        labels={
            "x": "December entry year",
            "y": "ETF",
            "color": "Return (%)",
        },
    )
    save_chart(fig, "12_leverage_year_by_year_heatmap")

    # Win rate vs worst seasonal return gives another useful risk view.
    fig = px.scatter(
        lev,
        x=lev["Win_Rate"] * 100,
        y=lev["Worst_Return"] * 100,
        text="Ticker",
        symbol="Family",
        size=np.abs(lev["Mean_Return"] * 100) + 1,
        title="Leverage: Win Rate vs Worst Year-End Return",
        labels={
            "x": "Win rate (%)",
            "y": "Worst seasonal return (%)",
        },
    )
    fig.update_traces(textposition="top center")
    save_chart(fig, "13_leverage_win_rate_vs_worst_return")

# 17. Save a short project summary

def write_research_report(
    primary,
    unlevered,
    train_test,
    regime_summary,
):
    p = primary["inference"].iloc[0]

    lines = [
        "# Year-End Equity Seasonality Research",
        "",
        "## Research question",
        "",
        "Is the Dec. 21 -> Jan. 5 SPY return unusually strong relative to comparable non-seasonal windows over the 2008 through 2025 seasonal sample?",
        "",
        "## Primary result",
        "",
        f"- Seasons tested: {int(p['N_Seasons'])}",
        f"- Mean SPY net return: {p['Mean_Net_Return']:.2%}",
        f"- 95% block-bootstrap CI: [{p['Mean_CI_Low']:.2%}, {p['Mean_CI_High']:.2%}]",
        f"- Matched random-window mean: {p['Matched_Random_Mean']:.2%}",
        f"- Mean timing advantage: {p['Mean_Advantage']:.2%}",
        f"- Random-window one-sided p-value: {p['Timing_P_One_Sided']:.4f}",
        "",
        "The random-window comparison is important because a positive return by itself does not prove there is a seasonal effect. Stocks generally have a positive long-run return.",
        "",
        "## Research design",
        "",
        "- Primary hypothesis is SPY.",
        "- Other assets are secondary replication tests.",
        "- Entry uses first trading session on/after Dec. 21.",
        "- Exit uses last trading session on/before Jan. 5.",
        f"- Transaction costs: {COST_BPS_PER_SIDE:.1f} bps per side.",
        f"- Random-window simulations: {RANDOM_WINDOW_SIMULATIONS:,}.",
        f"- Bootstrap simulations: {BOOTSTRAP_SIMULATIONS:,}.",
        f"- Historical train/test split: <= {TRAIN_END_YEAR} vs >= {TRAIN_END_YEAR + 1}.",
        "",
        "## Important limitations",
        "",
        "- This is a historical backtest, so it does not prove the strategy will work in the future.",
        "- There are only a limited number of year-end observations.",
        "- Leveraged ETFs are related to their underlying indexes, so they should not be treated as completely independent evidence.",
        "- The project uses adjusted closing prices, so real trading prices could be slightly different.",
        "- Taxes, changing bid-ask spreads, market impact, and intraday execution are not included.",
        "- The later sample is a historical stability check, not a pristine untouched future holdout.",
        "",
        "## Suggested review order",
        "",
        "1. primary_inference.csv",
        "2. primary_train_test.csv",
        "3. unlevered_inference.csv",
        "4. date_sensitivity.csv",
        "5. primary_leave_one_out.csv",
        "6. regime_summary.csv",
        "7. charts/",
    ]

    (OUTPUT_DIR / "RESEARCH_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# 18. Run the full project

def run_research():
    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    prices = load_all_prices()

    # use SPY dates as the trading calendar
    master_calendar = prices["SPY"].index

    base_schedule = build_schedule(
        master_calendar,
        entry_day=ENTRY_DAY,
        exit_day=EXIT_DAY,
    )

    print("\nBASE SCHEDULE")
    print(base_schedule.to_string())

    # --------------------------------------------------------
    # run the tests
    # --------------------------------------------------------
    results = {}

    for group_name, tickers in RESEARCH_GROUPS.items():
        print("\n" + "=" * 90)
        print(group_name)
        print("=" * 90)

        result = run_group_inference(
            group_name,
            tickers,
            prices,
            master_calendar,
            base_schedule,
        )

        results[group_name] = result

        print(
            result["inference"][
                [
                    "Ticker",
                    "N_Seasons",
                    "Mean_Net_Return",
                    "Mean_Advantage",
                    "Timing_P_One_Sided",
                    "Timing_Q_BH",
                ]
            ].to_string(index=False)
        )

        safe_name = group_name.lower()

        result["trades"].to_csv(
            OUTPUT_DIR / f"{safe_name}_trades.csv",
            index=False,
        )

        result["returns"].to_csv(
            OUTPUT_DIR / f"{safe_name}_returns.csv",
        )

        result["inference"].to_csv(
            OUTPUT_DIR / f"{safe_name}_inference.csv",
            index=False,
        )

        result["summary"].to_csv(
            OUTPUT_DIR / f"{safe_name}_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # extra checks for SPY
    # --------------------------------------------------------
    primary = results["PRIMARY_SPY"]

    primary_train_test = train_test_stability(
        primary["returns"],
        primary["expected"],
    )

    primary_leave_one_out = leave_one_season_out(
        primary["returns"],
        primary["expected"],
    )

    primary_rolling = rolling_stability(
        primary["returns"],
        window=5,
    )

    primary_train_test.to_csv(
        OUTPUT_DIR / "primary_train_test.csv",
        index=False,
    )

    primary_leave_one_out.to_csv(
        OUTPUT_DIR / "primary_leave_one_out.csv",
        index=False,
    )

    primary_rolling.to_csv(
        OUTPUT_DIR / "primary_rolling_stability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # nearby date windows
    # --------------------------------------------------------
    unlevered = results["UNLEVERED_REPLICATION"]

    sensitivity = date_window_sensitivity(
        prices,
        master_calendar,
        RESEARCH_GROUPS["UNLEVERED_REPLICATION"],
        unlevered["schedule"].index,
    )

    sensitivity.to_csv(
        OUTPUT_DIR / "date_sensitivity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 200-day moving-average check
    # --------------------------------------------------------
    regimes, regime_summary = regime_analysis(
        prices["SPY"],
        primary["schedule"],
        primary["trades"],
    )

    regimes.to_csv(
        OUTPUT_DIR / "primary_regime_labels.csv",
        index=False,
    )

    regime_summary.to_csv(
        OUTPUT_DIR / "regime_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # leveraged ETF comparison
    # --------------------------------------------------------
    leverage_tables = []

    for group_name in [
        "SP500_LEVERAGE_FAMILY",
        "NASDAQ_LEVERAGE_FAMILY",
    ]:
        table = leverage_comparison(
            results[group_name]
        )

        table["Group"] = group_name
        leverage_tables.append(table)

    leverage_results = pd.concat(
        leverage_tables,
        ignore_index=True,
    )

    leverage_results.to_csv(
        OUTPUT_DIR / "leverage_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # correlation
    # --------------------------------------------------------
    unlevered["returns"].corr().to_csv(
        OUTPUT_DIR / "unlevered_seasonal_correlation.csv"
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------
    make_charts(
        primary,
        unlevered,
    )

    make_leverage_charts(
        results["SP500_LEVERAGE_FAMILY"],
        results["NASDAQ_LEVERAGE_FAMILY"],
    )

    # --------------------------------------------------------
    # save a short write-up
    # --------------------------------------------------------
    write_research_report(
        primary,
        unlevered,
        primary_train_test,
        regime_summary,
    )

    # --------------------------------------------------------
    # print the main results
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("PRIMARY SPY RESULT")
    print("=" * 90)

    print(
        primary["inference"][
            [
                "Ticker",
                "N_Seasons",
                "Mean_Net_Return",
                "Mean_CI_Low",
                "Mean_CI_High",
                "Matched_Random_Mean",
                "Mean_Advantage",
                "Timing_P_One_Sided",
            ]
        ].to_string(index=False)
    )

    print("\nTRAIN / TEST STABILITY")
    print(primary_train_test.to_string(index=False))

    print("\nLEVERAGE COMPARISON")
    print(leverage_results.to_string(index=False))

    print("\nDone.")
    print("Results saved to:")
    print(OUTPUT_DIR)

    print("\nPlotly charts saved to:")
    print(CHART_DIR)
    print(
        "\nOpen the .html files in that folder for interactive charts. The leverage charts are files 09 through 13. "
        "PNG files will also be saved when kaleido is installed."
    )

    return {
        "prices": prices,
        "base_schedule": base_schedule,
        "results": results,
        "primary_train_test": primary_train_test,
        "primary_leave_one_out": primary_leave_one_out,
        "date_sensitivity": sensitivity,
        "regime_summary": regime_summary,
        "leverage_results": leverage_results,
    }


# 19. Run everything

research = run_research()
