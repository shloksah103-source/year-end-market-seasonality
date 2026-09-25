# Year-End Market Seasonality Study

I built this project because I was curious whether the market tends to behave differently around the end of the year.

The main question I wanted to test was:

**What happens if you buy around Dec. 21 and hold until Jan. 5?**

I used SPY as the main benchmark and then compared the same idea across QQQ, IWM, IWB, XLP, WMT, and leveraged ETFs such as SSO, UPRO, QLD, and TQQQ.

The study covers year-end periods from **2008 through 2025**, depending on the available history for each ETF.

## Why I Built It

I originally wanted to see whether the commonly discussed year-end or “Santa Claus” effect actually showed up in historical data.

Instead of stopping at average return and win rate, I added several tests to see whether the result was unusual, whether it stayed consistent over time, and how leverage affected the outcome.

## Methodology

The project includes:

- Dec. 21 to Jan. 5 historical backtesting
- transaction costs
- matched random-window comparisons
- bootstrap confidence intervals
- earlier-period vs. later-period testing
- leave-one-year-out analysis
- nearby date-window sensitivity
- 200-day moving-average regime analysis
- leveraged vs. unleveraged ETF comparisons
- rolling five-year analysis

## Main Result

For SPY, the Dec. 21 to Jan. 5 strategy produced an average net return of approximately **1.03%** across 18 year-end periods.

Comparable random windows averaged approximately **0.29%**, giving the year-end window an average advantage of about **0.74 percentage points**.

However, the random-window test produced a p-value of approximately **0.16**, and the bootstrap confidence interval included zero.

Because of that, I would not interpret the result as proof of a reliable market anomaly. The historical pattern is interesting, but the statistical evidence is not strong enough to conclude that the timing effect is persistent.

## Earlier vs. Later Period

One of the most interesting findings was the difference between the earlier and later parts of the sample.

### 2008–2018

- Average SPY return: approximately **1.78%**
- Win rate: approximately **82%**

### 2019–2025

- Average SPY return: approximately **-0.15%**
- Win rate: approximately **57%**

The year-end effect was much stronger in the earlier sample and weakened considerably in more recent years.

## Leveraged ETFs

I also compared regular ETFs with their leveraged versions.

For the S&P 500 group:

| ETF | Average Year-End Return |
|---|---:|
| SPY | 0.63% |
| SSO | 1.21% |
| UPRO | 1.74% |

Higher leverage increased the average return, but it also increased downside.

| ETF | Worst Year-End Return |
|---|---:|
| SPY | -2.87% |
| SSO | -5.66% |
| UPRO | -8.35% |

The Nasdaq group showed a weaker seasonal effect overall. QLD and TQQQ increased downside substantially without showing a clear improvement in the seasonal signal.

## Visualizations

The project creates 13 interactive Plotly charts covering:

- SPY bootstrap confidence intervals
- seasonal returns vs. matched random windows
- year-by-year return heatmaps
- rolling five-year performance
- average return comparisons
- risk vs. return
- win rates
- in-trade drawdowns
- SPY vs. SSO vs. UPRO
- QQQ vs. QLD vs. TQQQ
- leverage risk vs. return
- 1x, 2x, and 3x year-by-year comparisons
- win rate vs. worst seasonal return

All interactive charts are available in the `charts/` folder.

## What I Took Away From the Project

The biggest takeaway for me was that a strategy can look attractive when you only look at average return or win rate.

Once I compared the year-end window with random periods, split the sample into earlier and later years, and looked at leverage and downside risk, the result became much less straightforward.

The project also showed me why testing robustness matters. A historical pattern can exist without being statistically strong enough to treat as a dependable trading signal.

## Tools

- Python
- pandas
- NumPy
- SciPy
- yfinance
- Plotly

## Repository Structure

- `seasonality_research.py` — main analysis and backtest
- `README.md` — project overview and results
- `requirements.txt` — required Python packages
- `charts/` — interactive Plotly visualizations

## Disclaimer

This project is for educational and research purposes only. Historical results do not guarantee future performance and should not be considered investment advice.
