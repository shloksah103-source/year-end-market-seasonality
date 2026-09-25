 # Year-End Market Seasonality Study

I built this project because I was curious whether the market tends to behave differently around the end of the year.

The main question I tested was:

**What happens if you buy around Dec. 21 and hold until Jan. 5?**

I used SPY as the main benchmark, then compared the pattern across other ETFs including QQQ, IWM, IWB, XLP, WMT, and leveraged ETFs such as SSO, UPRO, QLD, and TQQQ.

The study covers year-end periods from **2008 through 2025**, depending on the available history for each ETF.

## Why I Built It

At first I only wanted to check whether the commonly discussed year-end or “Santa Claus” effect actually showed up in the data.

But instead of stopping at average return or win rate, I added more tests to see whether the pattern was unusual and whether it stayed stable over time.

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

Because of that, I would not interpret the result as proof of a reliable market anomaly.

## Earlier vs. Later Period

One of the most interesting results was the difference between the earlier and later samples.

### 2008–2018

- Average SPY return: approximately **1.78%**
- Win rate: approximately **82%**

### 2019–2025

- Average SPY return: approximately **-0.15%**
- Win rate: approximately **57%**

The year-end effect was much stronger in the earlier part of the sample and weakened considerably in more recent years.

## Leveraged ETFs

I also compared regular ETFs with their leveraged versions.

For the S&P 500 group:

| ETF | Average Year-End Return |
|---|---:|
| SPY | 0.63% |
| SSO | 1.21% |
| UPRO | 1.74% |

Higher leverage increased average return, but it also increased downside substantially.

Worst year-end return in the common sample:

| ETF | Worst Return |
|---|---:|
| SPY | -2.87% |
| SSO | -5.66% |
| UPRO | -8.35% |

The Nasdaq group showed a weaker seasonal effect overall, and leverage increased downside risk more than it improved the signal.

## Tools

- Python
- pandas
- NumPy
- SciPy
- yfinance
- Plotly

## Files

- `seasonality_research.py` — main Python project file
- `README.md` — project overview
- `requirements.txt` — Python packages needed
- `charts/` — chart images for the project

## What I Learned

The biggest takeaway for me was that a strategy can look attractive when only average return and win rate are considered.

Once I compared the strategy with random periods, split the sample into earlier and later years, and looked at leverage and downside risk, the conclusion became much more nuanced.

This project was mainly a way for me to practice backtesting, statistics, and market research with Python.

## Disclaimer

This project is for educational and research purposes only. Historical results do not guarantee future performance and should not be considered investment advice.
