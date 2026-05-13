"""
backtest.py — Lightweight backtester for wheat futures trading strategies.
Uses numpy/pandas only. No external backtesting frameworks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    """Lightweight backtester using numpy/pandas. No external frameworks."""

    def __init__(self, price_series: pd.Series):
        """
        Args:
            price_series: Daily closing prices aligned to strategy dates.
        """
        self.prices = price_series
        self.log_returns = np.log(price_series / price_series.shift(1)).dropna()

    def run(self, signals: pd.Series, label: str = 'Strategy') -> dict:
        """
        Run backtest given a signal series.

        Args:
            signals: pd.Series of +1, -1, or 0 aligned to price_series index.
                     Signal on day t is applied to return on day t+1.
            label: Name for plot legend.

        Returns:
            dict with keys: cumulative_return, sharpe_ratio, max_drawdown,
                            annual_return, num_trades, win_rate
        """
        # Align signal to returns (shift by 1 — signal on t, return on t+1)
        aligned_signal = signals.reindex(self.log_returns.index).shift(1).fillna(0)
        strat_returns = aligned_signal * self.log_returns

        # Cumulative return (in linear space)
        cum_returns = (1 + strat_returns).cumprod()

        # Annualized Sharpe ratio (252 trading days)
        sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)

        # Maximum drawdown
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Annualized return
        n_years = len(strat_returns) / 252
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Trade count (number of signal changes)
        num_trades = (aligned_signal.diff().abs() > 0).sum()

        # Win rate (fraction of positive return days when in position)
        in_position = aligned_signal != 0
        win_rate = (strat_returns[in_position] > 0).mean()

        return {
            'cumulative_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(max_drawdown, 4),
            'num_trades': int(num_trades),
            'win_rate': round(win_rate, 4),
            'equity_curve': cum_returns,
            'strategy_returns': strat_returns,
            'label': label
        }

    def plot_equity_curve(self, results_list: list, title: str = 'Equity Curve'):
        """
        Plot equity curves for one or more strategies + buy-and-hold.

        Args:
            results_list: list of dicts returned by run()
            title: plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                  gridspec_kw={'height_ratios': [3, 1]})

        # Top panel: equity curves
        bh = (1 + self.log_returns).cumprod()
        axes[0].plot(bh.index, bh.values, 'k--', alpha=0.5, label='Buy & Hold')

        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        for i, res in enumerate(results_list):
            axes[0].plot(res['equity_curve'].index,
                         res['equity_curve'].values,
                         color=colors[i % len(colors)],
                         label=res['label'])

        axes[0].set_title(title)
        axes[0].set_ylabel('Portfolio Value (starting at 1.0)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Bottom panel: drawdown of first strategy
        rolling_max = results_list[0]['equity_curve'].cummax()
        drawdown = (results_list[0]['equity_curve'] - rolling_max) / rolling_max
        axes[1].fill_between(drawdown.index, drawdown.values, 0,
                              alpha=0.4, color='red', label='Drawdown')
        axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
        plt.show()

    def print_metrics_table(self, results_list: list):
        """Print a formatted performance comparison table."""
        print(f"\n{'='*65}")
        print(f"{'Metric':<25} " +
              " ".join(f"{r['label']:>15}" for r in results_list))
        print(f"{'='*65}")
        metrics = [
            ('Cumulative Return', 'cumulative_return', '{:.2%}'),
            ('Annual Return',     'annual_return',     '{:.2%}'),
            ('Sharpe Ratio',      'sharpe_ratio',      '{:.3f}'),
            ('Max Drawdown',      'max_drawdown',      '{:.2%}'),
            ('Num Trades',        'num_trades',        '{:d}'),
            ('Win Rate',          'win_rate',          '{:.2%}'),
        ]
        for label, key, fmt in metrics:
            row = f"{label:<25} "
            for r in results_list:
                val = r[key]
                row += f"{fmt.format(val):>15} "
            print(row)
        print(f"{'='*65}\n")
