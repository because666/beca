import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeAction(Enum):
    BUY = 'buy'
    SELL = 'sell'

class BacktestEngine:
    def __init__(self, initial_cash: float = 100000, commission: float = 0.001, slippage: float = 0.001, 
                 buy_threshold: float = 0.5, sell_threshold: float = 0.5, stop_loss_threshold: float = 0.1,
                 max_hold_days: int = 5, max_position_pct: float = 0.2, max_positions: int = 5):
        self._validate_params(initial_cash, commission, slippage, buy_threshold, sell_threshold, 
                            stop_loss_threshold, max_hold_days, max_position_pct, max_positions)
        
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_hold_days = max_hold_days
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_value: List[Dict[str, Any]] = []
        self.benchmark_value: List[float] = []
        self.trailing_stops: Dict[str, float] = {} # Store highest price for each position
        self.trailing_stop_pct = 0.05 # Default trailing stop percentage (5%)
        self.debug_mode = True

    def _validate_params(self, initial_cash, commission, slippage, buy_threshold, sell_threshold,
                        stop_loss_threshold, max_hold_days, max_position_pct, max_positions):
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not (0 <= commission < 1):
            raise ValueError("Commission must be between 0 and 1")
        if not (0 <= slippage < 1):
            raise ValueError("Slippage must be between 0 and 1")
        if not (0 <= buy_threshold <= 1):
            raise ValueError("Buy threshold must be between 0 and 1")
        if not (0 <= sell_threshold <= 1):
            raise ValueError("Sell threshold must be between 0 and 1")
        if max_hold_days <= 0:
            raise ValueError("Max hold days must be positive")
        if not (0 < max_position_pct <= 1):
            raise ValueError("Max position percentage must be between 0 and 1")
        if max_positions <= 0:
            raise ValueError("Max positions must be positive")

    def run_backtest(self, df: pd.DataFrame, model: Any, feature_cols: List[str], 
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        self.reset()
        
        if self.debug_mode:
            logger.info("=" * 60)
            logger.info("开始回测")
            logger.info("=" * 60)
            logger.info(f"初始资金: ¥{self.initial_cash:,}")
            logger.info(f"买入阈值: {self.buy_threshold}")
            logger.info(f"卖出阈值: {self.sell_threshold}")
            logger.info(f"止损阈值: {self.stop_loss_threshold}")
            logger.info(f"最大持仓天数: {self.max_hold_days}")
            logger.info(f"最大仓位比例: {self.max_position_pct:.1%}")
            logger.info(f"最大持仓数量: {self.max_positions}")
        
        # Date filtering
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        df = df.sort_values('date').reset_index(drop=True)
        
        if self.debug_mode:
            logger.info(f"回测数据范围: {df['date'].min()} 至 {df['date'].max()}")
            logger.info(f"回测数据量: {len(df)} 条")
        
        # Batch prediction optimization
        logger.info("Running batch predictions...")
        
        # Ensure features are numeric
        features_df = df[feature_cols]
        
        # Check dtypes and log if suspicious
        if features_df.select_dtypes(include=['object']).shape[1] > 0:
            logger.warning(f"Found object columns in features: {features_df.select_dtypes(include=['object']).columns.tolist()}. Converting to numeric.")
        
        # Force conversion to numeric, coercing errors to NaN
        all_features = features_df.apply(pd.to_numeric, errors='coerce').values
        
        # Handle cases where feature columns might have NaNs
        if np.isnan(all_features).any():
             logger.warning("Features contain NaN values. Filling with 0.")
             all_features = np.nan_to_num(all_features)

        try:
            predictions = model.predict(all_features)
            # Check if predict_proba returns (N, 2) or just (N,) depending on model
            probas = model.predict_proba(all_features)
            if probas.ndim == 2 and probas.shape[1] == 2:
                probabilities = probas[:, 1]
            else:
                probabilities = probas # Assume it returns probability of positive class directly
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

        # Add predictions to a temporary dataframe for iteration
        # We don't modify original df to avoid side effects if reused
        sim_df = df.copy()
        sim_df['prediction'] = predictions
        sim_df['probability'] = probabilities

        buy_signals = 0
        sell_signals = 0
        buy_rejected_cash = 0
        buy_rejected_position = 0
        buy_rejected_threshold = 0
        
        # Use itertuples for faster iteration
        # Refactor: Iterate by date to ensure portfolio value is updated once per day
        unique_dates = sim_df['date'].sort_values().unique()
        
        for date in tqdm(unique_dates, desc="Running backtest"):
            # Get all stocks for this date
            day_data = sim_df[sim_df['date'] == date]
            
            # Process trading signals for each stock
            for row in day_data.itertuples():
                stock_code = row.stock_code
                close_price = row.close
                prediction = row.prediction
                probability = row.probability

                # Handle existing positions
                if stock_code in self.positions:
                    should_sell, sell_reason = self._check_sell_conditions(
                        stock_code, date, close_price, probability
                    )
                    
                    if should_sell:
                        self.sell_stock(stock_code, close_price, date, probability)
                        sell_signals += 1
                        if self.debug_mode and sell_signals % 100 == 0: 
                            logger.info(f"[卖出] {pd.to_datetime(date).date()} {stock_code} 原因: {sell_reason}")
                        elif self.debug_mode and sell_signals < 10: 
                             logger.info(f"[卖出] {pd.to_datetime(date).date()} {stock_code} 原因: {sell_reason}")

                # Handle buy signals
                if probability > self.buy_threshold and stock_code not in self.positions:
                    status, reason = self._check_buy_conditions(close_price)
                    
                    if status == 'ok':
                        self.buy_stock(stock_code, close_price, date, probability)
                        buy_signals += 1
                        if self.debug_mode and buy_signals % 100 == 0:
                             logger.info(f"[买入] {pd.to_datetime(date).date()} {stock_code} 价格:{close_price:.2f} 概率:{probability:.4f}")
                        elif self.debug_mode and buy_signals < 10:
                             logger.info(f"[买入] {pd.to_datetime(date).date()} {stock_code} 价格:{close_price:.2f} 概率:{probability:.4f}")

                    elif status == 'rejected_cash':
                        buy_rejected_cash += 1
                    elif status == 'rejected_position':
                        buy_rejected_position += 1
            
            # End of day: Update portfolio value
            # Pass day_data which contains all stocks for this day
            self.update_portfolio_value(date, None, day_data)
        
        self.close_all_positions(df)
        
        if self.debug_mode:
            self._log_statistics(buy_signals, sell_signals, buy_rejected_cash, 
                               buy_rejected_position, buy_rejected_threshold)
        
        return self.get_backtest_results()

    def _check_sell_conditions(self, stock_code, date, close_price, probability):
        position = self.positions[stock_code]
        days_held = (date - position['entry_date']).days
        current_return = (close_price - position['entry_price']) / position['entry_price']
        
        # Update trailing stop high water mark
        if stock_code not in self.trailing_stops:
            self.trailing_stops[stock_code] = close_price
        else:
            self.trailing_stops[stock_code] = max(self.trailing_stops[stock_code], close_price)
            
        # Check trailing stop
        high_price = self.trailing_stops[stock_code]
        drawdown_from_high = (high_price - close_price) / high_price
        
        # Only trigger trailing stop if we are in profit overall (optional, but good practice)
        # Or strictly follow the rule: if price drops X% from peak, sell.
        if drawdown_from_high >= self.trailing_stop_pct and current_return > 0:
             return True, f"触发移动止盈(回撤{drawdown_from_high:.1%})"
        
        if days_held >= self.max_hold_days:
            return True, f"达到最大持仓天数({days_held}天)"
        elif current_return <= -self.stop_loss_threshold:
            return True, f"触发止损({current_return:.2%})"
        elif probability < self.sell_threshold:
            return True, f"预测概率低于阈值({probability:.4f} < {self.sell_threshold})"
            
        return False, ""

    def _check_buy_conditions(self, price):
        if self.cash <= 0:
            return 'rejected_cash', "资金不足"
        
        if len(self.positions) >= self.max_positions:
            return 'rejected_position', f"持仓已满({len(self.positions)}/{self.max_positions})"
            
        return 'ok', ""
        
    def calculate_position_size(self, probability: float, price: float) -> int:
        """
        Calculate position size based on Kelly Criterion or Probability Scaling.
        Simplified Kelly: f = p - q (where p is win prob, q is loss prob)
        Here we scale max_position_pct based on probability confidence.
        """
        # Base scale: linearly map probability 0.5-0.8 to 0.5-1.0 of max_position_pct
        # If prob < 0.5, we shouldn't be buying anyway
        
        scale_factor = min(1.0, max(0.5, (probability - 0.5) / 0.3 * 0.5 + 0.5))
        # Example: 
        # prob=0.5 -> scale=0.5 (Half position)
        # prob=0.65 -> scale=0.75
        # prob=0.8 -> scale=1.0 (Full position)
        
        target_pct = self.max_position_pct * scale_factor
        max_val = self.cash * target_pct
        
        return int(max_val / (price * (1 + self.slippage)))

    def _log_statistics(self, buy_signals, sell_signals, buy_rejected_cash, 
                       buy_rejected_position, buy_rejected_threshold):
        logger.info("=" * 60)
        logger.info("回测统计")
        logger.info("=" * 60)
        logger.info(f"买入信号数: {buy_signals}")
        logger.info(f"卖出信号数: {sell_signals}")
        logger.info(f"拒绝买入(资金不足): {buy_rejected_cash}")
        logger.info(f"拒绝买入(持仓已满): {buy_rejected_position}")
        logger.info(f"拒绝买入(概率不足): {buy_rejected_threshold}")
        logger.info(f"总交易数: {len(self.trades)}")
        logger.info(f"最终现金: ¥{self.cash:,.2f}")
        logger.info(f"持仓数量: {len(self.positions)}")
        
        return self.get_backtest_results()

    def buy_stock(self, stock_code, price, date, probability):
        if self.cash <= 0:
            return

        # Use dynamic position sizing
        shares = self.calculate_position_size(probability, price)
        
        if shares > 0:
            actual_price = price * (1 + self.slippage)
            cost = shares * actual_price * (1 + self.commission)
            
            if cost <= self.cash:
                self.cash -= cost
                self.positions[stock_code] = {
                    'shares': shares,
                    'entry_price': actual_price,
                    'entry_date': date,
                    'probability': probability
                }
                
                # Init trailing stop
                self.trailing_stops[stock_code] = actual_price
                
                self.trades.append({
                    'date': date,
                    'stock_code': stock_code,
                    'action': 'buy',
                    'price': actual_price,
                    'shares': shares,
                    'value': cost,
                    'probability': probability,
                    'profit': 0
                })

    def sell_stock(self, stock_code, price, date, probability):
        if stock_code not in self.positions:
            return

        position = self.positions[stock_code]
        actual_price = price * (1 - self.slippage)
        revenue = position['shares'] * actual_price * (1 - self.commission)
        
        self.cash += revenue
        
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'sell',
            'price': actual_price,
            'shares': position['shares'],
            'value': revenue,
            'probability': probability,
            'profit': revenue - (position['shares'] * position['entry_price'] * (1 + self.commission))
        })
        
        del self.positions[stock_code]

    def update_portfolio_value(self, date, current_price, current_df):
        total_value = self.cash
        
        for stock_code, position in self.positions.items():
            stock_data = current_df[current_df['stock_code'] == stock_code]
            if not stock_data.empty:
                stock_price = stock_data['close'].iloc[0]
                total_value += position['shares'] * stock_price
        
        self.portfolio_value.append({
            'date': date,
            'value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })

    def close_all_positions(self, df):
        for stock_code in list(self.positions.keys()):
            stock_data = df[df['stock_code'] == stock_code]
            if not stock_data.empty:
                last_price = stock_data['close'].iloc[-1]
                last_date = stock_data['date'].iloc[-1]
                self.sell_stock(stock_code, last_price, last_date, 0)

    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.benchmark_value = []

    def get_backtest_results(self):
        if not self.portfolio_value:
            return None

        portfolio_df = pd.DataFrame(self.portfolio_value)
        trades_df = pd.DataFrame(self.trades)
        
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_df['daily_return'])
        max_drawdown = self.calculate_max_drawdown(portfolio_df['value'])
        
        if trades_df.empty:
            return {
                'portfolio': portfolio_df,
                'trades': trades_df,
                'initial_cash': self.initial_cash,
                'final_value': portfolio_df['value'].iloc[-1],
                'total_return': (portfolio_df['value'].iloc[-1] / self.initial_cash - 1) * 100,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }

        if 'action' not in trades_df.columns:
            trades_df['action'] = 'unknown'
        if 'profit' not in trades_df.columns:
            trades_df['profit'] = 0

        results = {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'initial_cash': self.initial_cash,
            'final_value': portfolio_df['value'].iloc[-1],
            'total_return': (portfolio_df['value'].iloc[-1] / self.initial_cash - 1) * 100,
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[(trades_df['action'] == 'sell') & (trades_df['profit'] > 0)]),
            'losing_trades': len(trades_df[(trades_df['action'] == 'sell') & (trades_df['profit'] <= 0)])
        }

        if results['total_trades'] > 0:
            sell_trades = trades_df[trades_df['action'] == 'sell']
            if not sell_trades.empty:
                results['win_rate'] = (results['winning_trades'] / len(sell_trades)) * 100
                results['average_profit'] = sell_trades['profit'].mean()
                results['max_profit'] = sell_trades['profit'].max()
                results['max_loss'] = sell_trades['profit'].min()
                
                profit_sum = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
                loss_sum = sell_trades[sell_trades['profit'] < 0]['profit'].sum()
                
                if loss_sum != 0:
                    results['profit_factor'] = abs(profit_sum / loss_sum)
                else:
                    results['profit_factor'] = float('inf') if profit_sum > 0 else 0

        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        results['sharpe_ratio'] = self.calculate_sharpe_ratio(portfolio_df['daily_return'])
        results['max_drawdown'] = self.calculate_max_drawdown(portfolio_df['value'])

        return results

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        returns = returns.dropna()
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return sharpe_ratio

    def calculate_max_drawdown(self, values):
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min()
        
        return max_drawdown * 100


class PerformanceEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_strategy(self, backtest_results, benchmark_returns=None):
        portfolio = backtest_results['portfolio']
        
        self.metrics['total_return'] = backtest_results['total_return']
        self.metrics['final_value'] = backtest_results['final_value']
        self.metrics['sharpe_ratio'] = backtest_results['sharpe_ratio']
        self.metrics['max_drawdown'] = backtest_results['max_drawdown']
        self.metrics['win_rate'] = backtest_results.get('win_rate', 0)
        self.metrics['total_trades'] = backtest_results['total_trades']
        self.metrics['average_profit'] = backtest_results.get('average_profit', 0)
        self.metrics['max_profit'] = backtest_results.get('max_profit', 0)
        self.metrics['max_loss'] = backtest_results.get('max_loss', 0)
        self.metrics['profit_factor'] = backtest_results.get('profit_factor', 0)

        if benchmark_returns is not None:
            self.metrics['alpha'] = self.calculate_alpha(portfolio['value'], benchmark_returns)
            self.metrics['beta'] = self.calculate_beta(portfolio['value'], benchmark_returns)

        return self.metrics

    def calculate_alpha(self, portfolio_values, benchmark_returns, risk_free_rate=0.02):
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        if len(portfolio_returns) != len(benchmark_returns):
            benchmark_returns = benchmark_returns[:len(portfolio_returns)]
        
        excess_portfolio = portfolio_returns - risk_free_rate / 252
        excess_benchmark = benchmark_returns - risk_free_rate / 252
        
        beta = self.calculate_beta(portfolio_values, benchmark_returns)
        alpha = (excess_portfolio.mean() - beta * excess_benchmark.mean()) * 252
        
        return alpha

    def calculate_beta(self, portfolio_values, benchmark_returns):
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        if len(portfolio_returns) != len(benchmark_returns):
            benchmark_returns = benchmark_returns[:len(portfolio_returns)]
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance
        
        return beta

    def generate_report(self, save_path=None):
        report = f"""
        ====== 策略回测报告 ======
        
        总收益率: {self.metrics['total_return']:.2f}%
        最终资金: {self.metrics['final_value']:.2f}
        夏普比率: {self.metrics['sharpe_ratio']:.4f}
        最大回撤: {self.metrics['max_drawdown']:.2f}%
        
        胜率: {self.metrics['win_rate']:.2f}%
        总交易次数: {self.metrics['total_trades']}
        平均盈亏: {self.metrics['average_profit']:.2f}
        最大盈利: {self.metrics['max_profit']:.2f}
        最大亏损: {self.metrics['max_loss']:.2f}
        盈亏比: {self.metrics['profit_factor']:.2f}
        """

        if 'alpha' in self.metrics:
            report += f"""
        Alpha: {self.metrics['alpha']:.4f}
        Beta: {self.metrics['beta']:.4f}
            """

        print(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def plot_performance(self, backtest_results, save_path=None):
        portfolio = backtest_results['portfolio']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(portfolio['date'], portfolio['value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].plot(portfolio['date'], portfolio['cash'], label='Cash')
        axes[0, 1].plot(portfolio['date'], portfolio['positions_value'], label='Positions')
        axes[0, 1].set_title('Cash vs Positions Value')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].tick_params(axis='x', rotation=45)

        trades = backtest_results['trades']
        if not trades.empty:
            sell_trades = trades[trades['action'] == 'sell']
            if not sell_trades.empty:
                axes[1, 0].bar(sell_trades['date'], sell_trades['profit'], 
                               color=['green' if p > 0 else 'red' for p in sell_trades['profit']])
                axes[1, 0].set_title('Trade Profits/Losses')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Profit/Loss')
                axes[1, 0].grid(True)
                axes[1, 0].tick_params(axis='x', rotation=45)

        cumulative_returns = (portfolio['value'] / portfolio['value'].iloc[0] - 1) * 100
        axes[1, 1].plot(portfolio['date'], cumulative_returns)
        axes[1, 1].set_title('Cumulative Returns (%)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Returns (%)')
        axes[1, 1].grid(True)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class BenchmarkComparator:
    def __init__(self):
        pass

    def calculate_benchmark_returns(self, df, benchmark_code='000300'):
        benchmark_data = df[df['stock_code'] == benchmark_code]
        
        if benchmark_data.empty:
            logger.warning(f"Benchmark data for {benchmark_code} not found")
            return None

        benchmark_data = benchmark_data.sort_values('date')
        benchmark_data['returns'] = benchmark_data['close'].pct_change()
        
        return benchmark_data['returns'].dropna()

    def compare_with_benchmark(self, backtest_results, benchmark_returns):
        portfolio = backtest_results['portfolio']
        portfolio_returns = portfolio['value'].pct_change().dropna()

        comparison = {
            'strategy_return': backtest_results['total_return'],
            'strategy_volatility': portfolio_returns.std() * np.sqrt(252) * 100,
            'strategy_sharpe': backtest_results['sharpe_ratio'],
            'benchmark_return': (benchmark_returns.mean() * 252) * 100,
            'benchmark_volatility': benchmark_returns.std() * np.sqrt(252) * 100,
            'excess_return': backtest_results['total_return'] - (benchmark_returns.mean() * 252) * 100
        }

        comparison['information_ratio'] = (
            (portfolio_returns.mean() - benchmark_returns.mean()) / 
            (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        )

        return comparison

    def plot_comparison(self, backtest_results, benchmark_returns, save_path=None):
        portfolio = backtest_results['portfolio']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        portfolio_cumulative = (portfolio['value'] / portfolio['value'].iloc[0] - 1) * 100
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1

        axes[0].plot(portfolio['date'], portfolio_cumulative, label='Strategy', linewidth=2)
        axes[0].plot(portfolio['date'][:len(benchmark_cumulative)], 
                    benchmark_cumulative * 100, label='Benchmark', linewidth=2)
        axes[0].set_title('Strategy vs Benchmark Cumulative Returns')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Returns (%)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].tick_params(axis='x', rotation=45)

        portfolio_returns = portfolio['value'].pct_change().dropna()
        
        axes[1].scatter(portfolio_returns[:len(benchmark_returns)], benchmark_returns, alpha=0.5)
        axes[1].set_title('Strategy vs Benchmark Returns Scatter')
        axes[1].set_xlabel('Strategy Returns')
        axes[1].set_ylabel('Benchmark Returns')
        axes[1].grid(True)

        z = np.polyfit(portfolio_returns[:len(benchmark_returns)], benchmark_returns, 1)
        p = np.poly1d(z)
        axes[1].plot(portfolio_returns[:len(benchmark_returns)], p(portfolio_returns[:len(benchmark_returns)]), 
                    "r--", alpha=0.8, label=f'β={z[0]:.2f}')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
