import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class MACDBacktester:
    def __init__(self, csv_path, initial_capital=100000):
        self.data = pd.read_csv(csv_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.positions = []
        self.trades = []
        self.current_position = None
        self.initial_capital = initial_capital  # Starting capital
        self.capital = self.initial_capital
        self.portfolio_values = []

    def run_backtest(self):
        """Run the backtesting simulation using the MACD strategy."""
        print("Starting backtest...")
        last_signal_date = None  # To prevent multiple signals on same day

        for i in range(1, len(self.data)):
            current_row = self.data.iloc[i]
            previous_rows = self.data.iloc[max(0, i-6):i]  # Get up to 6 previous rows for pattern detection
            
            # Update portfolio value before any trades
            self._update_portfolio_value(current_row)
            
            # Skip if we've already had a signal today
            current_date = pd.to_datetime(current_row['timestamp']).date()
            if last_signal_date == current_date:
                continue

            # Only look for sell/exit signals if we have a position
            if self.current_position is not None:
                # Check for sell signal (5 consecutive lesser green)
                if self._check_sell_signal(previous_rows, current_row):
                    self._exit_position(current_row, 'SELL')
                    last_signal_date = current_date
                    continue

                # Check for exit signal (3 consecutive red candles)
                if self._check_exit_signal(previous_rows, current_row):
                    self._exit_position(current_row, 'EXIT')
                    last_signal_date = current_date
                    continue

            # Look for buy signal if we don't have a position
            elif self._check_buy_signal(previous_rows, current_row):
                self._enter_long(current_row)
                last_signal_date = current_date

    def _check_buy_signal(self, previous_rows, current_row):
        """Check for 6 consecutive lesser red MACD histogram bars"""
        if len(previous_rows) < 5:  # Need at least 5 previous bars plus current
            return False
            
        # Get the last 6 histogram values (5 previous + current)
        hist_values = list(previous_rows['MACDh_12_26_9'].values) + [current_row['MACDh_12_26_9']]
        if len(hist_values) < 6:
            return False
            
        # Check the last 6 values are all negative and each is less negative than the previous
        lesser_red = True
        for i in range(1, 6):
            if hist_values[i] >= 0 or abs(hist_values[i]) >= abs(hist_values[i-1]):
                lesser_red = False
                break
                
        return lesser_red

    def _check_sell_signal(self, previous_rows, current_row):
        """Check for 5 consecutive lesser green MACD histogram bars with 20% minimum decrease"""
        if len(previous_rows) < 4:  # Need at least 4 previous bars plus current
            return False
            
        # Get the last 5 histogram values (4 previous + current)
        hist_values = list(previous_rows['MACDh_12_26_9'].values) + [current_row['MACDh_12_26_9']]
        if len(hist_values) < 5:
            return False
            
        # Check conditions:
        # 1. All bars must be positive
        # 2. Each bar must be at least 20% smaller than previous
        # 3. Not in strong uptrend (using MACD line)
        if not all(val > 0 for val in hist_values[-5:]):
            return False
            
        MIN_DECREASE_PCT = 0.20
        lesser_green = True
        for i in range(1, 5):
            if hist_values[i] >= hist_values[i-1] or \
               (hist_values[i-1] - hist_values[i]) / hist_values[i-1] < MIN_DECREASE_PCT:
                lesser_green = False
                break
                
        # Check if not in strong uptrend
        if lesser_green:
            macd_values = list(previous_rows['MACD_12_26_9'].values) + [current_row['MACD_12_26_9']]
            macd_trend = np.mean(macd_values[-10:])  # Look at last 10 periods
            if macd_trend > current_row['MACD_12_26_9']:
                lesser_green = False
                
        return lesser_green

    def _check_exit_signal(self, previous_rows, current_row):
        """Check for 3 consecutive red candles with decreasing closing prices"""
        if len(previous_rows) < 2:  # Need at least 2 previous bars plus current
            return False
            
        # Get the last 3 candles (2 previous + current)
        candles = pd.concat([previous_rows.iloc[-2:], pd.DataFrame([current_row])])
        
        # Check if all 3 are red candles (close < open) with decreasing closes
        red_candles = all(candle['close'] < candle['open'] for _, candle in candles.iterrows())
        decreasing_closes = all(candles['close'].iloc[i] > candles['close'].iloc[i+1] 
                              for i in range(len(candles)-1))
        
        return red_candles and decreasing_closes

    def _enter_long(self, row):
        """Enter a long position"""
        if self.current_position is None:
            price = row['close']
            shares = int(self.capital * 0.2 / price)  # Using 95% of capital
            if shares > 0:
                self.current_position = {
                    'entry_price': price,
                    'shares': shares,
                    'entry_date': row['timestamp'],
                    'cost': price * shares
                }
                self.capital -= price * shares
                self.trades.append({
                    'type': 'BUY',
                    'date': row['timestamp'],
                    'price': price,
                    'shares': shares,
                    'value': price * shares
                })
                print(f"BUY: {shares} shares at ${price:.2f} on {row['timestamp']}")

    def _exit_position(self, row, exit_type='SELL'):
        """Exit the current position"""
        if self.current_position is not None:
            price = row['close']
            shares = self.current_position['shares']
            entry_price = self.current_position['entry_price']
            profit = (price - entry_price) * shares
            self.capital += price * shares
            
            self.trades.append({
                'type': exit_type,
                'date': row['timestamp'],
                'price': price,
                'shares': shares,
                'value': price * shares,
                'profit': profit
            })
            
            print(f"{exit_type}: {shares} shares at ${price:.2f} on {row['timestamp']}, Profit: ${profit:.2f}")
            self.current_position = None

    def _update_portfolio_value(self, row):
        """Update the current portfolio value"""
        value = self.capital
        if self.current_position is not None:
            value += self.current_position['shares'] * row['close']
        # Ensure we have a portfolio value for the first row
        if not self.portfolio_values:
            self.portfolio_values.append(self.initial_capital)
        self.portfolio_values.append(value)

    def get_performance_metrics(self):
        """Calculate and return performance metrics"""
        if not self.trades:
            return "No trades executed"

        profits = [t['profit'] for t in self.trades if 'profit' in t]
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p <= 0])
        total_trades = winning_trades + losing_trades
        metrics = {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': winning_trades / total_trades if total_trades else 0,
            'Total Profit/Loss': sum(profits),
            'Average Profit/Loss per Trade': np.mean(profits) if profits else 0,
            'Final Portfolio Value': self.portfolio_values[-1],
            'Return': (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100,
            'Max Drawdown %': self._calculate_max_drawdown()
        }
        return metrics

    def _calculate_max_drawdown(self):
        """Calculate the maximum drawdown percentage"""
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        return max_drawdown

    def plot_results(self):
        """Plot the backtest results"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        
        # Plot stock price and portfolio value
        ax1.plot(self.data['timestamp'], self.data['close'], 'b-', label='Stock Price', alpha=0.6)
        ax1.set_ylabel('Stock Price ($)', color='b')
        
        # Create second y-axis for portfolio value
        ax1_twin = ax1.twinx()
        # Ensure portfolio_values matches timestamp length
        portfolio_values = self.portfolio_values[:len(self.data)]
        ax1_twin.plot(self.data['timestamp'], portfolio_values, 'g-', label='Portfolio Value', alpha=0.8)
        ax1_twin.set_ylabel('Portfolio Value ($)', color='g')
        
        # Plot buy and sell points
        for trade in self.trades:
            if trade['type'] == 'BUY':
                ax1.plot(trade['date'], trade['price'], '^', color='g', markersize=10, label='Buy')
            elif trade['type'] == 'SELL':
                ax1.plot(trade['date'], trade['price'], 'v', color='r', markersize=10, label='Sell')
            else:  # EXIT
                ax1.plot(trade['date'], trade['price'], 's', color='k', markersize=10, label='Exit')
        
        # Plot MACD
        ax2.plot(self.data['timestamp'], self.data['MACD_12_26_9'], 'b-', label='MACD', alpha=0.6)
        ax2.plot(self.data['timestamp'], self.data['MACDs_12_26_9'], 'r-', label='Signal', alpha=0.6)
        
        # Plot histogram with color coding
        for i in range(len(self.data)-1):
            if self.data['MACDh_12_26_9'].iloc[i] >= 0:
                color = 'g' if self.data['MACDh_12_26_9'].iloc[i] > self.data['MACDh_12_26_9'].iloc[i-1] else 'lightgreen'
            else:
                color = 'r' if abs(self.data['MACDh_12_26_9'].iloc[i]) > abs(self.data['MACDh_12_26_9'].iloc[i-1]) else 'pink'
            ax2.bar(self.data['timestamp'].iloc[i], self.data['MACDh_12_26_9'].iloc[i], color=color)
        
        # Set titles and legends
        ax1.set_title('Backtest Results')
        ax2.set_title('MACD Indicator')
        
        # Handle legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def run_backtest(csv_path, initial_capital=100000):
    """Run a backtest and display results"""
    print(f"Running backtest with {initial_capital:,} initial capital...")
    backtester = MACDBacktester(csv_path, initial_capital)
    backtester.run_backtest()
    
    metrics = backtester.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")
            
    backtester.plot_results()
    return backtester

if __name__ == "__main__":
    csv_path = "/../data/VRT_macd_data.csv"
    backtester = run_backtest(csv_path)