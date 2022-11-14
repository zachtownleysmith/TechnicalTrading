import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_rows = 1000


def remove_dup_trades(signals):
    output = signals
    temp = True
    while temp:
        running_sum = output.cumsum()
        temp = running_sum.index[(running_sum < -1) | (running_sum > 1)].tolist()

        if temp:
            output[temp[0]] = 0

    return output


class Equity:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices = yf.download(ticker, start, end)['Adj Close']


class TradingSMA:
    def __init__(self, equity, fast_window, slow_window, delta):
        self.equity = equity
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.delta = delta

        self.fast_ma = equity.prices.rolling(fast_window).mean()
        self.slow_ma = equity.prices.rolling(slow_window).mean()

    def buy_sell(self):
        indicator = (self.fast_ma - self.slow_ma) / self.fast_ma

        # Code for buy_sell indicator
        buy_sell = indicator

        # Buy if indicator > delta, Sell if indicator < delta
        # 1 to represent Buys, -1 to represent Sells
        buy_sell = buy_sell.mask(buy_sell > self.delta, 1)
        buy_sell = buy_sell.mask(buy_sell < -self.delta, -1)
        buy_sell = buy_sell.mask((buy_sell <= self.delta) & (buy_sell >= -self.delta), 0)

        # If Buy/Sell signal is the same as prior day, the no signal today
        buy_sell = buy_sell.mask((buy_sell == buy_sell.shift(1)), 0)

        # Remove successive buy and sell signals.
        buy_sell = remove_dup_trades(buy_sell)

        # Shift back one time period for when signal can be acted upon
        buy_sell = buy_sell.shift(1)

        return buy_sell

    def pnl(self, show=True):

        trades = self.buy_sell()
        output = trades.multiply(self.equity.prices.multiply(-1))
        output = output.mask(output.isna(), 0)
        output = output.cumsum()

        if show:
            plt.style.use('fivethirtyeight')
            plt.plot_date(output.index, output, linestyle='solid', marker="")
            plt.title('Profit and Loss')
            plt.ylabel('Dollars')
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            plt.show()

        return output[-1]

    def trade_plot(self):

        buy_trades = self.buy_sell()
        sell_trades = buy_trades

        buy_trades = buy_trades.mask(buy_trades != 1, None)
        buy_trades = buy_trades.multiply(self.equity.prices)

        sell_trades = sell_trades.mask(sell_trades != -1, None)
        sell_trades = sell_trades.multiply(self.equity.prices.multiply(-1))

        plt.style.use('fivethirtyeight')

        plt.plot_date(self.equity.prices.index, self.equity.prices,
                      linestyle='solid', marker='', label='Asset Price', linewidth=1.5, color='k')

        plt.plot_date(self.fast_ma.index, self.fast_ma,
                      linestyle='solid', marker='', label='Fast Moving Avg', linewidth=1.5, color='b')

        plt.plot_date(self.slow_ma.index, self.slow_ma,
                      linestyle='solid', marker='', label='Slow Moving Avg', linewidth=1.5, color='m')

        plt.plot_date(buy_trades.index, buy_trades, label='Buy Trades', color='g')
        plt.plot_date(sell_trades.index, sell_trades, label='Sell Trades', color='r')

        plt.legend()
        plt.show()


class TradingEMA(TradingSMA):
    def __init__(self, equity, fast_window, slow_window, delta):
        self.equity = equity
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.delta = delta

        self.fast_ma = self.equity.prices.ewm(span=self.fast_window, adjust=False).mean()
        self.slow_ma = self.equity.prices.ewm(span=self.slow_window, adjust=False).mean()


if __name__ == '__main__':
    tesla = Equity('TSLA', '2021-1-1', '2022-1-1')
    first_strat = TradingSMA(tesla, 5, 20, 0.03)
    print(first_strat.pnl(show=True))
    first_strat.trade_plot()


    print('')
    print('DONE')


# To Add:
# Trading Costs
# Close out position @ end
# Channel Breakout, MACD, RSI

