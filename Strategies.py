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


def count_instances(series, target):
    temp = series[series == target]
    return len(temp)

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

        # Close out our position at the end of the period
        final_position = trades.cumsum()[-1] * self.equity.prices[-1]
        output[-1] = output[-1] + final_position

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


class TradingMACD:
    def __init__(self, equity, fast_window=12, slow_window=26, macd_window=9):
        self.equity = equity
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.macd_window = macd_window

        self.macd_line = self.equity.prices.ewm(span=self.fast_window, adjust=False).mean() - \
                         self.equity.prices.ewm(span=self.slow_window, adjust=False).mean()

        self.sig_line = self.macd_line.ewm(span=self.macd_window, adjust=False).mean()

    def buy_sell(self):

        # Buy if MACD line crosses Signal Line from below
        # Sell if MACD line crosses SIgnal Line from above
        buy_sell = np.sign(self.macd_line - self.sig_line)
        buy_sell = buy_sell.mask(buy_sell == buy_sell.shift(1), 0)
        buy_sell[1] = 0

        # Shift back one time period for when signal can be acted upon
        buy_sell = buy_sell.shift(1)

        return buy_sell

    def pnl(self, show=True):

        trades = self.buy_sell()
        output = trades.multiply(self.equity.prices.multiply(-1))
        output = output.mask(output.isna(), 0)
        output = output.cumsum()

        # Close out our position at the end of the period
        final_position = trades.cumsum()[-1] * self.equity.prices[-1]
        output[-1] = output[-1] + final_position

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

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax1.plot_date(self.equity.prices.index, self.equity.prices,
                      linestyle='solid', marker='', label='Asset Price', linewidth=1.5, color='k')

        ax1.plot_date(buy_trades.index, buy_trades, label='Buy Trades', color='g')
        ax1.plot_date(sell_trades.index, sell_trades, label='Sell Trades', color='r')

        ax2.plot_date(self.macd_line.index, self.macd_line,
                      linestyle='solid', marker='', label='MACD', linewidth=1.5, color='b')

        ax2.plot_date(self.sig_line.index, self.sig_line,
                      linestyle='solid', marker='', label='EMA(MACD)', linewidth=1.5, color='m')

        ax1.legend()
        ax2.legend()
        ax1.set_title('MACD Strategy Trades')
        fig.set_figheight(6)
        fig.set_figwidth(10)
        plt.show()


class TradingRSI:
    def __init__(self, equity, smoothing=14, up_thresh=70, lr_thresh=30):
        self.equity = equity
        self.smoothing = smoothing
        self.up_thresh = up_thresh
        self.lr_thresh = lr_thresh

    def ind_gen(self):
        diffs = np.sign(self.equity.prices - self.equity.prices.shift(1))
        n_up = diffs.mask(diffs == -1, 0)
        n_up = n_up.rolling(self.smoothing).sum()
        n_up = n_up.ewm(span=self.smoothing, adjust=False).mean()

        n_down = diffs.mask(diffs == 1, 0)
        n_down = n_down.mask(n_down == -1, 1)
        n_down = n_down.rolling(self.smoothing).sum()
        n_down = n_down.ewm(span=self.smoothing, adjust=False).mean()

        r_s = n_up/n_down
        indicator = 100 * r_s / (1 + r_s)

        return indicator

    def buy_sell(self):
        buy_sell = self.ind_gen()
        buy_sell = buy_sell.mask(buy_sell < self.lr_thresh, 1)
        buy_sell = buy_sell.mask(buy_sell > self.up_thresh, -1)
        buy_sell = buy_sell.mask((buy_sell <= self.up_thresh) & (buy_sell >= self.lr_thresh), 0)
        buy_sell = remove_dup_trades(buy_sell)
        buy_sell = buy_sell.shift(1)

        return buy_sell

    def pnl(self, show=True):

        trades = self.buy_sell()
        output = trades.multiply(self.equity.prices.multiply(-1))
        output = output.mask(output.isna(), 0)
        output = output.cumsum()

        # Close out our position at the end of the period
        final_position = trades.cumsum()[-1] * self.equity.prices[-1]
        output[-1] = output[-1] + final_position

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

        indicator = self.ind_gen()
        upper = indicator * 0 + self.up_thresh
        lower = indicator * 0 + self.lr_thresh

        plt.style.use('fivethirtyeight')

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax1.plot_date(self.equity.prices.index, self.equity.prices,
                      linestyle='solid', marker='', label='Asset Price', linewidth=1.5, color='k')

        ax1.plot_date(buy_trades.index, buy_trades, label='Buy Trades', color='g')
        ax1.plot_date(sell_trades.index, sell_trades, label='Sell Trades', color='r')

        ax2.plot_date(indicator.index, indicator,
                      linestyle='solid', marker='', label='RSI', linewidth=1.5, color='m')

        ax2.plot_date(upper.index, upper,
                      linestyle='solid', marker='', label='OverBought', linewidth=1.5, color='k')

        ax2.plot_date(lower.index, lower,
                      linestyle='solid', marker='', label='OverSold', linewidth=1.5, color='k')

        ax1.legend()
        ax2.legend()
        ax1.set_title('RSI Strategy Trades')
        fig.set_figheight(6)
        fig.set_figwidth(10)
        plt.show()


if __name__ == '__main__':
    tesla = Equity('VZ', '2021-1-1', '2022-1-1')
    #first_strat = TradingSMA(tesla, 5, 20, 0.03)
    #print(first_strat.pnl(show=True))
    #first_strat.trade_plot()


    #test = TradingMACD(tesla)
    #print(test.pnl())
    #test.trade_plot()

    #test = TradingRSI(tesla)
    #test.trade_plot()
    #print(test.pnl(show=False))

# To Add:
# Channel Breakout, MACD, RSI
# Trading Costs
# Clean up PNL plots
