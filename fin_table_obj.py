import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input a yfinance dataframe
# generates a simple returns table with daily returns, cumulative returns, and volatility
class Table:
    def __init__(self, df):
        self.df = df
    def gen_table(self, optional_bool=False):
    #adding day count
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count

    #dropping volume column
        if 'Volume' in self.df.columns:
            self.df.drop(columns=['Volume'], inplace = True)

    #create returns and the model returns colummns
    # the difference between the current row ln and previous row ln gives a daily return
        self.df['Return'] = (np.log(self.df['Close']).diff()) * 100

    # Cumulative Return
        self.df['Cumulative Return'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100
    # Volatilty
        self.df['Volatility'] = self.df['Return'].rolling(21).std().shift()

    # formatting the table
        self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', 'Return', 'Cumulative Return', 'Volatility']]), 3)

        return self.df
    
    def print_cumulative_return(self, ticker):
        print(f"{ticker} Cumulative Buy/Hold Return:", round(self.df['Cumulative Return'].iloc[-1], 2))