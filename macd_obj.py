import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fin_table_obj import Table

class MACDTable(Table):
    def __init__(self, df):
        super().__init__(df)
        self.ma1 = 12
        self.ma2 = 26
        self.macd_line = 9

    def gen_table(self, optional_bool=True):
        super().gen_table()
    
        # initalising EMA1, EMA2, EMA1 - EMA2, and Macd line
        self.df[f'{self.ma1}-day EMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
        self.df[f'{self.ma2}-day EMA'] = self.df['Close'].rolling(int(self.ma2)).mean().shift()
        for i in range(int(self.ma1) + 1, len(self.df)):
            self.df.iloc[i, self.df.columns.get_loc(f'{self.ma1}-day EMA')] = (
                (self.df.iloc[i - 1, self.df.columns.get_loc('Close')] * (2 / (int(self.ma1) + 1))) +
                (self.df.iloc[i - 1, self.df.columns.get_loc(f'{self.ma1}-day EMA')] * (1 - (2 / (int(self.ma1) + 1))))
            )
  
        for i in range(int(self.ma2) + 1, len(self.df)):
            self.df.iloc[i, self.df.columns.get_loc(f'{self.ma2}-day EMA')] = (
                (self.df.iloc[i - 1, self.df.columns.get_loc('Close')] * (2 / (int(self.ma2) + 1))) +
                (self.df.iloc[i - 1, self.df.columns.get_loc(f'{self.ma2}-day EMA')] * (1 - (2 / (int(self.ma2) + 1))))
            )

        #calculating the macd line

        self.df['MACD Line'] = self.df[f'{self.ma1}-day EMA'] - self.df[f'{self.ma2}-day EMA']
        if self.df['MACD Line'].isnull().sum() > 0:
            self.df['MACD Line'] = self.df['MACD Line'].fillna(0)

        #calculating the 9 day EMA of the macd line

        self.df['MACD Signal Line'] = self.df['MACD Line'].rolling(int(self.macd_line)).mean().shift()
        for i in range(int(self.macd_line) + 1, len(self.df)):
            self.df.iloc[i, self.df.columns.get_loc('MACD Signal Line')] = (
               (self.df.iloc[i - 1, self.df.columns.get_loc('MACD Line')] * (2 / (int(self.macd_line) + 1))) +
                (self.df.iloc[i - 1, self.df.columns.get_loc('MACD Signal Line')] * (1 - (2 / (int(self.macd_line) + 1))))
            )
        
        self.df['MACD Histogram'] = self.df['MACD Line'] - self.df['MACD Signal Line']


        # Signal to long
        self.df['Signal'] = np.where(self.df['MACD Line'] > self.df['MACD Signal Line'], 1, 0)

        # Signal to short
        self.df['Signal'] = np.where(self.df['MACD Line'] < self.df['MACD Signal Line'], -1, self.df['Signal'])

        # Model return
        self.df['MACD Model Return'] = self.df['Return'] * self.df['Signal']

        # Entry column for visualization
        self.df['Entry'] = self.df.Signal.diff()

        # drop rows
        self.df.dropna(inplace=True)

        # Cumulative Returns
        self.df['Cumulative MACD Model Return'] = (np.exp(self.df['MACD Model Return'] / 100).cumprod() - 1) * 100

        # Recalculate return and cumulative return to include model returns
        self.df['Return'] = (np.log(self.df['Close']).diff()) * 100
        self.df['Cumulative Return'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100

        # Formatting the table
        self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', f'{self.ma1}-day EMA', f'{self.ma2}-day EMA', 'MACD Signal Line', 'MACD Line', 'MACD Histogram', 'Return', 'Cumulative Return', 'MACD Model Return', 'Cumulative MACD Model Return', 'Signal', 'Entry']]), 3)
        
        #format date as YYYY-MM-DD
        self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
        
        if optional_bool:
            #options to show all rows and columns
            #pd.set_option('display.max_rows', None)
            #pd.set_option('display.max_columns', None)
            #pd.set_option('display.width', None)
            #pd.set_option('display.max_colwidth', None)
            return self.df
        pass

    def gen_macd_visual(self, model_days,):
# Closing price figure
        self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
        fig1 = plt.figure(figsize=(12, 6))

# Use the actual index for x-values
        x_values = range(len(self.df.iloc[-model_days:]))

# Plot the closing prices
        plt.plot(x_values, self.df.iloc[-model_days:]['Close'], label='Close')

#Plot 12 and 26 day EMAs
        plt.plot(x_values, self.df.iloc[-model_days:][f'{self.ma1}-day EMA'], label=f'{self.ma1}-day EMA') 
        plt.plot(x_values, self.df.iloc[-model_days:][f'{self.ma2}-day EMA'], label=f'{self.ma2}-day EMA')

# Plot buy signals (Entry == 2)
        plt.scatter(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == 2],
            self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Entry'] == 2],
            marker='^', color='g', s=100, label='Buy Signal'
        )

# Plot sell signals (Entry == -2)
        plt.scatter(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == -2],
            self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Entry'] == -2],
            marker='v', color='r', s=100, label='Sell Signal'
        )

# Set x-axis to date values and make it so they dont spawn too many labels
        plt.xticks(ticks=x_values, labels=self.df.iloc[-model_days:].index, rotation=45)
        plt.locator_params(axis='x', nbins=10)

# grid and legend
        plt.grid(True, alpha=0.5)
        plt.legend(loc=2)
    
#MACD Figure 2
        fig2 = plt.figure(figsize=(12, 6))
        plt.grid(True, alpha=0.5)

#generate x values
        x_values = range(len(self.df.iloc[-model_days:]))

# Plot MACD Signal Line and MACD Line
        plt.plot(x_values, self.df.iloc[-model_days:]['MACD Signal Line'], label='MACD Signal Line')
        plt.plot(x_values, self.df.iloc[-model_days:]['MACD Line'], label='MACD Line')

# Plot MACD Histogram bars
        plt.bar(
            x=x_values,  # Use numeric x-values for even spacing
            height=self.df.iloc[-model_days:]['MACD Histogram'],
            width=1,  # Adjust width for better spacing
            label='MACD Histogram',
            color=['g' if x > 0 else 'r' for x in self.df.iloc[-model_days:]['MACD Histogram']]
        )

# Add a horizontal line at 0
        plt.axhline(0, color='k', lw=1, ls='--')

# Set x-axis labels to the original index values
        plt.xticks(ticks=x_values, labels=self.df.iloc[-model_days:].index, rotation=45)
        plt.locator_params(axis='x', nbins=10)

#plotting entry points, .loc for labels
        plt.scatter(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == 2],
            self.df.iloc[-model_days:]['MACD Signal Line'][self.df.iloc[-model_days:]['Entry'] == 2],
            marker='^', color='g', s=100, label='Buy Signal'
        )

# Plot sell signals (Entry == -2)
        plt.scatter(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == -2],
            self.df.iloc[-model_days:]['MACD Line'][self.df.iloc[-model_days:]['Entry'] == -2],
            marker='v', color='r', s=100, label='Sell Signal'
        )

#print statements        
        print(f'from {self.df.index[-model_days]} to {self.df.index[-1]}')
        print(f'count of buy signals: {len(self.df[self.df["Entry"] == 2])}')
        print(f'count of sell signals: {len(self.df[self.df["Entry"] == -2])}')

    def gen_buyhold_comp(self, ticker):
        labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
        fig1= plt.figure(figsize=(12, 6))
        x_values = range(len(self.df))

# add buy/hold to legend if it doesn't exist
        if f'{ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
            plt.plot(x_values, self.df['Cumulative Return'], label=f'{ticker} Buy/Hold')
# model plot
        plt.plot(x_values, self.df['Cumulative MACD Model Return'], label=f'{ticker} MACD Model')

# Set x-axis to date values and make it so they dont spawn too many labels
        plt.xticks(ticks=x_values, labels=labels, rotation=45)
        plt.locator_params(axis='x', nbins=10)

# grid and legend
        plt.legend(loc=2)
        plt.grid(True, alpha=.5)
# print cumulative return if not already printed
        print(f"{ticker} Cumulative MACD Model Return:", round(self.df['Cumulative MACD Model Return'].iloc[-1], 2))
        print(f" from {self.df.index[0]} to {self.df.index[-1]}")