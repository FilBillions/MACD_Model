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
        if optional_bool:
            return self.df
        pass

    def gen_macd_visual(self, model_days,):
    #Closing price figure
        fig1 = plt.figure(figsize=(12, 6))
        plt.grid(True, alpha=0.5)

        plt.plot(self.df.iloc[-model_days:]['Close'])

    #plot entries and exits
        plt.plot(self.df[-model_days:].loc[self.df.Entry == 2].index, self.df[-model_days:]['Close'][self.df.Entry == 2], '^', color = 'g', markersize = 10)
        plt.plot(self.df[-model_days:].loc[self.df.Entry == -2].index, self.df[-model_days:]['Close'][self.df.Entry == -2], 'v', color = 'r', markersize = 10)

    #plot legend
        plt.legend(['MACD Signal Line', 'MACD Line', 'MACD Historgram', 'Buy Signal', 'Sell Signal'], loc = 2)

    #MACD Figure
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

    # Remove x-axis labels
        plt.gca().xaxis.set_visible(False)
    # Set x-axis labels to the original index values
        plt.xticks(ticks=x_values, labels=self.df.iloc[-model_days:].index, rotation=45)

    #plotting entry points, .loc for labels
        plt.plot(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == 2],
            self.df.iloc[-model_days:]['MACD Signal Line'][self.df.iloc[-model_days:]['Entry'] == 2],
            '^', color='g', markersize=10, label='Buy Signal'
        )

        # Plot sell signals (Entry == -2)
        plt.plot(
            [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == -2],
            self.df.iloc[-model_days:]['MACD Line'][self.df.iloc[-model_days:]['Entry'] == -2],
            'v', color='r', markersize=10, label='Sell Signal'
        )
    
    def gen_buyhold_comp(self, ticker):
    #add buy/hold to legend if it doesn't exist
        if f'{ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
            plt.plot(self.df['Cumulative Return'], label=f'{ticker} Buy/Hold')
    #model plot
        plt.plot(self.df['Cumulative MACD Model Return'], label=f'{ticker} MACD Model')
        plt.legend(loc=2)
        plt.grid(True, alpha=.5)
    #print cumulative return if not already printed
        print(f"{ticker} Cumulative MACD Model Return:", round(self.df['Cumulative MACD Model Return'].iloc[-1], 2))