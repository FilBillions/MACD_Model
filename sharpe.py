import numpy as np

# takes in a pandas array of returns and calculates the annualized sharpe ratio

def gen_sharpe(returns):
    avg_r = float((np.mean(returns)))
    std = float((np.std(returns)))
    sharpe = (avg_r / std) * 252 ** 0.5
    return sharpe