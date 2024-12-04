import yfinance as yf
import pandas as pd

# Define the ticker symbol
ticker = 'VGT'

# Fetch the data
vgt_data = yf.Ticker(ticker)

# Get historical market data from December 1, 2021 to March 31, 2022
historical_data = vgt_data.history(start='2021-12-01', end='2022-03-31')

# Write the data to a CSV file
historical_data.to_csv(f'{ticker}_historical_data.csv')

print("Data saved to VGT_historical_data.csv")
