import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# List to store DataFrames
dfs = []
stock_files = glob.glob("Stock Data/*_historical_data.csv")  # Match files with the specific pattern

# Read each CSV file and append DataFrame to the list
for stock_file in stock_files:
    df = pd.read_csv(stock_file)
    # Extract the ticker symbol from the file name
    ticker_symbol = os.path.splitext(os.path.basename(stock_file))[0].split('_')[0]
    df['Stock'] = ticker_symbol  # Add the ticker symbol as a new column
    dfs.append(df)

# Concatenate all DataFrames
all_data = pd.concat(dfs, ignore_index=True)

# Assuming the CSV has columns 'Date' and 'Close'
all_data.rename(columns={'Date': 'Date', 'Close': 'Close'}, inplace=True)

# Pivot the DataFrame to have Dates as index and stocks as columns
pivot_df = all_data.pivot(index='Date', columns='Stock', values='Close')

# Convert index to datetime
pivot_df.index = pd.to_datetime(pivot_df.index)

# Load the actual COVID-19 data with 'new_cases_smoothed'
covid_data = pd.read_csv("COVID Data/filtered_us_covid_data.csv")  # Replace with your actual CSV file path
covid_data['date'] = pd.to_datetime(covid_data['date'])
covid_data.set_index('date', inplace=True)

# Ensure the column name matches your CSV
covid_data = covid_data[['new_cases_smoothed']]

# Create a plot for all stocks together with twin axis for COVID-19 cases
fig, ax1 = plt.subplots(figsize=(14, 7))

# Loop over each stock and plot it
for stock in pivot_df.columns:
    ax1.plot(pivot_df.index, pivot_df[stock], label=stock, linewidth=2)

# Set up the primary Y-axis (for stock prices)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Closing Price', fontsize=12)
ax1.set_title('Stock Closing Prices and COVID-19 Cases Over Time', fontsize=16)

# Add a grid for clarity
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set up the twin Y-axis (for COVID-19 cases)
ax2 = ax1.twinx()
ax2.plot(covid_data.index, covid_data['new_cases_smoothed'], color='red', linewidth=2, label='Cases')

# Label the secondary Y-axis
ax2.set_ylabel('New COVID 19 Cases (Smoothed)', fontsize=12, labelpad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust the plot to ensure space on the left for both legends
fig.subplots_adjust(left=0.14)  # Shift the plot to the right to make room for the legends

# Common parameters for both legends
legend_params = {
    'handletextpad': 1.25,  # Space between marker and text
    'borderpad': 1,      # Space between the text and legend box edge
    'labelspacing': 0.4   # Space between labels
}

# Adjust the stock legend with common parameters
stock_legend = ax1.legend(
    title='Stocks', 
    loc='center left', 
    bbox_to_anchor=(-0.25, 0.5), 
    fontsize=10, 
    **legend_params  # Apply common legend parameters
)

# Adjust the COVID-19 legend with the same parameters
covid_legend = ax2.legend(
    title='COVID-19 Cases', 
    loc='upper left', 
    bbox_to_anchor=(-0.25, 0.77), 
    fontsize=10, 
    **legend_params  # Apply common legend parameters
)

# Adjust the layout for a clean appearance
plt.tight_layout()

# Show the plot
plt.show()
