import pandas as pd

# Load the dataset
us_file = 'COVID Data/us_covid_data.csv'
us_data = pd.read_csv(us_file)

# Convert the date column to datetime format
us_data['date'] = pd.to_datetime(us_data['date'])

# Define the date range
start_date = '2021-12-01'
end_date = '2022-03-31'

# Filter the data for the specified date range
filtered_us_data = us_data[(us_data['date'] >= start_date) & (us_data['date'] <= end_date)]

# Save the filtered data to a new CSV file
filtered_us_data.to_csv('COVID Data/filtered_us_covid_data.csv', index=False)

print("Filtered data related to the United States has been saved to 'filtered_us_covid_data.csv'.")
