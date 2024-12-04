import pandas as pd

# Load the dataset
data_file = 'COVID Data/COVID19_data_entire_catalog.csv'
covid_data = pd.read_csv(data_file)

# Display the first few rows of the dataset to understand its structure
print(covid_data.head())

# Filter the data for the United States
us_data = covid_data[covid_data['location'] == 'United States']

# Save the filtered data to a new CSV file
us_data.to_csv('COVID Data/us_covid_data.csv', index=False)

print("Data related to the United States has been saved to 'us_covid_data.csv'.")
