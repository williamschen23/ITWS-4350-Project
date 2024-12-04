from owid import catalog
import pandas as pd
####################################################
# WARNING THE OWID IS DATASET IS MASSIVE!! (100MB) #
# DOWNLOAD AT YOUR OWN RISK!!                      # 
####################################################

# Look for Covid-19 data, return a data frame of matches
catalog.find('covid')

# Load Covid-19 data from the OWID in df
covid_df = catalog.find('covid', namespace='owid').load()

# Download data to .csv format
covid_df.to_csv(f'COVID19_data_entire_catalog.csv')

