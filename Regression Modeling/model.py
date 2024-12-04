import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def predict_stock_with_covid_cases(ticker, covid_data, stock_data, target_column='Close'):
    # Select only date and new_cases_smoothed from COVID data
    covid_data = covid_data[['Date', 'new_cases_smoothed']]

    # Ensure consistent date formatting
    covid_data['Date'] = pd.to_datetime(covid_data['Date']).dt.date
    
    # Remove timestamps from stock data Date column
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.date

    # Inner join on date columns
    merged_data = pd.merge(covid_data, stock_data, left_on='Date', right_on='Date', how='inner')

    # Write merged data to CSV
    merged_data.to_csv(f'{ticker}_merged_data.csv', index=False)
    print(f"Merged data written to {ticker}_merged_data.csv")

    
    # Prepare the data
    X = merged_data['new_cases_smoothed'].values.reshape(-1, 1)
    y = merged_data[target_column].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Stock Price')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.title(f'COVID New Cases vs Stock Price - Linear Regression')
    plt.xlabel('Smoothed New COVID Cases')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Print model performance metrics
    print("Model Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print(f"\nRegression Equation:")
    print(f"Stock Price = {model.intercept_:.2f} + {model.coef_[0]:.4f} * New COVID Cases")

    return model, mse, r2

# Main script
def main():
    # Load COVID data
    covid_data = pd.read_csv('COVID Data\\filtered_us_covid_data.csv')
    
    # Prompt for stock ticker
    ticker = input("Enter the stock ticker (e.g., AAPL): ").upper()
    
    # Load stock historical data
    try:
        stock_data = pd.read_csv(f'Stock Data\\{ticker}_historical_data.csv')
    except FileNotFoundError:
        print(f"Error: Could not find {ticker}_historical_data.csv")
        return

    # Ensure date columns are in the same format
    covid_data['Date'] = pd.to_datetime(covid_data['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Run the analysis
    model, mse, r2 = predict_stock_with_covid_cases(ticker, covid_data, stock_data)

if __name__ == "__main__":
    main()