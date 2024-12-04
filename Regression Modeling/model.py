import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def predict_linear(ticker, covid_data, stock_data, target_column='Close', threshold=0.01):
    # Select only date and new_cases_smoothed from COVID data
    covid_data = covid_data[['date', 'new_cases_smoothed']]

    # Ensure consistent date formatting
    covid_data = covid_data.copy()
    covid_data['date'] = pd.to_datetime(covid_data['date']).dt.date
    
    # Remove timestamps from stock data Date column
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.date

    # Inner join on date columns
    merged_data = pd.merge(covid_data, stock_data, left_on='date', right_on='Date', how='inner')

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

    # Calculate MSE and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate accuracy-like metric based on a threshold
    error = np.abs(y_test - y_pred)  # Absolute error
    correct_predictions = np.sum(error <= threshold * y_test)  # Predictions within threshold (e.g., 1%)
    accuracy = correct_predictions / len(y_test)  # Proportion of correct predictions

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
    print(f"Accuracy (within {threshold*100}% error): {accuracy * 100:.2f}%")
    print(f"\nRegression Equation:")
    print(f"Stock Price = {model.intercept_:.2f} + {model.coef_[0]:.4f} * New COVID Cases")


def predict_logistic(ticker, covid_data, stock_data, target_column='Close'):
    # Select only date and new_cases_smoothed from COVID data
    covid_data = covid_data[['date', 'new_cases_smoothed']]

    # Ensure consistent date formatting
    covid_data = covid_data.copy()
    covid_data['date'] = pd.to_datetime(covid_data['date']).dt.date

    # Remove timestamps from stock data Date column
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.date

    # Inner join on date columns
    merged_data = pd.merge(covid_data, stock_data, left_on='date', right_on='Date', how='inner')

    # Prepare the data
    X = merged_data['new_cases_smoothed'].values.reshape(-1, 1)
    
    # Create a binary target: 1 if stock price went up, 0 if stock price went down
    y = np.where(merged_data[target_column].diff().shift(-1) > 0, 1, 0)  # 1 for up, 0 for down

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Down', 'Up'])
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.show()

    # Print model performance metrics
    print("Model Performance:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Visualize the results with a smooth curve
    plt.figure(figsize=(10, 6))

    # Sort the test data for a smooth curve
    sorted_indices = np.argsort(X_test.flatten())
    X_sorted = X_test[sorted_indices]
    
    # Predict probabilities for each test data point (logistic curve)
    y_prob = model.predict_proba(X_sorted)[:, 1]  # Probability for class 1 (stock price going up)

    plt.plot(X_sorted, y_prob, color='red', label='Predicted Probability of Stock Price Going Up')
    plt.scatter(X_test, y_test, color='blue', label='Actual Stock Price Direction', alpha=0.5)
    
    plt.title(f'COVID New Cases vs Stock Price Direction - Logistic Regression')
    plt.xlabel('Smoothed New COVID Cases')
    plt.ylabel('Stock Price Direction (Up = 1, Down = 0)')
    plt.legend()
    plt.show()

    # Print model coefficients
    print(f"\nLogistic Regression Coefficients:")
    print(f"Intercept: {model.intercept_[0]:.2f}, Coefficient: {model.coef_[0][0]:.4f}")

# Function to overlay COVID cases and Stock prices on the same graph
def plot_covid_and_stock(covid_data, stock_data):
    # Extract the relevant columns for COVID data
    covid_data['date'] = pd.to_datetime(covid_data['date'])  
    covid_data = covid_data[['date', 'new_cases_smoothed']].dropna()

    # Extract the relevant columns for Stock data
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True)  
    stock_data = stock_data[['Date', 'Close']].dropna()

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot COVID cases (left y-axis)
    ax1.plot(covid_data['date'], covid_data['new_cases_smoothed'], color='blue', label='New COVID Cases (Smoothed)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Smoothed New Cases', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('COVID Cases and Stock Price Over Time')
    
    # Create a second y-axis for Stock prices
    ax2 = ax1.twinx()
    ax2.plot(stock_data['Date'], stock_data['Close'], color='green', label='Stock Price')
    ax2.set_ylabel('Stock Price', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add gridlines and layout adjustments
    ax1.grid(True)
    fig.tight_layout()

    # Show the plot
    plt.show()

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
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True)

    # Run the analysis
    predict_linear(ticker, covid_data, stock_data)
    predict_logistic(ticker, covid_data, stock_data)

    # Comparison
    plot_covid_and_stock(covid_data, stock_data)

if __name__ == "__main__":
    main()