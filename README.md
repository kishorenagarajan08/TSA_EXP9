# EX.NO.09 A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 23/10/2025
### NAME: KISHORE N
### REG.NO: 212223230106
### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
Import necessary library:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```
Load and clean the data:
```data = pd.read_csv("GoogleStockPrices.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
```
Define ARIMA model function:
```
def arima_model(data, target_variable, order):
    series = data[target_variable]
    train_size = int(len(series) * 0.8)
    train_data, test_data = series[:train_size], series[train_size:]

    print(f"Total data points: {len(series)}")
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")

        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()

        forecast = fitted_model.forecast(steps=len(test_data))

        rmse = np.sqrt(mean_squared_error(test_data, forecast))

        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Training Data', color='blue')
        plt.plot(test_data.index, test_data, label='Actual Testing Data', color='red', alpha=0.6)
        plt.plot(test_data.index, forecast, label='Forecasted Data (ARIMA)', color='green', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel(target_variable + ' Price (USD)')
        plt.title(f'ARIMA{order} Forecasting for Google {target_variable} Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"ARIMA Model Order (p, d, q): {order}")
        print(f"\nAn error occurred during ARIMA modeling: {e}")
        print("Suggestion: Try different ARIMA orders (p, d, q).")
```
Run ARIMA Model:
```
arima_model(data, 'Close', order=(5, 1, 0))
```
### OUTPUT:
<img width="901" height="425" alt="image" src="https://github.com/user-attachments/assets/debee141-ad72-4625-aed7-6513bf8872b1" />
<img width="372" height="25" alt="image" src="https://github.com/user-attachments/assets/cd5c698c-568d-4810-b63c-2fe59e8ed10f" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.

