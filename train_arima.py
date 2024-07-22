import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load data from local folder
file_path = 'data/TLKM.JK.csv'  
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(['Open', 'Low', 'Close', 'Adj Close', 'Volume'], axis='columns')

# Convert 'Date' to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Initialize the 'High' column as time series data
ts = data['High']

# Train the ARIMA model
model_ARIMA = ARIMA(ts, order=(1, 1, 1))
result_ARIMA = model_ARIMA.fit()

# Save the trained model to a file
with open('model/arima_model.pkl', 'wb') as file:
    pickle.dump(result_ARIMA, file)

print("Model trained and saved successfully.")

# Predict future values (example for the next 10 periods)
forecast_steps = 10
forecast = result_ARIMA.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['forecast'])

print("Forecast:")
print(forecast_df)
