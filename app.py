import pandas as pd
import streamlit as st
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load the trained ARIMA model
try:
    with open('model/arima_model.pkl', 'rb') as file:
        model_ARIMA = pickle.load(file)
    st.success('Model ARIMA berhasil dimuat')
except FileNotFoundError:
    st.error('File model tidak ditemukan. Pastikan path file benar.')
    st.stop()

st.title('Data Historis Saham PT. Telkom')

start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

def predict(start_date, end_date):
    try:
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        num_dates = len(date_range)

        # Use the ARIMA model to forecast the stock prices
        forecast_result = model_ARIMA.get_forecast(steps=num_dates)
        forecast = forecast_result.predicted_mean
        
        # Ensure forecast list length matches date range length
        if len(forecast) != num_dates:
            raise ValueError("Length of forecast does not match length of date range.")

        # Prepare the results
        results = {
            'date': date_range.strftime('%Y-%m-%d').tolist(),
            'predictions': forecast.tolist()
        }
        
        return results
    except Exception as e:
        return {'error': str(e)}

if st.button('Prediksi'):
    if start_date and end_date:
        st.write(f'Start Date: {start_date}')
        st.write(f'End Date: {end_date}')
        results = predict(start_date, end_date)
        if 'error' in results:
            st.write('Terjadi kesalahan:', results['error'])
        else:
            # Create DataFrame from results
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])  # Convert date strings to datetime objects
            df.set_index('date', inplace=True)  # Set date as index
            
            # Display results as a table
            st.write('Hasil prediksi:')
            st.dataframe(df)  # Menampilkan DataFrame sebagai tabel
            
            # Display results as a line chart
            st.line_chart(df)
    else:
        st.write('Silakan masukkan tanggal mulai dan tanggal akhir.')
