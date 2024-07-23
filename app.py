import pandas as pd
import streamlit as st
import requests
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

        # Use the ARIMA model to predict the stock prices
        start_index = len(model_ARIMA.fittedvalues)
        end_index = start_index + num_dates - 1
        predictions_diff = model_ARIMA.predict(start=start_index, end=end_index)
        
        # Cumulative sum to get the actual prediction
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Ensure predictions list length matches date range length
        if len(predictions) != num_dates:
            raise ValueError("Length of predictions does not match length of date range.")

        # Prepare the results
        results = {
            'date': date_range.strftime('%Y-%m-%d').tolist(),
            'predictions': predictions.tolist()
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
