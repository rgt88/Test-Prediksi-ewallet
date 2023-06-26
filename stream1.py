import streamlit as st
import pickle
import pandas as pd

# Fungsi untuk melakukan prediksi menggunakan model Prophet
def predict_next_day(model, target_date):
    future = model.make_future_dataframe(periods=1, freq='D')
    forecast = model.predict(future)

    prediction = forecast[forecast['ds'] == target_date]['yhat'].values[0]
    return prediction

# Fungsi untuk mengubah format tanggal menjadi string 'YYYY-MM-DD'
def format_date(year, month, day):
    return f'{year}-{month:02d}-{day:02d}'

# Memuat model dari file .sav
model_path = 'Predict Total ovopay.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# Menampilkan judul aplikasi
st.title('Prediksi Time Series menggunakan Model Prophet')

# Input tanggal
st.header('Input Tanggal')
year = st.number_input('Tahun', value=2023)
month = st.number_input('Bulan', min_value=1, max_value=12, value=6)
day = st.number_input('Tanggal', min_value=1, max_value=31, value=27)

# Memeriksa apakah tanggal valid
if day > 0 and month > 0 and year > 0:
    try:
        target_date = format_date(year, month, day)

        # Melakukan prediksi
        prediction = predict_next_day(loaded_model, target_date)

        # Menampilkan hasil prediksi
        st.header('Hasil Prediksi')
        st.write(f'Prediksi untuk tanggal {target_date}: {prediction:.2f}')
    except IndexError:
        st.write('Data tidak tersedia untuk tanggal yang dimasukkan.')
else:
    st.write('Masukkan tanggal yang valid.')
