import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load model dan encoder
with open('random_forest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encode.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Fungsi untuk prediksi
def predict_fruit(diameter, weight, red, green, blue):
    features = np.array([[diameter, weight, red, green, blue]])
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]

# Judul aplikasi
st.title("Aplikasi Prediksi Buah")

# Input dari pengguna
diameter = st.number_input("Masukkan Diameter (cm):", min_value=0.0)
weight = st.number_input("Masukkan Berat (gram):", min_value=0.0)
red = st.number_input("Masukkan Nilai Merah (0-255):", min_value=0, max_value=255)
green = st.number_input("Masukkan Nilai Hijau (0-255):", min_value=0, max_value=255)
blue = st.number_input("Masukkan Nilai Biru (0-255):", min_value=0, max_value=255)

# Tombol untuk prediksi
if st.button("Prediksi"):
    result = predict_fruit(diameter, weight, red, green, blue)
    st.success(f"Buah yang diprediksi: {result}")

    