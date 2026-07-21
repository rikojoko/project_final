import streamlit as st
import pandas as pd
import pickle
import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

# --- LOAD MODEL MULTI-OUTPUT ---
@st.cache_resource
def load_multitask_model():
    nama_file = 'model_multitask_diagnosa.pkl'
    if os.path.exists(nama_file):
        with open(nama_file, 'rb') as file:
            return pickle.load(file)
    return None

model = load_multitask_model()

# --- SETUP UI streamlit ---
st.title("🧠 AI Multi-Output Assistant")
st.markdown("Model ini memprediksi **Diagnosa** sekaligus **Terapi** secara langsung menggunakan Machine Learning.")

with st.container(border=True):
    st.markdown("### 📋 Input Kondisi Klinis")

    symptom_input = st.text_area("Keluhan Utama / Gejala", value="sakit perut, mual")

    col1, col2 = st.columns(2)
    with col1:
        usia_input = st.number_input("Usia", min_value=1, max_value=120, value=30, step=1)
    with col2:
        gula_input = st.number_input("Kadar Gula Darah (mg/dL)", min_value=10, max_value=500, value=100, step=1)

    opsi_td = df['tekanan_darah'].unique() if 'df' in locals() else ['120/80', '110/70', '130/85', '115/75', '130/90', '140/90', '150/95', '160/100', '100/65']
    default_td_index = list(opsi_td).index("120/80") if "120/80" in opsi_td else 0
    td_input = st.selectbox("Tekanan Darah", options=opsi_td, index=default_td_index)

    submit_button = st.button("Jalankan prediksi AI Ganda")

# --- EKSEKUSI PREDIKSI ---
if submit_button:
    if model is not None:
        input_data = pd.DataFrame({
            'symptom_description': [symptom_input],
            'usia': [usia_input],
            'tekanan_darah': [td_input],
            'kadar_gula': [gula_input]
        })

        prediksi = model.predict(input_data)
        diagnosa_pred = prediksi[0][0]
        terapi_pred = prediksi[0][1]

        st.success("📊 Hasil Prediksi Model Machine Learning")
        st.caption("Hasil Diagnosa Penyakit")

        st.header(diagnosa_pred.replace('_', ' ').title())
        st.markdown("### 💊 Rekomendasi Tindakan / Terapi (Hasil Prediksi AI):")
        st.info(terapi_pred)

        st.caption("ℹ️ Kedua output di atas diproduksi langsung secara matematis oleh model `.pkl` Anda berdasarkan pola dari data latih.")
    else:
        st.error("Model tidak ditemukan. Pastikan file 'model_multitask_diagnosa.pkl' sudah ada di direktori yang sama.")