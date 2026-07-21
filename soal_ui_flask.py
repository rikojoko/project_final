import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Konfigurasi Path Model Multi-Output Anda
MODEL_PATH = 'model_multitask_diagnosa.pkl'

# Memuat model saat server startup
model = None
if os.path.exists(MODEL_PATH):
    with open (MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

# Route Utama: Tampilan Awal Sistem (Keadaan Standby)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Route Analisis: Memproses Form POST dan Mengembalikan Dua Output Sekaligus
@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return render_template('index.html', error="Model .pkl tidak ditemukan di direktori.")
    try:
        symptom = request.form.get('symptom')
        usia = request.form.get('usia', type=int)
        tekanan_darah = request.form.get('tekanan_darah')
        kadar_gula = request.form.get('kadar_gula', type=float)

        inputs = {
            'symptom': symptom,
            'usia': usia,
            'tekanan_darah': tekanan_darah,
            'kadar_gula': kadar_gula
        }
        
        input_df = pd.DataFrame({
            'symptom_description': [symptom],
            'usia': [usia],
            'tekanan_darah': [tekanan_darah],
            'kadar_gula': [kadar_gula]
        })

        prediction = model.predict(input_df)
        
        diagnosa_raw = prediction[0][0]
        diagnosa_format = str(diagnosa_raw).replace('_', ' ').upper()

        terapi_format = prediction[0][1]

        outputs = {
            'diagnosis': diagnosa_format,
            'therapy': terapi_format
        }

        return render_template('index.html', inputs=inputs, outputs=outputs)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Jalankan server lokal di port 5000 dengan mode debug aktif
    app.run(debug=True, port=8000)