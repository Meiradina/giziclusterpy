from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

@app.route('/cluster', methods=['GET'])
def clustering():
    # load dataset
    df = pd.read_csv('data/food.csv')
    
    df.rename(columns={'Food Name': 'name'}, inplace=True)
    
    # fitur nutrisi
    features = [
        'Energy (kJ)',
        'Protein (g)',
        'Fat (g)',
        'Carbohydrates (g)',
        'Vitamin C (mg)',
        'Calcium (mg)',
        'Dietary Fiber (g)',
        'Iron (mg)'
    ]
    
    # Preprocessing 
    for col in features:

        # ubah ke string
        df[col] = df[col].astype(str)

        # hapus spasi
        df[col] = df[col].str.replace(' ', '', regex=False)

        # ubah koma jadi titik
        df[col] = df[col].str.replace(',', '.', regex=False)

        # convert ke angka
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[features] = df[features].fillna(0)

    # ambil fitur
    X = df[features]
    
    # normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # mapping cluster ke kategori
    cluster_labels = {
        0 : 'Protein Sedang',
        1 : 'Tinggi Energi Lengkap',
        2 : 'Tinggi Karbohidrat',
        3 : 'Seimbang',
    }

    df['kategori'] = df['cluster'].map(cluster_labels)

    # ubah ke JSON
    result = df.to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)