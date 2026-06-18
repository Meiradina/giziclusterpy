from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)


def clean_data(df):

    # pastikan pakai kolom Menu (sesuai Laravel)
    possible_name_cols = ['Nama Makanan', 'Food Name', 'Menu', 'name']

    name_col = next((c for c in possible_name_cols if c in df.columns), None)

    if name_col is None:
        raise ValueError("Kolom nama tidak ditemukan")

    df = df.rename(columns={name_col: 'Menu'})

    df['Menu'] = df['Menu'].astype(str)

    df['Menu'] = (
        df['Menu']
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )

    # buang data rusak
    df = df[
        (df['Menu'] != '-') &
        (~df['Menu'].str.contains(r'\?|nan|none', case=False, na=False))
    ]

    # duplicate
    df['menu_clean'] = df['Menu'].str.lower().str.strip()
    df = df.drop_duplicates(subset='menu_clean', keep='first')

    # rapikan
    df['Menu'] = df['menu_clean'].str.title()

    df.drop(columns=['menu_clean'], inplace=True)

    return df.reset_index(drop=True)


@app.route('/cluster', methods=['GET'])
def clustering():

    df = pd.read_csv('data/food.csv')

    # CLEANING
    df = clean_data(df)

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

    # preprocessing angka
    for col in features:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(' ', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[features] = df[features].fillna(0)

    # clustering
    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {
        0: 'Protein Sedang',
        1: 'Tinggi Energi Lengkap',
        2: 'Tinggi Karbohidrat',
        3: 'Seimbang'
    }

    df['kategori'] = df['cluster'].map(cluster_labels)

    # 🔥 OUTPUT SESUAI LARAVEL
    return jsonify(df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)