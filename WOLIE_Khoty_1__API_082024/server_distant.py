import joblib
import boto3
import os
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

s3 = boto3.client('s3')
bucket_name = 'mon-projet-ml'  # Nom réel de votre bucket
model_path = 'model/model.pkl'

# Télécharger le modèle depuis S3
s3.download_file(bucket_name, model_path, 'model.pkl')

# Charger le modèle avec joblib
model = joblib.load('model.pkl')

@app.route('/api/', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    prediction = model.predict_proba(df.values.tolist())
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
