from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn

app = Flask(__name__)

# Charger le modèle MLflow à partir du chemin spécifié
model_path = "file:///C:/Users/Infogene/Documents/Khoty_Privé/DOSSIER%20FORMATION%20DATA%20SCIENTIST/PROJET%207%20ML/Notebook/mlflow_runs/290555362347125930/1e46374402274ffe9572106d93203ef9/artifacts/model"
model = mlflow.sklearn.load_model(model_path)

@app.route('/api/', methods=['POST'])
def predict():
    # Recevoir les données JSON envoyées par le client
    data = request.json
    # Convertir les données en DataFrame pour faire la prédiction
    df = pd.DataFrame(data)
    # Faire la prédiction
    prediction = model.predict_proba(df.values.tolist())
    # Retourner la prédiction sous forme de JSON
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
