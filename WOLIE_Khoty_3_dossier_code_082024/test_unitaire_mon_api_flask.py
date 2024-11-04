import json
import pytest
import sys
import os
import numpy as np

# Ajouter le chemin du répertoire parent au chemin de recherche de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from server_local import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Créer un ensemble de données de test avec 619 caractéristiques
data = [np.zeros(619).tolist()]  # les vrais valeurs sont beaucoup a répertoriées

def test_predict_endpoint(client):
    response = client.post('/api/', json=data)
    
    assert response.status_code == 200
    
    result = json.loads(response.data)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], list)
    assert len(result[0]) == 2  # Probabilité pour les deux classes

if __name__ == '__main__':
    pytest.main()
