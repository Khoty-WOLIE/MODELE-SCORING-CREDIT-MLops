import pytest
import pandas as pd
from dashboardStreamlit_Flask_distant_heroku import charger_donnees, liste_nouveaux_clients, afficher_jauge_client

def test_charger_donnees():
    df = charger_donnees(r"C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\Application_projet7_Streamlit.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] > 0
    assert not df.empty

def test_liste_nouveaux_clients():
    liste_clients, df_clients = liste_nouveaux_clients(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\ListeNouveauxClients.csv')
    assert isinstance(liste_clients, list)
    assert len(liste_clients) > 0
    assert ' ' in liste_clients
    assert isinstance(df_clients, pd.DataFrame)
    assert not df_clients.empty

def test_afficher_jauge_client():
    for score in [10, 50, 90]:
        result = afficher_jauge_client(score)
        assert result is not None

if __name__ == '__main__':
    pytest.main()
