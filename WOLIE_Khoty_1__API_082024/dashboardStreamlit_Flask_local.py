# Importation des bibliothèques nécessaires pour la manipulation de données et les calculs
import numpy as np
import pandas as pd
import random
import warnings
import re
import json
from faker import Faker  # Importation de Faker pour générer des prénoms et noms aléatoires

# Importation des bibliothèques pour la création de graphiques et la visualisation des données
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from PIL import Image

# Importation de la bibliothèque Streamlit pour créer une interface utilisateur interactive
import streamlit as st

# Importation de joblib pour charger et enregistrer des modèles de machine learning
import joblib

# Importation de la bibliothèque requests pour envoyer des requêtes HTTP, utilisée pour communiquer avec l'API
import requests

# Importation de SHAP pour l'explication des prédictions des modèles
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import shap
shap.initjs()

# Ignorer les avertissements de dépréciation spécifiques à Numba
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Configuration de la page Streamlit avec un titre spécifique pour l'onglet du navigateur
st.set_page_config(page_title='Scoring Nouveau Client')

# Fonction pour charger les données à partir d'un fichier CSV
# Le résultat est mis en cache pour optimiser la performance de l'application
@st.cache_data
def charger_donnees(url):
    df = pd.read_csv(url)
    return df

# Essayer de charger les données depuis un chemin local pour les tests
try:
    AnciennesDonnees = charger_donnees(
        r"C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\Application_projet7_Streamlit.csv")
except:
    # Si le chemin local échoue, charger les données depuis un chemin relatif (utilisé pour Streamlit en ligne)
    AnciennesDonnees = charger_donnees("Application//Donnees_generees//Application_projet7_Streamlit.csv")

# Création d'une liste contenant les noms de toutes les colonnes du dataframe chargé
ListeVariables = list(AnciennesDonnees.columns)

# Création d'une barre latérale dans Streamlit permettant de naviguer entre différentes sections (onglets)
option = st.sidebar.selectbox(
    "Sommaire :",
    ("Page d'accueil", "Informations Clients"))

# Ajout de contenu à gauche sur la page d'accueil
if option == "Page d'accueil":
    with st.sidebar:
        st.markdown("### À propos de l'application")
        st.markdown("""
        Cette application a été développée pour permettre une transparence totale dans l'analyse financière, 
        en aidant à comprendre les scores de crédit et les décisions financières de manière claire et accessible.
        """)

        st.markdown("### Conseils Financiers")
        st.markdown("""
        - **Épargnez régulièrement** : Mettez de côté une partie de vos revenus chaque mois.
        - **Minimisez vos dettes** : Évitez de contracter des dettes à haut intérêt.
        - **Investissez intelligemment** : Diversifiez vos investissements pour réduire les risques.
        """)

        st.markdown("### Support & Contact")
        st.markdown("""
        Si vous avez des questions ou avez besoin d'assistance, veuillez contacter notre équipe de support.
        - **Email** : support@finapp.com
        - **Téléphone** : +33 1 23 45 67 89
        - **Chat en ligne** : Disponible 24/7 sur notre site web.
        """)

# Fonction pour ouvrir et afficher une image à partir d'un fichier donné
# La taille de l'image est ajustée selon le paramètre 'taille'
def ouvrir_image(url, taille):
    image = Image.open(url)
    return st.image(image, width=taille)

# Fonction pour charger la liste des nouveaux clients à partir d'un fichier CSV
# La liste est formatée pour être utilisée dans une interface de sélection
def liste_nouveaux_clients(fichier_csv):
    df_nouveaux_clients = pd.read_csv(fichier_csv)
    df_nouveaux_clients.reset_index(inplace=True)
    liste_clients = list(df_nouveaux_clients['SK_ID_CURR'])
    # Ajout d'un client fictif (vide) pour éviter l'affichage de données au démarrage
    liste_clients.insert(0, ' ')
    return liste_clients, df_nouveaux_clients

# Fonction pour afficher un graphique de type jauge, indiquant le niveau de risque ou score du client
def afficher_jauge_client(score):
    couleur_arriere_plan = "#def"
    couleurs_quadrants = [couleur_arriere_plan, "#2bad4e", "#90EE90", "#f2a529", "#f25829"]
    textes_quadrants = ["", "<b>Bon</b>", "<b>Faible</b>", "<b>Très faible</b>", "<b>Ultra faible</b>"]
    nombre_quadrants = len(couleurs_quadrants) - 1

    # Calcul de l'angle de l'aiguille en fonction du score actuel
    valeur_actuelle = score
    valeur_minimale = 0
    valeur_maximale = 100
    longueur_aiguille = np.sqrt(2) / 4
    angle_aiguille = np.pi * (1 - (max(valeur_minimale, min(valeur_maximale, valeur_actuelle)) - valeur_minimale) / (valeur_maximale - valeur_minimale))

    # Création de la figure avec les différents éléments de la jauge
    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(nombre_quadrants) / 2 / nombre_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=couleurs_quadrants,
                text=textes_quadrants,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=50, l=5, r=5),
            width=350,
            height=350,
            paper_bgcolor=couleur_arriere_plan,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>Niveau de remboursement :</b><br>{valeur_actuelle}%",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.2, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            # Ajout de l'aiguille pour indiquer la position du score
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + longueur_aiguille * np.cos(angle_aiguille),
                    y0=0.5, y1=0.5 + longueur_aiguille * np.sin(angle_aiguille),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    # Affichage du graphique dans Streamlit
    return st.plotly_chart(fig)

# Fonction pour générer une explication locale (SHAP) pour un client donné
# Cette explication aide à comprendre quelles caractéristiques influencent le plus la prédiction pour ce client
def expliquer_shap_local(modele_charge, anciennes_donnees):
    # Initialisation de l'explainer SHAP avec le modèle et les données historiques
    explainer = shap.TreeExplainer(modele_charge, anciennes_donnees)
    # Calcul des valeurs SHAP pour le client actuel
    valeurs_shap = explainer(DataClient, check_additivity=False)
    return valeurs_shap[0]

# Fonction pour extraire les variables les plus importantes pour une prédiction donnée
# Retourne une liste des 15 variables ayant le plus d'influence sur la décision du modèle
def recuperer_variables_importantes(valeurs_shap, anciennes_donnees):
    # Récupération des noms des colonnes des données du client
    colonnes = DataClient.columns
    # Création d'un dataframe associant les valeurs SHAP aux noms des colonnes
    variables_importantes = pd.DataFrame(zip(expliquer_shap_local(modele_charge, anciennes_donnees).values, colonnes))
    # Prendre la valeur absolue des valeurs SHAP pour obtenir leur magnitude, puis arrondir
    variables_importantes[0] = abs(variables_importantes[0]).round(2)
    # Trier les variables par ordre décroissant d'importance
    variables_importantes = variables_importantes.sort_values(0, ascending=False)
    # Sélectionner les 15 variables les plus importantes
    variables_principales = list(variables_importantes.iloc[:15][1])
    # Ajouter un choix vide au début de la liste pour l'interface utilisateur
    variables_principales.insert(0, ' ')
    return variables_principales

# Fonction pour tracer des graphiques comparant deux variables sélectionnées par l'utilisateur
# Permet de visualiser les distributions des variables et leur relation
def tracer_graphiques_importants(var1, var2, cible, anciennes_donnees, data_client, liste_resultats):
    # Création de légendes personnalisées pour le graphique de dispersion
    legendes_personnalisees = [Line2D([0], [0], marker='o', color='w', label='Classe 1', markerfacecolor='g'),
                               Line2D([0], [0], marker='o', color='w', label='Classe 0', markerfacecolor='r')]

    # Initialisation de la figure avec une disposition spécifique (subplots)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = fig.subplot_mosaic("""
                            AB
                            CC
                            """)

    # Tracé des courbes de densité pour la première variable sélectionnée
    sns.kdeplot(anciennes_donnees, x=var1, hue=cible, multiple="stack", ax=ax['A'])
    # Ajout d'une ligne verticale indiquant la valeur de la variable pour le client sélectionné
    ax["A"].axvline(data_client[var1].values, linewidth=2, color='r')

    # Tracé des courbes de densité pour la deuxième variable sélectionnée
    sns.kdeplot(anciennes_donnees, x=var2, hue=cible, multiple="stack", ax=ax['B'])
    # Ajout d'une ligne verticale indiquant la valeur de la variable pour le client sélectionné
    ax["B"].axvline(data_client[var2].values, linewidth=2, color='r')

    # Tracé d'un graphique de dispersion pour visualiser la relation entre les deux variables
    ax['C'] = sns.scatterplot(anciennes_donnees, x=var1, y=var2, hue=liste_resultats, palette="blend:red,green", s=100)
    # Ajout d'un marqueur pour représenter le client sélectionné
    ax['C'] = sns.scatterplot(data_client, x=var1, y=var2, s=400, hue=var2, palette=['blue'], marker='*')
    # Affichage des légendes personnalisées
    ax['C'] = plt.legend(handles=legendes_personnalisees, loc='upper right')

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

# Configuration de la première page "Page d'accueil" de l'application
if option == "Page d'accueil":
    # Titre de la page d'accueil centré et stylisé
    st.markdown("<h1 style='text-align: center; color: red;'>Bonjour, Bienvenue sur votre portail de transparence financière :</h1>", unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')

    # Affichage du logo de l'entreprise
    # Division en colonnes pour centrer l'image au milieu de la page
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        try:
            ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Images\Logo_Entreprise_pres_a_depenser.png', 300)
        except:
            ouvrir_image('Application/Images/Logo_Entreprise_pres_a_depenser.png', 300)

    st.markdown('')
    st.markdown('')

    # Slogan de l'application centré et stylisé
    st.markdown("<h2 style='text-align: center; color: grey;'>Faciliter l’analyse et la compréhension des résultats pour tous les utilisateurs.</h2>", unsafe_allow_html=True)
    st.markdown('')

    # Description du but de l'application
    st.markdown("<h3 style='font-weight:bold;'>Objectif de l’application:</h3>", unsafe_allow_html=True)
    st.markdown('')
    st.markdown("""
    <p style='text-align: center; font-size:22px;'>
    Aider les clients et conseillers à naviguer facilement dans les données et à prendre des décisions éclairées.
    </p>
    """, unsafe_allow_html=True)

# Configuration de la deuxième page "Informations Clients" de l'application
if option == "Informations Clients":
    # Titre de la page centré et stylisé
    st.markdown("<h2 style='text-align: center; color: green;'>Informations Clients :</h2>", unsafe_allow_html=True)

    # Chargement de la liste des clients à partir d'un fichier CSV
    try:
        liste_clients = liste_nouveaux_clients(
            r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\ListeNouveauxClients.csv')[0]
        df_nouveaux_clients = liste_nouveaux_clients(
            r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\ListeNouveauxClients.csv')[1]
    except:
        liste_clients = liste_nouveaux_clients('Application/Donnees_generees/ListeNouveauxClients.csv')[0]
        df_nouveaux_clients = liste_nouveaux_clients('Application/Donnees_generees/ListeNouveauxClients.csv')[1]

    # Fonction pour sélectionner un client à partir de la liste déroulante
    def selectionner_client(liste_clients):
        client = st.selectbox('Veuillez choisir le numéro de votre client : ', liste_clients)
        return client

    # Stockage du numéro de client sélectionné
    client = selectionner_client(liste_clients)

    # Si aucun client n'est sélectionné, afficher un message par défaut
    if client == ' ':
        st.markdown("### Veuillez sélectionner un client pour afficher ses informations détaillées.")
        st.markdown("L'application vous permet de visualiser les scores de crédit, l'historique des clients et d'autres informations financières importantes.")
    else:
        # Récupération de l'index du client sélectionné dans la liste des nouveaux clients
        index_client = list(df_nouveaux_clients[df_nouveaux_clients['SK_ID_CURR'] == client]['index'].values)
        for i in index_client:
            index_client = i
        # Création d'une liste des index des autres clients pour éviter de charger des données inutiles
        index_autres = list(df_nouveaux_clients['index'])
        index_autres.remove(index_client + 1)
        index_autres.remove(0)

        # Chargement des données du client sélectionné à partir d'un fichier CSV
        try:
            DataClient = pd.read_csv(
                r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Donnees_generees\NouveauxClientsReduits.csv',
                skiprows=index_autres, nrows=1)
        except:
            DataClient = pd.read_csv('Application/Donnees_generees/NouveauxClientsReduits.csv', skiprows=index_autres, nrows=1)

        # Nettoyage des noms de colonnes pour enlever les caractères spéciaux
        DataClient = DataClient.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # Extraction du numéro de client
        numero_client = DataClient['SK_ID_CURR'].values
        for i in numero_client:
            numero_client = i
        # Affichage du numéro de client dans l'interface
        st.write('**N° Client :** ', numero_client)
        # Sélection des variables à afficher pour ce client, en excluant la cible 'TARGET'
        DataClient = DataClient[ListeVariables]
        DataClient = DataClient.drop(columns='TARGET')

        # Initialisation de Faker pour générer des données en français
        fake = Faker('fr_FR')

        # Génération aléatoire d'un prénom et d'un nom pour le client
        prenom_selectionne = fake.first_name()
        nom_selectionne = fake.last_name()

        # Affichage dans Streamlit
        st.write('**Prénom :** ', prenom_selectionne)
        st.write('**Nom :** ', nom_selectionne)

        # Titre pour la section des résultats du prêt
        st.markdown("<h2 style='text-align: center; color: green;'>  Résultats du prêt :</h2>", unsafe_allow_html=True)

        # Envoi des données du client à une API Flask pour obtenir une prédiction du score
        url = 'http://localhost:3000/api/'

        # Conversion des données du client en format JSON pour les envoyer à l'API
        ListeVariables.remove('TARGET')
        data = DataClient.values.tolist()
        j_data = json.dumps(data)
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=j_data, headers=headers)

        # Affichage de la réponse de l'API pour débogage
        st.write("Réponse brute de l'API : ", r.text)

        # Extraction du score de la réponse de l'API et arrondi pour l'affichage
        try:
            # La réponse semble être une liste de listes, donc nous devons traiter cela correctement
            prediction = json.loads(r.text)
            score = int(round(prediction[0][1] * 100, 2))
        except Exception as e:
            st.error(f"Erreur : La réponse de l'API n'est pas formatée comme prévu. Détails : {e}")
            score = 50  # Valeur par défaut si l'API ne fonctionne pas correctement

        # Si l'API n'est pas utilisée, charger le modèle directement depuis un fichier
        try:
            modele_charge = joblib.load(open(
                r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Notebook\mlflow_runs\290555362347125930\1e46374402274ffe9572106d93203ef9\artifacts\model\model.pkl',
                'rb'))
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")

        # Affichage du score du client sous forme de jauge
        col1, col2 = st.columns(2)
        with col2:
            afficher_jauge_client(score)

        # Affichage d'une image conditionnée au score (rouge pour un score faible, vert pour un bon score)
        with col1:
            if score >= 59:
                try:
                    ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Images\pngtree_vert.png', 300)
                except:
                    ouvrir_image('Application/Images/pngtree_vert.png', 300)
            if score < 59:
                try:
                    ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Images\pngtree_rouge.png', 300)
                except:
                    ouvrir_image('Application/Images/pngtree_rouge.png', 300)

        # Affichage de l'importance globale des variables (features) via une image SHAP pré-générée
        st.markdown("<h2 style='text-align: center; color: green;'>Importance des caractéristiques globales :</h2>", unsafe_allow_html=True)
        try:
            ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 7 ML\Application\Images\SHAP_Image_Globale.png', 600)
        except:
            ouvrir_image('Application/Images/SHAP_Image_Globale.png', 700)

        # Affichage de l'importance locale des variables (features) pour ce client spécifique
        st.markdown("<h2 style='text-align: center; color: green;'>Importance des caractéristiques locales :</h2>", unsafe_allow_html=True)
        valeurs_shap = expliquer_shap_local(modele_charge, AnciennesDonnees)

        # Tracé d'un graphique en cascade (waterfall) pour visualiser les contributions des principales variables
        shap.waterfall_plot(valeurs_shap, max_display=10)
        fig = plt.gcf()  # Récupère la figure courante pour l'afficher de manière thread-safe
        st.pyplot(fig)

        # Création d'une colonne `predict_proba` pour l'ensemble des données, indiquant la probabilité de risque
        cible = AnciennesDonnees['TARGET']
        AnciennesDonnees = AnciennesDonnees.drop(columns='TARGET')
        resultats = modele_charge.predict_proba(AnciennesDonnees)
        resultats = pd.DataFrame(resultats)
        resultats = 100 * resultats
        resultats = resultats.astype(int)
        liste_resultats = list(resultats[1])

        # Analyse graphique de deux variables sélectionnées par l'utilisateur
        st.markdown("<h2 style='text-align: center; color: green;'>Analyse des variables importantes :</h2>", unsafe_allow_html=True)

        variable1 = st.selectbox('Veuillez choisir la variable N°1 : ', recuperer_variables_importantes(valeurs_shap, AnciennesDonnees))
        variable2 = st.selectbox('Veuillez choisir la variable N°2 : ', recuperer_variables_importantes(valeurs_shap, AnciennesDonnees))

        # Affichage des graphiques uniquement si deux variables sont sélectionnées
        if variable1 == ' ' or variable2 == ' ':
            st.write('')

        else:
            tracer_graphiques_importants(variable1, variable2, cible, AnciennesDonnees, DataClient, liste_resultats)