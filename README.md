# OPC_DATA_SCIENTIST_PROJET7
Implémentez un modèle de scoring


# Modèle de Scoring Crédit et Intégration MLOps - Prêt à Dépenser

## Contexte

Je suis Data Scientist chez **"Prêt à dépenser"**, une entreprise spécialisée dans les crédits à la consommation pour des clients ayant peu ou pas d'historique de prêt. L'entreprise souhaite développer un **modèle de scoring de crédit** pour évaluer automatiquement la probabilité de remboursement d’un client. Ce modèle classifie chaque demande en deux catégories : **crédit accordé** ou **refusé**. 

Le projet est divisé en deux missions :
1. **Mission 1** : Élaborer le modèle de scoring avec une approche MLOps complète, incluant l'évaluation des modèles et leur mise en production.
2. **Mission 2** : Intégrer et optimiser un système MLOps pour le suivi du cycle de vie du modèle et la détection de dérive de données (Data Drift).

---

## Première Mission : Élaboration du Modèle de Scoring Crédit

### Objectifs
- Construire un modèle de scoring prédictif pour évaluer la probabilité de défaut de paiement.
- Analyser les features les plus importantes pour la prise de décision (globale et locale).
- Mettre en place un environnement MLOps pour gérer le suivi des expérimentations, la version des modèles et l’optimisation.

### Étape 1 : Préparation de l’Environnement d'Expérimentation
- **Objectif** : Mettre en place un environnement d’expérimentation **MLFlow** pour le suivi des entraînements des modèles et la gestion des versions.
- **Détails** :
  - Initialiser un environnement **MLFlow** pour suivre les métriques et les hyperparamètres des modèles.
  - Lancer l’UI MLFlow pour visualiser et comparer les différentes expérimentations.
  - Centraliser le stockage des modèles dans un **model registry**.
- **Livrable** : Un notebook avec l’environnement MLFlow initialisé et les premières expérimentations enregistrées.

### Étape 2 : Préparation des Données à la Modélisation
- **Objectif** : Préparer les données pour la création du modèle de scoring en adaptant des kernels Kaggle.
- **Détails** :
  - Sélectionner et adapter un **kernel Kaggle** pertinent pour la préparation des données et le **feature engineering**.
  - Joindre les différentes tables de données pour constituer un dataset complet.
  - Effectuer le nettoyage et la préparation des données (gestion des valeurs manquantes, encodage des variables catégorielles, etc.).
- **Livrable** : Un notebook contenant la préparation complète des données, prêtes pour l’entraînement des modèles.

### Étape 3 : Création d'un Score Métier
- **Objectif** : Définir un score métier pour optimiser le modèle en prenant en compte les coûts associés aux faux positifs et faux négatifs.
- **Détails** :
  - Définir un **score métier** pondéré pour minimiser l’erreur de prédiction en fonction des coûts d’un faux négatif (FN) et d’un faux positif (FP).
  - Optimiser le seuil de classification pour minimiser le coût métier, et non uniquement l’accuracy ou l’AUC.
  - Comparer les modèles en fonction de ce score métier et choisir les meilleurs hyperparamètres.
- **Livrable** : Un notebook contenant le calcul du score métier et l’optimisation des modèles en fonction de ce score.

### Étape 4 : Simulation et Comparaison de Modèles
- **Objectif** : Tester et comparer plusieurs modèles en utilisant une validation croisée et un **GridSearchCV** pour l'optimisation des hyperparamètres.
- **Détails** :
  - Utiliser plusieurs modèles de classification (par exemple : **Logistic Regression**, **Random Forest**, **XGBoost**).
  - Comparer les résultats avec une baseline et utiliser des techniques pour gérer le déséquilibre des classes.
  - Analyser la **feature importance globale** et **locale** (SHAP, LIME) pour le meilleur modèle.
- **Livrable** : Un notebook contenant les simulations de modèles, leurs comparaisons, et l’analyse des features importantes.

---

## Deuxième Mission : Intégration et Optimisation du Système MLOps

### Objectifs
- Intégrer le modèle de scoring dans un système MLOps complet, incluant le déploiement de l’API et la détection de **Data Drift**.
- Mettre en place une solution pour suivre les performances du modèle en production et détecter d’éventuelles dérives de données.

### Étape 1 : Déploiement de l’API dans le Cloud
- **Objectif** : Mettre en production l'API du modèle de scoring pour calculer la probabilité de défaut de paiement des clients.
- **Détails** :
  - Utiliser **Git** pour la gestion de version du code de l’API.
  - Mettre en production l’API en utilisant **Github Actions** pour automatiser le déploiement dans une solution Cloud (AWS, Google Cloud, ou autre).
  - Utiliser **FastAPI** ou **Flask** pour implémenter l’API.
- **Livrable** : Un code d'API déployé sur le Cloud et accessible via des requêtes HTTP.

### Étape 2 : Mise en Place d'une Interface de Test pour l'API
- **Objectif** : Créer une interface utilisateur pour tester le modèle de scoring à travers l'API.
- **Détails** :
  - Utiliser **Streamlit** pour créer une interface utilisateur simple permettant de tester l'API avec des données de clients en entrée.
  - Afficher les résultats de scoring en temps réel pour simuler l’usage du modèle en production.
- **Livrable** : Une interface **Streamlit** permettant de tester l'API de scoring de manière interactive.

### Étape 3 : Détection du Data Drift avec Evidently
- **Objectif** : Mettre en place un système de détection de dérive de données (Data Drift) pour surveiller les performances du modèle en production.
- **Détails** :
  - Utiliser la librairie **Evidently** pour analyser la dérive de données entre le dataset d'entraînement et les nouvelles données de production.
  - Comparer les distributions des principales features et générer un rapport HTML montrant les dérives détectées.
  - Tester la détection de dérives sur les datasets `application_train` (modélisation) et `application_test` (production simulée).
- **Livrable** : Un rapport Evidently en HTML montrant les résultats de l'analyse de dérive de données.

### Étape 4 : Vérification et Préparation de la Soutenance
- **Objectif** : Préparer une présentation résumant l'ensemble du processus de modélisation, les décisions prises, et les résultats obtenus.
- **Détails** :
  - Structurer la présentation autour des étapes clés : préparation des données, création du modèle, mise en production, et suivi en production.
  - Justifier les choix méthodologiques et discuter des résultats du modèle ainsi que des optimisations mises en œuvre.
  - Prévoir des questions à poser lors du débriefing pour obtenir des retours constructifs.
- **Livrable** : Une présentation prête pour la soutenance (30 slides maximum) résumant les résultats et les décisions du projet.

---

## Détails Techniques

- **Fichiers** :
  - `Dataset Prêt à Dépenser` : Contient les données des clients et des institutions financières.
  - **Notebook de Modélisation** : Contient la préparation des données, l’entraînement des modèles, et la gestion des expérimentations via MLFlow.
  - **API de Scoring** : Code de l’API déployée pour le scoring des clients.
  - **Script Evidently** : Contient l’analyse du Data Drift pour surveiller la performance du modèle en production.

- **Outils Utilisés** :
  - **Python** (pandas, scikit-learn, MLFlow) pour la préparation des données, la modélisation, et l’optimisation.
  - **FastAPI** ou **Flask** pour l’API.
  - **Streamlit** pour l’interface de test.
  - **Github Actions** pour l’automatisation du déploiement.
  - **Evidently** pour la détection de dérive de données.
  - **MLFlow** pour le suivi des expérimentations et la gestion des modèles.

- **Compétences Utilisées** :
  - Modélisation de scoring avec gestion du déséquilibre des classes.
  - Optimisation des hyperparamètres avec **GridSearchCV** et suivi des expérimentations avec **MLFlow**.
  - Déploiement d’une API et gestion de l’intégration continue avec **Github Actions**.
  - Détection de **Data Drift** avec **Evidently**.

## Résumé

Ce projet vise à élaborer un **modèle de scoring** pour prédire la probabilité de défaut de paiement des clients de **"Prêt à dépenser"**. Le projet est structuré en deux missions : la création du modèle avec une optimisation métier, et l'intégration MLOps pour assurer un suivi efficace

 du modèle en production.
