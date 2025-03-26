## Aperçu de l'entreprise

![Aperçu du site web](images/DS_projet7.PNG)

## 📌 Contexte professionnel

En tant que **Data Scientist** chez **Prêt à Dépenser**, entreprise spécialisée dans les crédits à la consommation, j’ai été chargé de développer un **modèle de scoring crédit** afin d’évaluer automatiquement la probabilité de remboursement d’un client.  
L’objectif est double :
1. Développer un modèle prédictif robuste basé sur les données clients.
2. Intégrer ce modèle dans un **pipeline MLOps complet** pour son suivi, son déploiement et la détection de dérive de données.

---

## 🎯 Objectifs de la mission

- Créer un **modèle de scoring métier** optimisé pour les décisions de crédit.
- Intégrer un système de suivi des performances avec **MLFlow**.
- Déployer une **API de scoring** sur le Cloud.
- Créer une interface de test via **Streamlit**.
- Mettre en place une détection de **Data Drift** avec **Evidently**.

---

## 🧩 Étapes de réalisation

### 🔹 Mission 1 : Élaboration du modèle de scoring

#### 1. Environnement d’expérimentation MLFlow
- Suivi des versions, des hyperparamètres et des métriques.
- Mise en place du **model registry**.

#### 2. Préparation des données
- Nettoyage, jointure et feature engineering à partir de kernels Kaggle.
- Gestion des valeurs manquantes, encodage des variables, etc.

#### 3. Création d’un **score métier**
- Définition d’un score pondéré prenant en compte les **coûts associés aux faux positifs et faux négatifs**.
- Optimisation du **seuil de classification** en fonction du coût métier.

#### 4. Comparaison des modèles
- Modèles testés : **Logistic Regression**, **Random Forest**, **XGBoost**.
- Utilisation de **GridSearchCV** et gestion du déséquilibre des classes.
- Analyse globale et locale des features importantes (via **SHAP**, **LIME**).

---

### 🔹 Mission 2 : Mise en œuvre MLOps

#### 1. Déploiement de l’API de scoring
- Développement d’une API avec **FastAPI** ou **Flask**.
- Automatisation du déploiement avec **GitHub Actions** vers une plateforme Cloud.

#### 2. Interface de test via Streamlit
- Interface utilisateur pour tester le modèle en direct.
- Simulation de requêtes API avec affichage des scores de risque.

#### 3. Détection de **Data Drift**
- Implémentation de **Evidently** pour comparer les données d’entraînement vs production.
- Génération d’un rapport HTML illustrant les dérives sur les principales features.

#### 4. Préparation de la soutenance
- Diaporama professionnel (30 slides max) retraçant la démarche :
  - Données → Modèle → MLOps → Surveillance
- Justification des choix techniques et résultats métiers.

---

## 📂 Livrables

- **Notebook de préparation des données et modélisation**
- **Interface MLFlow** configurée et utilisée
- **API de scoring** (code et documentation)
- **Interface Streamlit** de test
- **Rapport Evidently** de détection de dérive
- **Présentation finale** du projet

---

## 🛠️ Outils et technologies

- **Python** : pandas, scikit-learn, XGBoost, SHAP
- **MLFlow** : suivi des modèles et des expérimentations
- **FastAPI / Flask** : déploiement d’API
- **Streamlit** : interface utilisateur
- **GitHub Actions** : CI/CD automatisée
- **Evidently** : détection de **data drift**
- **Jupyter Notebooks** pour la modélisation

---

## ✅ Résultats et impact

- Modèle de scoring robuste avec score métier personnalisé
- Déploiement opérationnel via une API testable et documentée
- Pipeline MLOps complet avec suivi, test et surveillance
- Système prêt à l’intégration dans un SI bancaire ou fintech

---

## 🔍 Aperçu

> Ce projet illustre mes compétences en **data science appliquée au scoring**, en **mise en production de modèles** et en **mise en place d’un système MLOps complet**, de l’expérimentation à la surveillance.

---

*Projet réalisé dans un cadre professionnel simulé, avec des responsabilités comparables à celles d’un Data Scientist en environnement bancaire ou fintech.*
