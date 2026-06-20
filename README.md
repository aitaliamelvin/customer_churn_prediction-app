# 📉 Customer Churn Prediction

## Présentation

Ce projet a pour objectif de prédire le risque de départ (churn) d'un client d'une entreprise de télécommunications.

L'application permet à un utilisateur de saisir les caractéristiques d'un client et d'obtenir une estimation de son risque de résiliation.

Le projet couvre l'ensemble du cycle de vie d'un modèle de Machine Learning :

* Analyse exploratoire des données (EDA)
* Préparation des données
* Entraînement de modèles
* Évaluation des performances
* Déploiement via FastAPI
* Interface utilisateur avec Streamlit

---

## Objectif métier

Le churn représente la perte de clients pour une entreprise.

Pouvoir identifier les clients à risque permet :

* d'améliorer la fidélisation ;
* de réduire les pertes de revenus ;
* de mettre en place des actions commerciales ciblées.

---

## Dataset

Dataset utilisé :

Telco Customer Churn Dataset

Variables principales :

* Informations démographiques
* Services souscrits
* Méthodes de paiement
* Ancienneté du client
* Montants facturés
* Valeur client (CLTV)

Variable cible :

* Churn Value

  * 0 : client conservé
  * 1 : client perdu

---

## Méthodologie

### 1. Analyse exploratoire

Analyse des variables :

* Tenure Months
* Contract
* Tech Support
* Internet Service
* Monthly Charges
* Total Charges

Étude des relations entre les variables et le churn.

### 2. Préparation des données

* Suppression des variables inutiles
* Gestion des valeurs manquantes
* Encodage des variables catégorielles
* Séparation train / test

### 3. Modélisation

Modèles testés :

* Logistic Regression
* Random Forest

### 4. Évaluation

Métriques utilisées :

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

Le modèle retenu est la régression logistique.

---

## Résultats

Performances du modèle :

* Accuracy : 0.80
* Precision : 0.65
* Recall : 0.57
* F1 Score : 0.61
* ROC AUC : 0.85

Le modèle permet d'identifier efficacement les clients présentant un risque élevé de départ.

---

## Architecture

Utilisateur

↓

Streamlit

↓

FastAPI

↓

Pipeline Scikit-Learn

↓

Prédiction

---

## Installation

Cloner le projet :

```bash
git clone <url-du-repo>
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'API :

```bash
python -m uvicorn app.main:app --reload
```

Lancer l'interface :

```bash
streamlit run streamlit_app.py
```

---

## Utilisation

1. Ouvrir l'application Streamlit.
2. Compléter les informations du client.
3. Cliquer sur le bouton de prédiction.
4. Consulter le risque de churn estimé.

---

## Améliorations futures

* Déploiement cloud
* Explication des prédictions (SHAP)
* Tableau de bord analytique
* Suivi des performances du modèle
* Réentraînement automatique

```
```
