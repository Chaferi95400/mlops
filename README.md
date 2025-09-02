MLOps – Course Project


Ce dépôt contient les diapositives, les laboratoires et un projet complet démontrant un workflow MLOps de base de bout en bout. Le projet final consiste en un microservice de Machine Learning qui sert un modèle de classification du cancer du sein, avec un suivi des expériences via MLflow.

Architecture Finale


Le projet est structuré pour séparer clairement l'entraînement du service de prédiction :


- labs/Lab_1/train.py: Un script responsable de l'entraînement du modèle. Il utilise scikit-learn pour créer un pipeline de prétraitement et un classificateur LogisticRegression.
- project/ml_microservice/preprocessing.py: Contient la logique de feature engineering personnalisée (add_combined_feature). Ce code est partagé entre l'entraînement et l'inférence pour garantir la cohérence.
- MLflow: Utilisé pour suivre les expériences d'entraînement. Chaque exécution enregistre les hyperparamètres, les métriques de performance (score CV) et l'artefact du modèle final. Le serveur est lancé localement et stocke ses données dans le dossier mlruns.
- project/ml_microservice/app.py: Un microservice FastAPI qui expose un point d'accès /predict. Au lieu de charger un fichier de modèle local, il se connecte au registre MLflow pour charger dynamiquement le modèle spécifié par un "Run ID".
- project/ml_microservice/test_app.py: Un ensemble de tests utilisant pytest pour valider le bon fonctionnement du microservice, y compris le chargement du modèle et la prédiction.

Guide d'Exécution Complet


Pour lancer le projet de A à Z, suivez ces étapes depuis la racine du dépôt.

1. Initialiser l'Environnement du Projet


Cette commande crée un environnement virtuel dédié dans project/ml_microservice/.venv et installe toutes les dépendances nécessaires (FastAPI, scikit-learn, MLflow, etc.).