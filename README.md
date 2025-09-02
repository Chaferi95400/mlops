# MLOps – Course Project

Ce dépôt contient les diapositives, les laboratoires et un projet complet démontrant un workflow MLOps de base de bout en bout. Le projet final consiste en un microservice de Machine Learning qui sert un modèle de classification du cancer mammaire, avec un suivi des expériences via MLflow.

## Architecture Finale

Le projet est structuré pour séparer clairement l'entraînement du service de prédiction :

*   **`labs/Lab_1/train.py`**: Un script responsable de l'entraînement du modèle. Il utilise `scikit-learn` pour créer un pipeline de prétraitement et un classificateur `LogisticRegression`.
*   **`project/ml_microservice/preprocessing.py`**: Contient la logique de feature engineering personnalisée (`add_combined_feature`). Ce code est partagé entre l'entraînement et l'inférence pour garantir la cohérence.
*   **`MLflow`**: Utilisé pour suivre les expériences d'entraînement. Chaque exécution enregistre les hyperparamètres, les métriques de performance (score CV) et l'artefact du modèle final. Le serveur est lancé localement et stocke ses données dans le dossier `mlruns`.
*   **`project/ml_microservice/app.py`**: Un microservice FastAPI qui expose un point d'accès `/predict`. Au lieu de charger un fichier de modèle local, il se connecte au registre MLflow pour charger dynamiquement le modèle spécifié par un "Run ID".
*   **`project/ml_microservice/test_app.py`**: Un ensemble de tests utilisant `pytest` pour valider le bon fonctionnement du microservice, y compris le chargement du modèle et la prédiction.

## Guide d'Exécution Complet

Pour lancer le projet de A à Z, suivez ces étapes depuis la racine du dépôt.

### 1. Initialiser l'Environnement du Projet

Cette commande crée un environnement virtuel dédié dans `project/ml_microservice/.venv` et installe toutes les dépendances nécessaires (FastAPI, scikit-learn, MLflow, etc.).

```bash
make -C project/ml_microservice init
```

### 2. Entraîner et Enregistrer le Modèle

Exécutez le script d'entraînement. Cela va :
1.  Charger les données.
2.  Lancer une recherche d'hyperparamètres (`GridSearchCV`).
3.  Créer un dossier `mlruns` à la racine pour stocker les résultats de l'expérience.
4.  Enregistrer les paramètres, les métriques et le pipeline du modèle dans MLflow.

```bash
python labs/Lab_1/train.py
```

### 3. Lancer l'Interface MLflow et Récupérer le "Run ID"

Pour identifier le modèle que nous voulons servir, nous avons besoin de son ID d'exécution.

```bash
# Lance le serveur web de MLflow (Sur http://127.0.0.1:5000)
mlflow ui
```

Dans l'interface web :
1.  Cliquez sur l'expérience "Default".
2.  Cliquez sur le nom de l'exécution (ex: `unleashed-cub-377`) pour voir ses détails.
3.  Copiez la valeur du **`Run ID`** (ex: `9c68a9d825b74493a68894489cc28505`).

### 4. Configurer et Lancer le Microservice

Ouvrez le fichier `project/ml_microservice/app.py` et collez le `Run ID` que vous venez de copier dans la variable `RUN_ID`.

```python
# project/ml_microservice/app.py

# ...
RUN_ID = "9c68a9d825b74493a68894489cc28505" # <-- COLLEZ VOTRE ID ICI
MODEL_URI = f"runs:/{RUN_ID}/model"
# ...
```

Ensuite, lancez le serveur FastAPI. Il occupera ce terminal.

```bash
make -C project/ml_microservice run
```

### 5. Tester le Service

Ouvrez un **nouveau terminal**. Pour que les tests fonctionnent, Vous devez activer le bon environnement virtuel et indiquer à MLflow où trouver sa base de données.

```bash
source project/ml_microservice/.venv/bin/activate.fish
OU
source project/ml_microservice/.venv/bin/activate

# 2. Indiquer le chemin vers la base de données MLflow
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" dans ton shell

# 3. Lancer les tests
make -C project/ml_microservice test
```

Si les tests passent (`2 passed`), le workflow est complet et fonctionnel. (Ils passent je les ai tester)

## Problèmes Résolus (Key Learnings)

Au cours de ce projet, plusieurs problèmes MLOps courants ont été identifiés et résolus :

1.  **Incohérence de Données API/Modèle** : L'API attendait initialement du texte alors que le modèle attendait une liste de nombres. Le contrat de données de l'API a été corrigé pour correspondre aux attentes du modèle.
2.  **Dépendance de `joblib` au Contexte** : Le chargement du modèle échouait car la fonction de prétraitement personnalisée (`add_combined_feature`) n'était pas définie dans le contexte de l'application. La solution a été de déplacer cette fonction dans un module partagé (`preprocessing.py`) importé à la fois par le script d'entraînement et l'application.
3.  **Contexte d'Exécution de MLflow** : L'application ne trouvait pas le "Run ID" car elle ne savait pas où se trouvait le dossier `mlruns`. La solution a été de définir la variable d'environnement `MLFLOW_TRACKING_URI` pour fournir explicitement le chemin.
