import pandas as pd
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from joblib import dump

sys.path.append('project/ml_microservice')
from preprocessing import add_combined_feature

def get_data():
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    return df

def main():
    print("Loading data...")
    data = get_data()

    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Setting up training pipeline...")
    preprocessing_pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(add_combined_feature)),
        ('scaler', StandardScaler())
    ])

    training_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('classifier', LogisticRegression())
    ])

    param_grid = {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__max_iter': [1000]
    }

    print("Starting model training with GridSearchCV...")
    grid_search = GridSearchCV(training_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

    print("Saving the new, corrected model...")
    dump(best_model, 'best_cancer_model_pipeline.joblib')
    print("Model saved successfully to best_cancer_model_pipeline.joblib")

if __name__ == "__main__":
    main()
