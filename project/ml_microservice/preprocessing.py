import pandas as pd

def add_combined_feature(X):
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
        'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
        'area error', 'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
        'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
        'worst fractal dimension'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['Combined_radius_texture'] = X_df['mean radius'] * X_df['mean texture']
    
    return X_df.values
