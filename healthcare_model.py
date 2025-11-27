"""
healthcare_disease_prediction.py

Baseline pipeline:
- Load data (CSV)
- Preprocess (impute, encode, scale as needed)
- Train/test split
- Hyperparameter-tuned RandomForest classifier
- Evaluation (ROC AUC, precision/recall, confusion matrix)
- Feature importance (permutation importance)
- Save model and preprocessing pipeline

Usage:
python healthcare_disease_prediction.py --data path/to/your_data.csv --target target_column_name
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_pipeline(numeric_features, categorical_features):
    # Numeric preprocessing: impute then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: impute then onehot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')  # drop any columns not specified

    # Full pipeline with a classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
    ])

    return clf

def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Evaluation ===")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {'auc': auc, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm}

def compute_permutation_importance(pipeline, X_test, y_test, feature_names, n_repeats=10):
    # permutation_importance accepts an estimator that implements predict
    # but we need to pass the final estimator. permutation_importance will call pipeline.predict
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False)
    print("\nTop feature importances (permutation):")
    print(importances_sorted.head(20))
    return importances_sorted

def main(args):
    df = load_data(args.data)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Basic checks
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in the data.")

    # Simple approach: auto-detect numeric and categorical columns
    X = df.drop(columns=[args.target])
    y = df[args.target]
    # If target is not binary, try to convert into binary (0/1)
    if y.dtype == 'object' or y.nunique() <= 10:
        # If it's categorical strings like 'yes'/'no', map to 0/1 if possible
        if y.nunique() == 2:
            y = pd.factorize(y)[0]
        else:
            # if multi-class, you may want to handle that separately; for now we assume binary
            print("Warning: target has more than 2 unique values; this script assumes binary classification.")
            y = pd.factorize(y)[0]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Train-test split with stratification to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)

    pipeline = build_pipeline(numeric_features, categorical_features)

    # Hyperparameter grid for RandomForest
    param_grid = {
        'classifier__n_estimators': [100, 250],
        'classifier__max_depth': [None, 8, 16],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': [None, 'balanced']  # often useful for imbalanced health datasets
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=1)

    print("Starting grid search...")
    grid.fit(X_train, y_train)
    print("Grid search complete.")
    print("Best params:", grid.best_params_)
    print("Best CV ROC AUC:", grid.best_score_)

    # Evaluate on test set
    metrics = evaluate_model(grid.best_estimator_, X_test, y_test)

    # Create a list of output feature names after ColumnTransformer / OneHotEncoder
    # We need to extract feature names from the ColumnTransformer pipeline for permutation importances
    # NOTE: For sklearn >=1.0, OneHotEncoder has get_feature_names_out
    preprocessor = grid.best_estimator_.named_steps['preprocessor']
    # numeric names are the same
    numeric_out = numeric_features
    # categorical names after onehot
    cat_out = []
    if categorical_features:
        # get the transformer for categorical pipeline
        cat_pipeline = preprocessor.named_transformers_['cat']
        ohe = cat_pipeline.named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        cat_out = cat_feature_names.tolist()

    feature_names = numeric_out + cat_out

    # For permutation importance we must pass raw X (not preprocessed) because pipeline will handle preprocessing
    importances = compute_permutation_importance(grid.best_estimator_, X_test, y_test, feature_names)

    # Save the best pipeline to disk
    joblib.dump(grid.best_estimator_, args.output_model)
    print(f"Saved trained pipeline to: {args.output_model}")

    # Optionally save a small report
    report = {
        'best_params': grid.best_params_,
        'cv_best_score': grid.best_score_,
        'test_metrics': metrics,
        'top_features': importances.head(20).to_dict()
    }
    pd.Series(report).to_frame('value').to_csv(args.output_report)
    print(f"Saved report to: {args.output_report}")

if name == "main":
    parser = argparse.ArgumentParser(description="Train a disease prediction ML model")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file containing data')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column (binary label)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--output-model', type=str, default='trained_pipeline.joblib', help='Path to save trained model')
    parser.add_argument('--output-report', type=str, default='training_report.csv', help='Path to save training report')
    args = parser.parse_args()
    main(args)