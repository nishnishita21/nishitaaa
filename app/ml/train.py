import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = r"c:\nishita\hackathon_dataset (1).csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Drop ID column if it exists
    df = df.drop(columns=['customer_id'], errors='ignore')
    
    X = df.drop(columns=['high_value_purchase'])
    y = df['high_value_purchase']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Get feature names post-transformation
    try:
        ohe_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(ohe_cols)
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_prep.shape[1])]

    # Define robust models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    metrics = {}
    feature_importances = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_prep, y_train)
        y_pred = model.predict(X_test_prep)
        
        metrics[name] = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred)),
            "Recall": float(recall_score(y_test, y_pred)),
            "F1-score": float(f1_score(y_test, y_pred))
        }

        joblib.dump(model, os.path.join(base_dir, f"{name.replace(' ', '_').lower()}.joblib"))

        # Primary extraction for features from Random Forest
        if name == "Random Forest":
            importances = model.feature_importances_
            for feat, imp in zip(feature_names, importances):
                feature_importances[feat] = float(imp)
            
            # Sort by importance
            feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    joblib.dump(preprocessor, os.path.join(base_dir, "preprocessor.joblib"))

    with open(os.path.join(base_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(base_dir, "feature_importances.json"), "w") as f:
        json.dump(feature_importances, f, indent=4)
        
    print("Training complete! High-Value dataset fully ingested.")

if __name__ == "__main__":
    train_and_evaluate()
