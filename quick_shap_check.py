import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from fina import compute_shap_importance

# Generate a small synthetic dataset
X, y = make_classification(n_samples=120, n_features=20, n_informative=5,
                           n_redundant=2, random_state=42)
feature_names = [f'Gene_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42, stratify=y
)

# Build and fit a simple pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
])
pipe.fit(X_train, y_train)

# Run SHAP importance with conservative limits
results = compute_shap_importance(
    pipe, X_train, X_test,
    feature_names=feature_names,
    max_samples=min(50, len(X_test)),
    max_features=min(15, X_train.shape[1])
)

if results is None:
    print("SHAP check failed: results is None")
else:
    imp = results['importance_df']
    print("SHAP check completed. Top 5 features:")
    print(imp.head(5))