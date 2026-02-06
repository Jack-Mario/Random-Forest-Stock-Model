import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np


def train_save_model(csv_path: str, model_dir: str = 'models', test_size: float = 0.2, n_estimators: int = 300):
    p = Path(csv_path)
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    y = df['Increase'].astype(int)
    X = df.drop(columns=['Increase'])
    X = X.select_dtypes(include=['number'])
    
    feature_names = X.columns.tolist()

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    
    # Get probabilities for positive class (1)
    proba = clf.predict_proba(X_test)
    pos_idx = np.where(clf.classes_ == 1)[0][0]
    prob_positive = proba[:, pos_idx]

    # Save model with feature names
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_file = Path(model_dir) / (p.stem + '.joblib')
    dump({'model': clf, 'features': feature_names}, model_file)

    # Return feature importance data (aggregated later)
    feature_importances = clf.feature_importances_
    feature_names_list = feature_names

    return {
        'model_file': str(model_file), 
        'accuracy': acc, 
        'confusion_matrix': cm,
        'ticker': p.stem.replace('data_', ''),
        'feature_importances': feature_importances,
        'feature_names': feature_names
    }
