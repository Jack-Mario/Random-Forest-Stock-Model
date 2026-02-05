import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# List of stocks (same as in RF_get_data.py)
stocks = {
    "VOLV_B_ST": "Volvo Group",
    "VOLCAR_B_ST": "Volvo Cars",
    "ERIC_B_ST": "Ericsson",
    "HM_B_ST": "H&M",
    "SAND_ST": "Sandvik",
    "SKA_B_ST": "Skanska",
    "AZN_ST": "AstraZeneca",
    "ICA_ST": "ICA Gruppen"
}

def read_data(filename):
    """Read and prepare data from CSV"""
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    y = df["Increase"]
    x = df.drop(columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Increase"])
    test_size = 0.2

    split_idx = int(len(df) * (1 - test_size))
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return x_train, x_test, y_train, y_test

def train_and_evaluate(x_train, x_test, y_train, y_test, stock_name):
    """Train Random Forest and evaluate"""
    classifier = RandomForestClassifier(n_estimators=300,
                                        random_state=42,
                                        n_jobs=-1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{stock_name}:')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'  Confusion matrix:')
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    print(cm)
    
    # Feature importances
    feature_importances = classifier.feature_importances_
    header = x_test.columns
    
    plt.figure(figsize=(10, 6))
    plt.barh(header, feature_importances)
    plt.xlabel('Feature importance')
    plt.title(f'Feature Importance - {stock_name}')
    plt.tight_layout()
    plt.savefig(f"feature_importance_{stock_name}.png")
    plt.close()
    
    return classifier, y_pred

# Train model for each stock
print("=" * 60)
print("Training Random Forest models for all Swedish stocks")
print("=" * 60)

models = {}
predictions = {}

for ticker, company_name in stocks.items():
    filename = f"data_{ticker}.csv"
    try:
        print(f"\nProcessing {company_name}...")
        x_train, x_test, y_train, y_test = read_data(filename)
        model, y_pred = train_and_evaluate(x_train, x_test, y_train, y_test, company_name)
        models[ticker] = {
            'model': model,
            'x_test': x_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        predictions[ticker] = y_pred
    except Exception as e:
        print(f"  Error processing {company_name}: {e}")

print("\n" + "=" * 60)
print("All models trained successfully!")
print("=" * 60)