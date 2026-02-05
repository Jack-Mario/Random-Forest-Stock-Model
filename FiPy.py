# Random Forest - demo (klassificering + regression)
# pip install scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

# ---------------------------
# 1) KLASSIFICERING
# ---------------------------
data = load_breast_cancer()
print(data)
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,       # antal träd
    max_features="sqrt",    # slumpade features per split (standardidé för RF)
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Klassificering (Breast Cancer) ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Feature importance (topp 10)
importances = clf.feature_importances_
idx = np.argsort(importances)[::-1][:10]

plt.figure()
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=60, ha="right")
plt.title("Random Forest - Top 10 feature importances (klassificering)")
plt.tight_layout()
plt.show()

# ---------------------------
# 2) REGRESSION
# ---------------------------
housing = fetch_california_housing()
Xr, yr = housing.data, housing.target
reg_feature_names = housing.feature_names

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=400,
    max_features=0.7,  # andel features per split (kan justeras)
    random_state=42,
    n_jobs=-1
)

reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)

print("\n=== Regression (California Housing) ===")
print("MAE:", round(mean_absolute_error(yr_test, yr_pred), 4))
print("R^2:", round(r2_score(yr_test, yr_pred), 4))

# Feature importance (topp 8)
r_importances = reg.feature_importances_
r_idx = np.argsort(r_importances)[::-1][:8]

plt.figure()
plt.bar(range(len(r_idx)), r_importances[r_idx])
plt.xticks(range(len(r_idx)), [reg_feature_names[i] for i in r_idx], rotation=45, ha="right")
plt.title("Random Forest - Top 8 feature importances (regression)")
plt.tight_layout()
plt.show()
