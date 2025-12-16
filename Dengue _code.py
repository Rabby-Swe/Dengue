# ===== 0) SETUP =====
!pip -q install xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
)

from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===== 1) LOAD DATA =====
CSV_PATH = "/content/public_health_surveillance_dataset.csv"
df = pd.read_csv(CSV_PATH)

# ===== 2) MISSING VALUE FILL (PAD + BFILL) =====
df = df.fillna(method="pad").fillna(method="bfill")

# ===== 3) BINARY TARGET (Severe=1, Non-Severe=0) =====
TARGET = "Disease_Severity"
y = df[TARGET].astype(str).str.strip().str.lower().apply(lambda v: 1 if v == "severe" else 0)
X = df.drop(columns=[TARGET])

# ===== 4) DROP DATE/ID-LIKE COLUMNS =====
drop_cols = [
    c for c in X.columns
    if any(k in c.lower() for k in ["date", "time", "timestamp"])
    or c.lower() in ["id", "patient_id", "record_id"]
    or c.lower().endswith("_id")
]
X = X.drop(columns=drop_cols, errors="ignore")

# ===== 5) PREPROCESSOR (IMPUTE + ONEHOT) =====
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# ===== 6) TRAIN/TEST SPLIT (STRATIFIED) =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ===== 7) XGBOOST MODEL (IMBALANCE HANDLING) =====
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    gamma=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

clf = Pipeline([("prep", preprocessor), ("model", model)])

# ===== 8) TRAIN =====
clf.fit(X_train, y_train)

# ===== 9) PREDICT =====
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

# ===== 10) ACCURACY + PRECISION/RECALL REPORT =====
print(" Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred, target_names=["Non-Severe", "Severe"]))

# ===== 11) CONFUSION MATRIX (PRINT + GUI) =====
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Non-Severe", "Severe"])
plt.title("Confusion Matrix")
plt.show()



# ===== 13) FEATURE IMPORTANCE MAP (TOP 20) =====
prep = clf.named_steps["prep"]
xgb = clf.named_steps["model"]

num_features = prep.transformers_[0][2]
ohe = prep.transformers_[1][1].named_steps["oh"]
cat_features = prep.transformers_[1][2]
cat_feature_names = ohe.get_feature_names_out(cat_features)

all_features = np.concatenate([num_features, cat_feature_names])
importances = xgb.feature_importances_

N = 20
idx = np.argsort(importances)[::-1][:N]
top_features = all_features[idx]
top_importances = importances[idx]

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

