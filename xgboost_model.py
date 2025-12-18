# src/xgboost_model.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)

from xgboost import XGBClassifier


DATA_PATH = "diabetes.csv"
OUT_DIR = "outputs"
TARGET_COL = "Diabetes_binary"



# DATA LOADING & PREPROCESS

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    feature_names = X.columns.tolist()
    return X_train_sc, X_test_sc, y_train, y_test, feature_names



# MODEL

def train_xgboost(X_train, y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model



# EVALUATION

def evaluate(model, X_test, y_test, threshold=0.35):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n=== Confusion Matrix (threshold={threshold}) ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {auc:.4f}")

    return y_proba, auc


def show_roc_curve(y_test, y_proba):
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("XGBoost ROC Curve")
    plt.tight_layout()
    plt.show()
    plt.close()


def save_feature_importance(model, feature_names, out_path, top_k=12):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    plt.figure()
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(
        range(len(idx)),
        [feature_names[i] for i in idx],
        rotation=30,
        ha="right",
    )
    plt.title(f"XGBoost Feature Importance (Top {top_k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()



# MAIN

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, target_col=TARGET_COL
    )

    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

    model = train_xgboost(X_train, y_train)
    y_proba, auc = evaluate(model, X_test, y_test, threshold=0.35)

    show_roc_curve(y_test, y_proba)

    save_feature_importance(
        model,
        feature_names,
        os.path.join(OUT_DIR, "xgboost_feature_importance.png"),
    )

    print("\nSaved:")
    print(f"- {OUT_DIR}/xgboost_feature_importance.png")


if __name__ == "__main__":
    main()
