from sklearn.model_selection import train_test_split
import pandas as pd

DATA_PATH = "diabetes.csv"
OUT_DIR = "outputs"
TARGET_COL = "Diabetes_binary"

df = pd.read_csv(DATA_PATH)

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("Accruacy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)