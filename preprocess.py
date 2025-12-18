import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("diabetes.csv")

print(df.shape)
print(df["Diabetes_binary"].value_counts())


y = df["Diabetes_binary"]
X = df.drop(columns=["Diabetes_binary"])

zero_as_nan = [
    "BMI",
    "PhysHlth",
    "MentHlth",
    "GenHlth"
]

for col in zero_as_nan:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)


zero_as_nan = [
    "BMI",
    "PhysHlth",
    "MentHlth",
    "GenHlth"
]

for col in zero_as_nan:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)


X = X.apply(lambda col: col.fillna(col.median()))


binary_cols = [c for c in X.columns if X[c].nunique() == 2]
cont_cols   = [c for c in X.columns if c not in binary_cols]

print("nuls: " , X.isnull().sum().sum())


scaler = StandardScaler()
X[cont_cols] = scaler.fit_transform(X[cont_cols])


for col in cont_cols:
    q1 = X[col].quantile(0.01)
    q99 = X[col].quantile(0.99)
    X[col] = X[col].clip(q1, q99)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

df_pre = pd.concat([X, y], axis=1)
df_pre.to_csv("diabetes.csv", index=False)
