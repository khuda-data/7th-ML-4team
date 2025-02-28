import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
   
pre_df = pd.read_csv ("heart.csv")
df = pre_df[["age", "sex", "cp", "trestbps", "fbs", "restecg", "thalach", "thal", "target"]].copy()

X = df.drop(columns=["target"])
y = df["target"]

xgb_model = XGBClassifier(n_estimators = 100, max_depth = 4, learning_rate = 0.1, eval_metric = "logloss", random_state = 42)
xgb_model.fit(X, y)

joblib.dump(xgb_model, "../model/heart_disease.pkl")
