import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib


df = pd.read_csv("nephritis data.csv")

df.columns = [
    "temperature", "nausea", "lumbar_pain", "urine_pushing", "micturition_pains",
    "urethra_burning_itch_swelling", "nephritis"
]

df.replace({'yes': 1, 'no': 0}, inplace = True)

X = df.drop(columns = ["nephritis"])
y = df["nephritis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

param_grid = {
    'C': [0.1, 1, 3, 10],
    'gamma': [ 0.0001,0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(random_state = 42), param_grid, cv = 5, scoring = 'accuracy', refit = True)
grid_search.fit(X_scaled, y)

best_model = grid_search.best_estimator_

joblib.dump(best_model, "../model/uti.pkl")
