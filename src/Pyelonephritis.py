import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv('Nephritis_of_renal_pelvis_origin.csv')

X = df.drop('Nephritis of renal pelvis origin', axis = 1)
y = df['Nephritis of renal pelvis origin']

categorical_features = ['Occurrence of nausea', 'Lumbar pain',
                         'Urine pushing (continuous need for urination)',
                         'Micturition pains',
                         'Burning of urethra, itch, swelling of urethra outlet']

for feature in categorical_features :
    X[feature] = X[feature].map({'yes': 1, 'no': 0}).astype(int)
    
numeric_features = ['Temperature of patient']

numeric_transformer = StandardScaler()
X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])

y = y.map({'yes': 1, 'no': 0}).astype(int)

model = LogisticRegression(random_state = 42)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

grid_search = GridSearchCV(model, param_grid, scoring = 'roc_auc', cv = cv, n_jobs = -1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

joblib.dump(best_model, "../model/pyelonephritis.pkl")
