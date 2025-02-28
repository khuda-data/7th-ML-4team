import pandas as pd
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import joblib


df = pd.read_csv ("healthcare-dataset-stroke-data.csv")

df.drop("id", axis = 1, inplace = True)

df['bmi'].fillna(df['bmi'].mean(), inplace = True)
df = pd.get_dummies(df, columns = ['ever_married','gender', 'work_type', 'Residence_type', 'smoking_status'])

for col in df.select_dtypes('bool').columns : df[col] = df[col].map({True: 1, False: 0})

drop = ['gender_Female', 'gender_Male', 'gender_Other', 'work_type_Govt_job', 'work_type_Never_worked',
        'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
        'Residence_type_Urban', 'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes']

df.drop(columns = drop, axis = 1, inplace = True)

X = df.drop('stroke', axis=1)
y = df['stroke']


# Undersampling
undersample = NearMiss(version=1)
X, y = undersample.fit_resample(X, y)

smt = SMOTETomek(random_state=42)
X, y = smt.fit_resample(X, y)


# Hyperparams grid
param_grid = {
    'n_estimators' : [100, 200],
    'max_depth' : [3,5,7,10, 20],
    'min_samples_split' : [2, 5, 10],
    'class_weight' : [{0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid, cv = 3, scoring = 'f1')
grid_search.fit(X, y)

model = grid_search.best_estimator_

joblib.dump(model, "../model/stroke.pkl")
