import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib


df = pd.read_excel("ALF_Data.xlsx", sheet_name="Sheet1")

drop = [
    "Region", "Good Cholesterol", "Bad Cholesterol", "Total Cholesterol", 
    "Dyslipidemia", "Source of Care", "Chronic Fatigue"
]

df = df.drop(columns = drop)
df = df.dropna()

object_columns = df.select_dtypes(include = ['object']).columns
df = pd.get_dummies(df, columns = object_columns, drop_first = True)

X = df.drop(columns = ["ALF"]) 
y = df["ALF"] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

xgb_model = XGBClassifier(
    n_estimators = 200,         
    learning_rate = 0.05,       
    max_depth = 4,              
    subsample = 0.9,            
    colsample_bytree = 0.9,     
    scale_pos_weight = 1.9,  
    eval_metric = "logloss",
    random_state = 42
)

xgb_model.fit(X, y)

joblib.dump(xgb_model, "../model/alf.pkl")
