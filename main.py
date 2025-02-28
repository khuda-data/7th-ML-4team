import joblib
import pandas as pd


class KTAS :

    def __init__ (self) :
        self.heart_disease = joblib.load("./model/heart_disease.pkl")
        self.stroke = joblib.load("./model/stroke.pkl")
        self.alf = joblib.load("./model/alf.pkl")
        self.uti = joblib.load("./model/uti.pkl")
        self.pyelonephritis = joblib.load("./model/pyelonephritis.pkl")
        
        
    def check (self, patient : dict) :

        def get_prediction (model, data, label) :
        
            df = pd.DataFrame([data]) 
            predicted_class = model.predict(df)[0] 
            predicted_proba = model.predict_proba(df)[0][1]
            
            return {f"{label}": predicted_class, f"{label}_probability": round(predicted_proba, 2)}

        
        results = {}
        level = 5
        
        if "pyelonephritis" in patient :
            results.update(get_prediction(self.pyelonephritis, patient["pyelonephritis"], "pyelonephritis"))
            if results["pyelonephritis"] : level = 3

        if "uti" in patient :
            results.update(get_prediction(self.uti, patient["uti"], "uti"))
            if results["uti"] : level = 3
            
        if "alf" in patient :
            results.update(get_prediction(self.alf, patient["alf"], "alf"))
            if results["alf"] : level = 2

        if "stroke" in patient :
            results.update(get_prediction(self.stroke, patient["stroke"], "stroke"))
            if results["stroke"] : level = 2
            
        if "heart_disease" in patient :
            results.update(get_prediction(self.heart_disease, patient["heart_disease"], "heart_disease"))
            if results["heart_disease"] : level = 1


        return results, level


if (__name__ == "__main__") :
    
    ktas = KTAS()

    sample_data = []
    result, level = ktas.check(sample_data)

    print(result)
    
    '''
    # sample output
    {
        "heart_disease": 1,
        "heart_disease_probability": 0.8123,
        "stroke": 0,
        "stroke_probability": 0.1376,
        "alf": 1,
        "alf_probability": 0.9234,
        "uti": 0,
        "uti_probability": 0.0982,
        "pyelonephritis": 1,
        "pyelonephritis_probability": 0.7856
    }
    '''
    
    print(level)
    
    '''
    # sample output
    1
    '''
