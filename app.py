from copyreg import pickle
from unittest import result
from flask import request, jsonify, Flask
import pandas as pd
import pickle
from imblearn import over_sampling

app = Flask(__name__)

filename = "lr_estimator.sav"
model = pickle.load(open(filename, 'rb'))

def create_feature_importance(model, sample):
    feature_importance = dict()
    result_dict = dict()
    feature_names = model.named_steps["model"].feature_names_in_
    coef_values = model.named_steps["model"].coef_[0]
    print(len(feature_names), len(coef_values), len(sample))
    for name, xvalue, coef in zip(feature_names, sample, coef_values):
        feature_importance[name] = xvalue * coef
    keys_list = sorted(feature_importance, key=lambda dict_key: abs(feature_importance[dict_key]), reverse=True)
    for key in keys_list:
        result_dict[key] = feature_importance[key]
    return list(result_dict.keys()), list(result_dict.values())

@app.route("/")
def main_page():
    return "<p>Loan Prediction API !</p>"

@app.route('/predict', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.json
        sample_values = data.values()
        sample = pd.DataFrame(data, index=[0])
        prediction = model.predict(sample)
        prediction_proba = model.predict_proba(sample)
        # print(model.named_steps['model'].coef_)
        features, importance = create_feature_importance(model, sample_values)
        # print("PREDICTION", prediction)
        response = jsonify({"class": str(prediction[0]), "proba": str(prediction_proba[0][1]), "features": features, "importance": importance})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response