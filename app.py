from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('cognitive_fatigue_model.pk')


@app.route('/')
def home():
    return "Endurance Tracker Test"


def calculate_risk_score(features):
    heart_rate = features['heart_rate']
    screen_time = features['screen_time']
    self_reported_fatigue = features['self_reported_fatigue']

    risk_score = 0
    if heart_rate >= 90 or screen_time > 240 or self_reported_fatigue >= 4:
        risk_score = 3 #High Risk
    elif heart_rate >= 80 or screen_time > 100 or self_reported_fatigue == 3:
        risk_score = 2 #Medium Risk
    else:
        risk_score = 1
    return risk_score

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    features = pd.DataFrame([data])
    prediction = model.predict(features)
    risk_score = calculate_risk_score(data)
    return jsonify({
        'fatigue_level':int(prediction),
        'risk_score': risk_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


# {
#     "heart_rate": 75,
#     "body_temperature": 36.5,
#     "screen_time": 120,
#     "activity_level": 3,
#     "self_reported_fatigue": 2,
#     "mood": 4
# }
