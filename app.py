from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

app = Flask(__name__)
# Load pre-trained model
model = joblib.load('cognitive_fatigue_model.pk')


@app.route('/')
def home():
    return "Endurance Tracker Test"


def calculate_risk_score(features):
    try:
        heart_rate = features['heart_rate']
        screen_time = features['screen_time']
        self_reported_fatigue = features['self_reported_fatigue']

        risk_score = 0
        if heart_rate >= 90 or screen_time > 240 or self_reported_fatigue >= 4:
            risk_score = 3  # High Risk
        elif heart_rate >= 80 or screen_time > 100 or self_reported_fatigue == 3:
            risk_score = 2  # Medium Risk
        else:
            risk_score = 1
        
        return risk_score
    
    except KeyError as e:
        raise ValueError(f'Missing required key in features: {str(e)}')
    except TypeError as e:
        raise ValueError(f'Invalid type in features: {str(e)}')
    except Exception as e:
        raise RuntimeError(f'Error in calculate_risk_score: {str(e)}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input data keys
        expected_keys = ['heart_rate', 'body_temperature', 'screen_time', 
                         'activity_level', 'self_reported_fatigue', 'mood']
        for key in expected_keys:
            if key not in data:
                return jsonify({'error': f'Missing required key: {key}'}), 400
        
        # Convert input data to DataFrame
        features = pd.DataFrame([data])
        
        # Perform prediction
        prediction = model.predict(features)
        
        # Calculate risk score
        risk_score = calculate_risk_score(data)
        
        return jsonify({
            'fatigue_level': int(prediction),
            'risk_score': risk_score
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
