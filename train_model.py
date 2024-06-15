import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


data  = pd.read_csv("fatigue_data.csv")
X = data.drop('fatigue_level', axis=1)
y = data['fatigue_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()

model.fit(X_train, y_train)

joblib.dump(model, 'cognitive_fatigue_model.pk')




