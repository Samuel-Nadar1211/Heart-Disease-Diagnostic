import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


# Data Preprocessing

df = pd.read_csv('Heart Disease data.csv')

X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
X = pd.get_dummies(X, drop_first = False, columns = ['cp', 'restecg', 'slope', 'thal']).astype(int)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

y = df['target']


# Training Model

model = RandomForestClassifier(n_estimators=100, random_state=32)
model.fit(X, y)

joblib.dump(model, 'model.pkl')
