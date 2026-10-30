import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


X_train = [[80, 1500, 10, 8, 128.5, 0], [443, 200, 2, 2, 64.0, 1]]
y_train = [0, 0]  
# 2. Fit scaler and model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 3. Save model and scaler using joblib
joblib.dump(model, 'threat_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and Scaler successfully saved!")