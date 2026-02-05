import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_excel("data/Training.xlsx")

# Separate features and target
X = data.drop("INJURY PERCENTAGE", axis=1)
y = data["INJURY PERCENTAGE"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_val)
print("R2 Score:", r2_score(y_val, predictions))
print("RMSE:", mean_squared_error(y_val, predictions, squared=False))

# Save model & scaler
joblib.dump(model, "injury_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training complete ✅")
