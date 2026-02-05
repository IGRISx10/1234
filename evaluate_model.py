import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# Load model & scaler
model = joblib.load("injury_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load test data
test_data = pd.read_excel("data/Testing.xlsx")

X_test = test_data.drop("INJURY PERCENTAGE", axis=1)
y_test = test_data["INJURY PERCENTAGE"]

X_test = scaler.transform(X_test)

predictions = model.predict(X_test)

print("Test R2 Score:", r2_score(y_test, predictions))
print("Test RMSE:", mean_squared_error(y_test, predictions, squared=False))
