import pandas as pd
import joblib  # For loading the saved model
from sklearn.metrics import classification_report

# ✅ Load test data
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv")
print("✅ Test data loaded successfully!")

# ✅ Load trained model
model = joblib.load("../models/weather_model.pkl")
print("✅ Model loaded successfully!")

# ✅ Ensure feature order consistency
expected_features = model.feature_names_in_  # Get the correct feature names order

# ✅ Test accuracy
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2f}\n")

# 🔍 Classification Report
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# 🆕 **Test with new unseen data**
new_data = pd.DataFrame({
    'TEMP_MEAN (°C)': [28, 15, 35, 10],
    'HUMIDITY (%)': [60, 85, 45, 90],  # Added missing humidity values
    'CLOUD_COVER (%)': [20, 80, 50, 90],
    'WIND_SPEED (km/h)': [10, 5, 15, 8],  # Added missing wind speed values
    'PRECIPITATION (mm)': [0, 0, 0, 5]
})

# ✅ Reorder columns to match the model training order
new_data = new_data[expected_features]  # Ensure the same column order as training data

# ✅ Predict weather types for new data
new_predictions = model.predict(new_data)
print("\n🔍 Predictions on new data:", new_predictions)
