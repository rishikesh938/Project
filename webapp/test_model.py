import joblib
import pandas as pd

# Load the model
model = joblib.load("C:\\Users\\user\\weather_project\\models\\weather_model.pkl")

# Define feature names
feature_names = ['TEMP_MEAN (°C)', 'PRECIPITATION (mm)', 'CLOUD_COVER (%)', 'HUMIDITY (%)', 'WIND_SPEED (km/h)']

# Create a DataFrame for input with correct column names
sample_input = pd.DataFrame([[25, 1, 23, 43, 12]], columns=feature_names)

# Predict using the model
prediction = model.predict(sample_input)

print("✅ Model loaded successfully!")
print(f"🔍 Model type: {type(model)}")
print(f"🌦️ Predicted Weather: {prediction[0]}")
