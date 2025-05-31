import pandas as pd
import joblib

# ✅ Load trained model
model = joblib.load("../models/weather_model.pkl")
print("✅ Model loaded successfully!")

# ✅ Get expected feature names from the trained model
expected_features = model.feature_names_in_
print("\n📌 Expected Feature Names:", expected_features)

# 📝 Ask for user input values
print("\n🔍 Enter weather details for prediction:")

temp = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
cloud_cover = float(input("Enter Cloud Cover (%): "))
wind_speed = float(input("Enter Wind Speed (km/h): "))
precipitation = float(input("Enter Precipitation (mm): "))

# ✅ Create DataFrame with correct column names and order
new_data = pd.DataFrame([[temp, humidity, cloud_cover, wind_speed, precipitation]], columns=expected_features)

# ✅ Display input values
print("\n📝 Input Values:")
print(new_data)

# 🔍 Make prediction
prediction = model.predict(new_data)[0]

# ✅ Display prediction result
print(f"\n🌦️ Predicted Weather: {prediction}")
