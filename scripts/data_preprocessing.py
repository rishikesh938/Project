import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/weather_dataset.csv")

# Drop 'DATE' column if it exists
if 'DATE' in df.columns:
    df = df.drop(columns=['DATE'])

# Automatically create a 'Weather_Type' column if it's missing
if 'Weather_Type' not in df.columns:
    print("⚠️ 'Weather_Type' column is missing. Generating one based on weather conditions...")

    def classify_weather(row):
        if row['PRECIPITATION (mm)'] > 0:
            return "Rainy"
        elif row['CLOUD_COVER (%)'] > 70:
            return "Cloudy"
        elif row['TEMP_MEAN (°C)'] > 30:
            return "Hot"
        else:
            return "Clear"

    df['Weather_Type'] = df.apply(classify_weather, axis=1)
    print("✅ 'Weather_Type' column generated.")

# Drop missing values
df = df.dropna()

# Split features and target
X = df.drop(columns=['Weather_Type'])  # Features
y = df['Weather_Type']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check label distribution
print("Train labels distribution:\n", y_train.value_counts())
print("Test labels distribution:\n", y_test.value_counts())

# Save preprocessed data
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("✅ Data preprocessing completed!")
