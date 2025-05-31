import pandas as pd

# Load dataset before preprocessing
df = pd.read_csv("../data/weather_dataset.csv")

# Print class distribution
print("Original Dataset Class Distribution:\n", df['Weather_Type'].value_counts())
