import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Load the trained model
model_path = "../models/weather_model.pkl"
model = joblib.load(model_path)  # Using joblib since it's used for saving

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')  # Loads the home page

@app.route('/index')
def index():
    return render_template('index.html')  # Loads the prediction page




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        temp = float(request.form['temperature'])
        precipitation = float(request.form['precipitation'])
        cloud_cover = float(request.form['cloud_cover'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])

        # Prepare input as a NumPy array (same order as during training)
        input_features = np.array([[temp, precipitation, cloud_cover, humidity, wind_speed]])

        # Make prediction
        prediction = model.predict(input_features)[0]  # Extract first prediction

        # Redirect to result page
        return redirect(url_for('result', pred=prediction))

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/result')
def result():
    pred = request.args.get('pred', None)  # Get prediction from URL
    return render_template('index.html', prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
