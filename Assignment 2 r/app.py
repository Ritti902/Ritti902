import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, static_folder='static')

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Check if model exists, otherwise train it
model_filename = 'crop_recommendation_model.pkl'
scaler_filename = 'crop_scaler.pkl'

if not os.path.exists(model_filename):
    # Prepare data for training
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    with open(scaler_filename, 'wb') as file:
        pickle.dump(scaler, file)
else:
    # Load the model and scaler
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)

# Get unique crop labels
crop_labels = sorted(df['label'].unique())

# Get statistics for each crop
crop_stats = {}
for crop in crop_labels:
    crop_data = df[df['label'] == crop]
    crop_stats[crop] = {
        'count': len(crop_data),
        'avg_temperature': round(crop_data['temperature'].mean(), 2),
        'avg_humidity': round(crop_data['humidity'].mean(), 2),
        'avg_rainfall': round(crop_data['rainfall'].mean(), 2),
        'avg_ph': round(crop_data['ph'].mean(), 2),
        'avg_N': round(crop_data['N'].mean(), 2),
        'avg_P': round(crop_data['P'].mean(), 2),
        'avg_K': round(crop_data['K'].mean(), 2),
    }

@app.route('/')
def home():
    return render_template('index.html', crop_labels=crop_labels, crop_stats=crop_stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Create input array for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data_scaled)[0]
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_crops = [(model.classes_[i], round(probabilities[i] * 100, 2)) for i in sorted_indices[:3]]
        
        return render_template('result.html', 
                               prediction=prediction, 
                               top_crops=top_crops,
                               input_data={
                                   'N': N,
                                   'P': P,
                                   'K': K,
                                   'temperature': temperature,
                                   'humidity': humidity,
                                   'ph': ph,
                                   'rainfall': rainfall
                               },
                               crop_stats=crop_stats)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True) 