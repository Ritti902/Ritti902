# CropSage - Crop Recommendation System

A web-based application that recommends suitable crops based on soil parameters and environmental conditions using machine learning.

## Overview

CropSage is a college student project designed to help farmers make data-driven decisions about what crops to plant based on soil composition and environmental factors. The system uses a Random Forest classifier trained on a dataset of various crops and their optimal growing conditions.

## Features

- **Crop Prediction Tool**: Enter soil and environmental parameters to get personalized crop recommendations
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Comprehensive Crop Information**: View detailed information about different crops and their growing requirements
- **No JavaScript Required**: Simple, lightweight interface that works without JavaScript

## Dataset

The application uses the `Crop_recommendation.csv` dataset, which contains information about:

- Nitrogen (N), Phosphorous (P), and Potassium (K) levels in soil
- Temperature, humidity, pH, and rainfall measurements
- Suitable crop labels for different combinations of these parameters

## Tech Stack

- **Backend**: Python, Flask
- **Data Analysis**: pandas, NumPy, scikit-learn
- **Frontend**: HTML, CSS, Bootstrap

## Installation and Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   python app.py
   ```
6. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Navigate to the Crop Recommendation Tool section
2. Enter the required soil parameters and environmental conditions
3. Click "Predict Best Crop" to get recommendations
4. View the results and other suitable crops

## Project Structure

```
├── app.py                  # Main Flask application
├── Crop_recommendation.csv # Dataset
├── requirements.txt        # Python dependencies
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   └── images/             # Images and icons
└── templates/              # HTML templates
    ├── index.html          # Home page
    ├── result.html         # Results page
    └── error.html          # Error page
```

## Disclaimer

This application is designed for educational purposes. For actual agricultural decisions, please consult with local agricultural experts and conduct proper soil testing. 