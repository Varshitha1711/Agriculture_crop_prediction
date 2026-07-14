
A machine learning-based crop recommendation system that predicts the most suitable crops using soil nutrients, weather conditions, and historical agricultural data. The application combines a Random Forest classifier, weather integration, NPK estimation, and yield analytics to support data-driven farming decisions.

## Features

* Crop recommendations based on N, P, K, temperature, humidity, pH, and rainfall
* Ranked predictions with confidence scores
* Automatic NPK estimation using weather similarity analysis
* Real-time weather integration through OpenWeather API
* District-level yield trend visualization
* Geolocation support using OpenStreetMap Nominatim
* Responsive web interface built with FastAPI and JavaScript

## Tech Stack

**Backend**

* Python
* FastAPI
* Uvicorn

**Machine Learning**

* Scikit-learn
* Random Forest Classifier
* StandardScaler
* SimpleImputer

**Frontend**

* HTML
* CSS
* JavaScript

**Data & Visualization**

* Pandas
* NumPy
* Chart.js

## Model Overview

The model uses:

* RandomForestClassifier (500 trees, max depth 20)
* 7 base features:

  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * Temperature
  * Humidity
  * pH
  * Rainfall
* 4 engineered features:

  * Total NPK
  * N-to-PK ratio
  * Temperature-Humidity interaction
  * Rainfall-Humidity interaction

The model outputs probability scores for each crop and returns ranked recommendations.

## Installation

Clone the repository:

```bash
git clone https://github.com/Varshitha1711/Agriculture_crop_prediction/

cd Agriculture_Crop_prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
OPENWEATHER_API_KEY=your_api_key
```

## Train the Model

```bash
python train_model.py
```

This generates:

```text
model.pkl
scaler.pkl
imputer.pkl
feature_columns.json
```

## Run the Application

```bash
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000
```

## API Endpoints

| Endpoint                   | Description                      |
| -------------------------- | -------------------------------- |
| POST `/predict`            | Generate crop recommendations    |
| GET `/api/weather`         | Retrieve weather by coordinates  |
| GET `/api/weather/city`    | Retrieve weather by city         |
| GET `/api/estimate-npk`    | Estimate NPK values from weather |
| GET `/api/yield-series`    | Fetch crop yield trends          |
| GET `/api/geocode/reverse` | Reverse geocode coordinates      |
| GET `/api/health`          | System health check              |

## Project Structure

```text
Agriculture_Crop_prediction/
│
├── main.py
├── train_model.py
├── features.py
├── apy_loader.py
├── geocode.py
│
├── static/
│   ├── predict.html
│   ├── predict.js
│   ├── recommendations.html
│   ├── crop-detail.html
│   └── style.css
│
├── data/
├── requirements.txt
├── README.md
└── .env.example
```

## Future Improvements

* Support additional crop classes
* Integrate weather forecasting data
* Add fertilizer recommendations
* Develop a mobile application
* Support multiple languages
