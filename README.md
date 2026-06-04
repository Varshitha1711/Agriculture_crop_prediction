#  Agriculture Crop Prediction System

##  Overview

This project is a Machine Learning-based Crop Prediction System that helps farmers choose the most suitable crop based on environmental and soil conditions. By analyzing key parameters such as temperature, humidity, rainfall, and soil nutrients (N, P, K) and pH, the system provides accurate ranked crop recommendations with confidence scores to improve agricultural productivity.

The system combines:
- **RandomForest classifier** (500 trees, max depth 20) for robust predictions
- **Data-driven NPK estimation** using k-NN weather similarity
- **Official APY (Area-Production-Yield) time-series** from local agricultural datasets
- **Weather integration** via OpenWeather API
- **Location-aware lookups** using OpenStreetMap Nominatim geocoding

---

##  Features

* **Ranked Crop Recommendations** — Predicts top crops with probability scores based on soil and weather
* **Intelligent Feature Engineering** — 11 features (7 base + 4 engineered) capture soil-climate interactions
* **NPK Auto-Estimation** — Estimates nitrogen, phosphorus, potassium from weather using k-NN on local data
* **Weather Integration** — Fetches real-time weather by location or city name
* **Yield Time-Series** — Displays official yield trends for recommended crops in your district
* **User-Friendly Interface** — Clean web UI with input validation and error recovery
* **Responsive Design** — Works on desktop and mobile browsers
* **Geolocation Support** — Automatic location detection for weather and district matching

---

##  Technologies Used

* **Backend**: Python, FastAPI, Uvicorn
* **ML/Data**: Scikit-learn (RandomForest, StandardScaler, SimpleImputer), Pandas, NumPy
* **Frontend**: HTML5, Vanilla JavaScript (async/await), CSS3
* **Visualization**: Chart.js (for yield trends)
* **External APIs**: OpenWeather, OpenStreetMap Nominatim
* **Data Processing**: Pandas, Lxml (for HTML-table APY exports)

---

##  Dataset & Features

### Base Features (7)

All directly measured or provided by the user:
- **Nitrogen (N)** — Soil macronutrient (kg/ha)
- **Phosphorus (P)** — Soil macronutrient (kg/ha)
- **Potassium (K)** — Soil macronutrient (kg/ha)
- **Temperature** — Air temperature (°C)
- **Humidity** — Relative humidity (%)
- **pH** — Soil acidity/alkalinity (4–9 scale)
- **Rainfall** — Water supply (mm)

### Engineered Features (4)

Derived from base features to help the model learn interactions:
- **npk_total** — N + P + K (overall nutrient load / fertilizer index)
- **n_to_pkratio** — N / (P + K + ε) (nutrient balance between nitrogen and phosphate)
- **temp_humidity** — (temperature × humidity) / 100 (combined climate stress indicator)
- **rainfall_humidity** — (rainfall × humidity) / 100 (combined wetness from rain and air moisture)

For full justification, see `ENGINEERED_FEATURE_JUSTIFICATION` in [features.py](features.py).

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anusreereddysama/Agriculture_Crop_prediction.git
cd Agriculture_Crop_prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional but Recommended)

Create a `.env` file in the project root:

```env
OPENWEATHER_API_KEY=your_api_key_here
# Optional: APY_CSV_PATH=path/to/your/apy.csv
```

To get an OpenWeather API key:
1. Sign up at https://openweathermap.org/api
2. Copy your API key
3. Add to `.env` file

### 5. Train the Model (First Time Only)

```bash
python train_model.py
```

This generates:
- `model.pkl` — Trained RandomForest
- `scaler.pkl` — StandardScaler for feature normalization
- `imputer.pkl` — SimpleImputer for missing values
- `feature_columns.json` — Feature names (for validation)

### 6. Run the Application

```bash
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

---

##  API Endpoints

All endpoints are JSON-based. Base URL: `http://localhost:8000`

### **POST /predict**
Predict top crops for given soil and weather conditions.

**Request:**
```json
{
  "N": 50,
  "P": 35,
  "K": 60,
  "temperature": 25.5,
  "humidity": 75,
  "ph": 7.2,
  "rainfall": 200
}
```

**Parameters:**
- `N` (0–500): Nitrogen in soil (kg/ha)
- `P` (0–200): Phosphorus in soil (kg/ha)
- `K` (0–400): Potassium in soil (kg/ha)
- `temperature` (-20–50): Air temperature (°C)
- `humidity` (0–100): Relative humidity (%)
- `ph` (4–9): Soil pH value
- `rainfall` (0–3000): Rainfall (mm)
- `top_k` (optional query param): Limit number of crops (default: all)

**Response:**
```json
{
  "rankings": [
    {
      "rank": 1,
      "crop": "rice",
      "probability": 92.5
    },
    {
      "rank": 2,
      "crop": "maize",
      "probability": 78.3
    }
  ]
}
```

### **GET /api/health**
Check system status.

**Response:**
```json
{
  "model_features": 11,
  "apy_csv_configured": true,
  "apy_note": null,
  "npk_estimator_ready": true,
  "npk_estimator_note": null
}
```

### **GET /api/weather**
Fetch weather by latitude/longitude.

**Query Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude

**Response:**
```json
{
  "temperature_c": 28.5,
  "humidity_pct": 72,
  "rainfall_mm": 0.5,
  "rainfall_note": "Rain in the last 1 hour (mm). Use manual override for seasonal/annual need.",
  "location_name": "Bengaluru",
  "country": "IN"
}
```

### **GET /api/weather/city**
Fetch weather by city name.

**Query Parameters:**
- `city` (string): City name (e.g., "Bengaluru", "Delhi")

**Response:** Same as above

### **GET /api/estimate-npk**
Estimate N, P, K from weather using k-NN on local training data.

**Query Parameters:**
- `temperature` (float): Air temperature (°C)
- `humidity` (float): Relative humidity (%)
- `rainfall` (float): Rainfall (mm)

**Response:**
```json
{
  "estimated_npk": {
    "N": 45.5,
    "P": 32.1,
    "K": 58.7
  },
  "confidence": 0.742,
  "neighbors_used": 40,
  "note": "Estimated from weather similarity using local training data. Use soil-test values when available."
}
```

### **GET /api/yield-series**
Fetch official yield time-series for a crop in a state/district.

**Query Parameters:**
- `crop` (string): Crop model label (e.g., "rice", "cotton")
- `state` (string): State name
- `district` (string): District name
- **OR** `lat` (float) + `lon` (float): Latitude/longitude (auto reverse-geocoded)

**Response:**
```json
{
  "configured": true,
  "series": [
    {"year": 2018, "yield": 5234.5},
    {"year": 2019, "yield": 5456.2},
    {"year": 2020, "yield": 5123.8}
  ],
  "note": "Values come only from your local APY CSV.",
  "location": {
    "state": "Karnataka",
    "district": "Bangalore Rural",
    "display_name": "Bengaluru, Karnataka, India"
  }
}
```

### **GET /api/geocode/reverse**
Reverse geocode latitude/longitude to state/district.

**Query Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude

**Response:**
```json
{
  "display_name": "Whitefield, Bengaluru, Karnataka, India",
  "state": "Karnataka",
  "district": "Bangalore Urban",
  "country_code": "IN"
}
```

---

##  How It Works

1. **Input Collection** — User fills soil (N, P, K, pH) and weather (temp, humidity, rainfall) on the Predict page
2. **Feature Engineering** — Raw inputs transformed into 11-dimensional feature vector with engineered features
3. **Preprocessing** — Features normalized with StandardScaler and missing values handled (rare)
4. **Model Prediction** — RandomForest classifier outputs probability distribution over all trained crops
5. **Results Display** — Top crops ranked by probability, displayed with links to crop detail pages
6. **Yield Lookup** — Optional: fetch official yield trends for recommended crops via APY data or manual district entry
7. **Guidance** — Static crop profiles (irrigation, water management) displayed for context

---

##  Supported Crops

The system is trained on the following crop labels:
- Rice
- Cotton
- Maize
- Mungbean (Green Gram)
- Blackgram (Urad)
- Pigeonpeas (Arhar/Tur)

To add more crops, retrain `train_model.py` with expanded `cleaned_crop_data.csv`.

---

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| **"OPENWEATHER_API_KEY not set"** | Create `.env` file and add your OpenWeather API key (free tier available) |
| **"Geolocation not supported"** | Use a modern browser (Chrome, Firefox, Safari, Edge); enable location permission |
| **"APY file not loaded"** | Ensure `data/apy/Rabi_prod.xls` and `Kharif_prod.xls` exist, or place a `district_apy.csv` (see data/apy/README.txt) |
| **"No yield points found"** | Check district name spelling; use the suggested districts from the APY hints |
| **Invalid input values** | HTML inputs enforce bounds (e.g., N: 0–500, humidity: 0–100). Adjust values to fit ranges |
| **Model.pkl not found** | Run `python train_model.py` first to train and save the model |

---

##  Configuration Files

- **`requirements.txt`** — Python dependencies (FastAPI, scikit-learn, pandas, etc.)
- **`.env`** (optional) — API keys and paths (not tracked by git)
- **`.env.example`** — Template for `.env`
- **`feature_columns.json`** — List of 11 feature names (auto-generated by train_model.py)

---

##  Directory Structure

```
Agriculture_Crop_prediction/
├── main.py                          # FastAPI server & endpoints
├── train_model.py                   # Model training pipeline
├── features.py                      # Feature engineering logic
├── feature_analysis.py              # EDA: correlations, feature importance
├── analysis.py                      # Data visualization & exploration
├── apy_loader.py                    # APY (yield) data loader
├── geocode.py                       # OSM Nominatim reverse geocoding
├── india_geo_context.py             # Geographic metadata
│
├── cleaned_crop_data.csv            # Training dataset (2200 rows, 7 features + label)
├── Crop_recommendation.csv          # Secondary dataset for EDA
├── feature_columns.json             # 11 feature names (generated)
├── model.pkl, scaler.pkl, imputer.pkl  # Trained ML artifacts
│
├── static/                          # Frontend files
│   ├── index.html                   # Landing page
│   ├── predict.html                 # Input form
│   ├── predict.js                   # Form logic & weather fetching
│   ├── recommendations.html         # Results display
│   ├── recommendations.js           # Crop rankings render
│   ├── crop-detail.html             # Crop profile & yield chart
│   ├── crop-detail.js               # Yield chart logic
│   ├── common.js                    # Shared utilities
│   ├── style.css                    # Styling
│   └── data/crops.json              # Static crop metadata
│
├── data/apy/                        # Official yield data (optional)
│   ├── Rabi_prod.xls                # Winter season yield
│   ├── Kharif_prod.xls              # Monsoon season yield
│   └── README.txt                   # APY setup instructions
│
├── requirements.txt
├── README.md                        # This file
├── .env.example                     # Environment template
└── .gitignore
```

---

##  Model Details

- **Algorithm**: RandomForestClassifier (500 trees, max_depth=20)
- **Training Data**: 2,200 samples across 6 crop classes
- **Features**: 11 (7 base + 4 engineered)
- **Preprocessing**: SimpleImputer (mean strategy) → StandardScaler
- **Output**: Probability distribution over crop classes
- **Random State**: 42 (for reproducibility)

---

##  Future Enhancements

* Deep learning models (CNN/LSTM) for temporal patterns
* Multi-crop rotation recommendations
* Mobile app (React Native / Flutter)
* Multi-language UI localization
* Advanced weather forecasting integration
* Farmer profiling and personalized recommendations
* Integration with agricultural extension services
* Export recommendations as PDF/SMS

---

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

##  License

This project is open source and available under the MIT License.

---

##  Acknowledgments

- Inspired by real-world agricultural challenges
- Thanks to open-source communities (scikit-learn, FastAPI, Chart.js)
- APY data sourced from official Indian agricultural statistics
- Built to support sustainable farming practices

---

##  Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com (optional)

---

**Happy Farming! 🌾**
