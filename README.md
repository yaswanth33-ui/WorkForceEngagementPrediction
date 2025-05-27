# Workforce Engagement Prediction

A machine learning-based web application that predicts employee engagement levels using various factors such as work patterns, health metrics, and performance indicators.

## Overview

This project implements a predictive analytics system for workforce engagement using multiple machine learning models. It provides a web interface where organizations can input employee data and receive engagement predictions, helping them identify potential engagement issues and take proactive measures.

## Features

- **Multiple ML Models**
  - Random Forest
  - XGBoost
  - LightGBM
  - Ensemble Model (combining all models)

- **Web Interface**
  - User-friendly dashboard
  - Real-time predictions
  - Input validation
  - Example data structure

- **Prediction Factors**
  - Absence patterns
  - Work schedule
  - Commute information
  - Performance metrics
  - Health indicators
  - Education level
  - Disciplinary history

## Project Structure

```
WorkforceEngagementPrediction/
├── server/                 # Web application server
│   ├── app.py             # Flask application
│   ├── utils.py           # Utility functions
│   ├── static/            # Static assets
│   └── templates/         # HTML templates
├── models/                # Trained ML models
│   ├── Random_Forest_model.pkl
│   ├── XGBoost_model.pkl
│   ├── LightGBM_model.pkl
│   └── ensemble_model.pkl
├── dataset/               # Data files
│   └── Workforce_Engagement.csv
└── README.md             # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yaswanth33-ui/WorkForceEngagementPrediction.git
cd WorkforceEngagementPrediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd server
python app.py
```

The application will be available at `http://localhost:5000`

## API Usage

### Prediction Endpoint

**URL**: `/api/verify_engagement`
**Method**: `POST`
**Content-Type**: `application/json`

#### Request Body Example:
```json
{
    "absence_reason_code": 4,
    "absence_month": 1,
    "weekday_code": 1,
    "season_indicator": 1,
    "commute_cost": 50.0,
    "commute_distance": 20.0,
    "years_at_company": 5.0,
    "daily_workload": 10.0,
    "performance_target": 85.0,
    "disciplinary_action": 1,
    "education_level": 2,
    "alcohol_consumption": 0,
    "tobacco_use": 0,
    "body_weight_kg": 70.0,
    "body_height_cm": 175.0,
    "bmi_score": 25.0
}
```

#### Response:
```json
{
    "predicted_class": 2,
    "class_description": "High Engagement",
    "confidence_score": 0.92
}
```

The response includes:
- `predicted_class`: Integer indicating the engagement level (0: Low, 1:Medium, 2: High)
- `class_description`: Text description of the engagement level
- `confidence_score`: Confidence score of the prediction (0-1)

## Input Parameters

- `absence_reason_code`: Code indicating reason for absence
- `absence_month`: Month of absence (1-12)
- `weekday_code`: Day of week (1-7)
- `season_indicator`: Season indicator (1-4)
- `commute_cost`: Daily commute cost
- `commute_distance`: Distance to workplace
- `years_at_company`: Years of service
- `daily_workload`: Daily workload units
- `performance_target`: Performance target percentage
- `disciplinary_action`: Disciplinary action indicator
- `education_level`: Education level code
- `alcohol_consumption`: Alcohol consumption indicator
- `tobacco_use`: Tobacco use indicator
- `body_weight_kg`: Weight in kilograms
- `body_height_cm`: Height in centimeters
- `bmi_score`: Body Mass Index

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 