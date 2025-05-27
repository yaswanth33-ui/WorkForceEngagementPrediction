import os
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np





def predict_single_employee(employee_data):
    required_features = [
        'absence_reason_code', 'absence_month', 'weekday_code',
        'season_indicator', 'commute_cost', 'commute_distance',
        'years_at_company', 'daily_workload', 'performance_target',
        'disciplinary_action', 'education_level', 'alcohol_consumption',
        'tobacco_use', 'body_weight_kg', 'body_height_cm', 'bmi_score'
    ]

    missing_features = [feat for feat in required_features if feat not in employee_data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Convert input data to DataFrame with all required features in the correct order
    data = pd.DataFrame([{feat: employee_data[feat] for feat in required_features}])
    
    # Create advanced features
    data = create_advanced_features(data)
    
    # Load the ensemble model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble_model.pkl')
    try:
        model = joblib.load(model_path)
    except:
        raise Exception("Ensemble model file not found.")
    
    # Make prediction
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    confidence = probabilities[prediction]
    
    # Map prediction to description
    class_descriptions = {
        0: "High Engagement",
        1: "Medium Engagement",
        2: "Low Engagement"
    }
    
    return {
        "predicted_class": int(prediction),
        "class_description": class_descriptions[prediction],
        "confidence_score": float(confidence)
    }


def create_advanced_features(data):
    """Create advanced feature interactions and transformations"""
    # Original numeric columns
    numeric_cols = ['commute_cost', 'commute_distance', 'daily_workload', 'bmi_score']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)  # Reduced to degree 2 to prevent overfitting
    poly_features = poly.fit_transform(data[numeric_cols])
    
    # Get feature names from PolynomialFeatures
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_features[:, len(numeric_cols):], 
                          columns=poly_feature_names[len(numeric_cols):],
                          index=data.index)
    
    # Create interaction features
    data['workload_x_cost'] = data['daily_workload'] * data['commute_cost']
    data['workload_x_distance'] = data['daily_workload'] * data['commute_distance']
    data['cost_x_distance'] = data['commute_cost'] * data['commute_distance']
    data['workload_x_bmi'] = data['daily_workload'] * data['bmi_score']
    
    # Create ratio features
    data['cost_per_distance'] = data['commute_cost'] / (data['commute_distance'] + 1e-6)
    data['workload_per_bmi'] = data['daily_workload'] / (data['bmi_score'] + 1e-6)
    
    # Create composite features
    data['workload_stress'] = data['daily_workload'] * (1 + data['bmi_score']/100)
    data['commute_stress'] = data['commute_cost'] * (1 + data['commute_distance']/100)
    
    # Create normalized features
    data['normalized_workload'] = (data['daily_workload'] - data['daily_workload'].mean()) / data['daily_workload'].std()
    data['normalized_bmi'] = (data['bmi_score'] - data['bmi_score'].mean()) / data['bmi_score'].std()
    
    # Create stress index
    data['stress_index'] = (data['workload_stress'] + data['commute_stress']) / 2
    
    # Create health impact score
    data['health_impact'] = data['bmi_score'] * (1 + data['daily_workload']/100)
    
    # Create additional composite features
    data['total_stress'] = data['workload_stress'] + data['commute_stress']
    data['health_workload_ratio'] = data['bmi_score'] / (data['daily_workload'] + 1e-6)
    data['commute_efficiency'] = data['commute_cost'] / (data['commute_distance'] + 1e-6)
    
    # Create advanced composite features
    data['stress_health_ratio'] = data['total_stress'] / (data['bmi_score'] + 1e-6)
    data['workload_efficiency'] = data['daily_workload'] / (data['commute_cost'] + 1e-6)
    data['health_stress_index'] = data['bmi_score'] * data['total_stress'] / 100
    
    # Create final composite features
    data['overall_stress'] = (data['workload_stress'] + data['commute_stress'] + data['health_stress_index']) / 3
    data['efficiency_score'] = (data['workload_efficiency'] + data['commute_efficiency']) / 2
    data['health_impact_score'] = data['health_impact'] * (1 + data['stress_health_ratio'])
    
    # Drop original numeric columns that were used for polynomial features
    data = data.drop(columns=numeric_cols)
    
    # Combine remaining original features with polynomial features
    data = pd.concat([data, poly_df], axis=1)
    
    return data