import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def create_advanced_features(data):
    
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

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Remove unnecessary columns and unnamed columns
    columns_to_drop = ['employee_id', 'employee_age', 'dependents', 'pet_count']
    unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
    columns_to_drop.extend(unnamed_cols)
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Create advanced features
    data = create_advanced_features(data)
    
    # Scale features
    scaler = StandardScaler()
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Prepare target variable with four classes
    # Class 0: Very Low absence (0-4 hours)
    # Class 1: Low absence (4-8 hours)
    # Class 2: Medium absence (8-16 hours)
    # Class 3: High absence (>16 hours)
    bins = [0, 4, 8, 16, np.inf]
    labels = [0, 1, 2, 3]  # 0: Very Low, 1: Low, 2: Medium, 3: High
    data['absence_class'] = pd.cut(data['absence_duration_hours'], bins=bins, labels=labels)
    
    # Print class distribution
    print("\nClass distribution:")
    print(data['absence_class'].value_counts().sort_index())
    print("\nClass boundaries:")
    print("Class 0 (Very Low): 0-4 hours")
    print("Class 1 (Low): 4-8 hours")
    print("Class 2 (Medium): 8-16 hours")
    print("Class 3 (High): >16 hours")
    
    # Fill any NaN values in the target variable with the most frequent class
    most_frequent_class = data['absence_class'].mode()[0]
    data['absence_class'] = data['absence_class'].fillna(most_frequent_class)
    
    return data

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Convert data to numpy arrays for XGBoost
    if isinstance(model, XGBClassifier):
        X_train = X_train.values
        X_test = X_test.values
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Print metrics
    print(f"\n{model_name} Results:")
    print("Training accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return model

def main():
    data_path= os.path.join(os.path.dirname(__file__),'..','dataset','Workforce_Engagement.csv')
    # Load and preprocess data
    data = load_and_preprocess_data(data_path)
    
    # Prepare features and target
    X = data.drop(columns=['absence_duration_hours', 'absence_class'])
    y = data['absence_class'].astype(int)
    
    # Handle any NaN values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    class_dist = pd.Series(y).value_counts().sort_index()
    print(class_dist)
    print("\nInitial class percentages:")
    print((class_dist / len(y) * 100).round(2))
    
    # Calculate minimum class size
    min_class_size = min(class_dist)
    k_neighbors = min(5, min_class_size - 1)  # Ensure k_neighbors is less than minimum class size
    
    # Apply SMOTE to balance the entire dataset
    print("\nBalancing the entire dataset...")
    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy='auto'  # Will oversample all minority classes
    )
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Print balanced class distribution
    print("\nBalanced class distribution:")
    balanced_dist = pd.Series(y_balanced).value_counts().sort_index()
    print(balanced_dist)
    print("\nBalanced class percentages:")
    print((balanced_dist / len(y_balanced) * 100).round(2))
    
    # Train-Test Split with Stratification on balanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
    )
    
    # Define models with optimized parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=1000,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.001,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42
        )
    }
    
    # Train and evaluate each model
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_and_evaluate_model(
            model, 
            X_train, 
            y_train, 
            X_test, 
            y_test,
            name
        )
    
    # Cross-validation with stratification
    print("\nCross-validation Results:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in trained_models.items():
        scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='f1_weighted')
        print(f"{name} CV scores: {scores}")
        print(f"{name} Average CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Feature Importance Analysis
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title(f'Feature Importances - {name}')
            plt.bar(range(X.shape[1]), importances[indices])
            plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
            plt.tight_layout()
            plt.show()
    
    # Ensemble Method with optimized weights
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft',
        weights=[1.2, 1.0, 1.1]  
    )
    
    ensemble_model = train_and_evaluate_model(
        ensemble,
        X_train,
        y_train,
        X_test,
        y_test,
        'Ensemble Model'
    )
    
    print("\nFinal Model Comparison:")
    for name, model in trained_models.items():
        print(f"\n{name}:")
        print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
        print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    
    print("\nEnsemble Model:")
    print(f"Training accuracy: {ensemble_model.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {ensemble_model.score(X_test, y_test):.3f}")
    
    joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
    joblib.dump(trained_models['XGBoost'], 'models/XGBoost_model.pkl')
    joblib.dump(trained_models['LightGBM'], 'models/LightGBM_model.pkl')
    joblib.dump(trained_models['Random Forest'], 'models/Random_Forest_model.pkl')

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
    
    data = pd.DataFrame([{feat: employee_data[feat] for feat in required_features}])

    data = create_advanced_features(data)
    
    model_path = os.path.join(os.path.dirname(__file__),'..','models','XGBoost_model.pkl')
    try:
        model = joblib.load(model_path)
    except:
        raise Exception("Model file not found. Please train the model first by running the main() function.")
    

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    confidence = probabilities[prediction]
    
    class_descriptions = {
        0: "Very Low absence (0-4 hours)",
        1: "Low absence (4-8 hours)",
        2: "Medium absence (8-16 hours)",
        3: "High absence (>16 hours)"
    }
    
    return {
        "predicted_class": int(prediction),
        "class_description": class_descriptions[prediction],
        "confidence_score": float(confidence)
    }






if __name__ == "__main__":

    main() 
    employee_data = {
    'absence_reason_code': 4,
    'absence_month': 1,
    'weekday_code': 1,
    'season_indicator': 1,
    'commute_cost': 50.0,
    'commute_distance': 20.0,
    'years_at_company': 5.0,
    'daily_workload': 10.0,
    'performance_target': 85.0,
    'disciplinary_action': 1,
    'education_level': 2,
    'alcohol_consumption': 0,
    'tobacco_use': 0,
    'body_weight_kg': 70.0,
    'body_height_cm': 175.0,
    'bmi_score': 25.0
    }

    result = predict_single_employee(employee_data)
    print(result)