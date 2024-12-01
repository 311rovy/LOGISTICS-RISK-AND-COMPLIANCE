from flask import Blueprint, request, jsonify, render_template
import pickle
import numpy as np
from app.config import Config
import logging
import os

pred_bp = Blueprint('prediction', __name__)
logger = logging.getLogger(__name__)

# Feature lists
CLASSIFIER_FEATURES = [
    'Client_Verification', 'Cargo_Value', 'Cargo_Weight', 'KRA_Lock',
    'KRA_Penalty', 'Incident_encoded', 'Cargo_Risk_encoded',
    'Client_Type_Business', 'Client_Type_Individual', 'Client_Location_Arusha',
    'Client_Location_Kampala', 'Client_Location_Kigali', 'Client_Location_Nairobi',
    'Cargo_Type_Alcohol', 'Cargo_Type_Clothing', 'Cargo_Type_Electronics',
    'Cargo_Type_Machinery', 'Transporter_Transporter A', 'Transporter_Transporter B',
    'Transporter_Transporter C', 'Route_Arusha-Kampala', 'Route_Arusha-Nairobi',
    'Route_Nairobi-Eldoret', 'Route_Nairobi-Kisumu', 'Border_Point_Busia',
    'Border_Point_Malaba', 'Border_Point_Namanga', 'Border_Point_Taveta',
    'Incident_Location_Border', 'Incident_Location_Transit',
    'Incident_Location_Warehouse', 'Incident_Year', 'Incident_Month',
    'Incident_Day', 'Incident_Weekday', 'Incident_Quarter', 'latitude',
    'longitude', 'Predicted_Incident_Type'
]

REGRESSOR_FEATURES = [
    'Cargo_Risk_encoded', 'Cargo_Value', 'latitude', 'Cargo_Weight', 'longitude',
    'KRA_Penalty', 'Incident_Day', 'Incident_Month', 'Incident_Weekday',
    'Incident_Quarter', 'Predicted_Incident_Type', 'Incident_encoded', 'KRA_Lock',
    'Client_Location_Kampala', 'Client_Verification', 'Route_Arusha-Kampala',
    'Cargo_Type_Electronics', 'Border_Point_Namanga', 'Transporter_Transporter B',
    'Route_Nairobi-Eldoret', 'Border_Point_Busia', 'Transporter_Transporter C',
    'Incident_Location_Transit', 'Route_Arusha-Nairobi', 'Cargo_Type_Clothing',
    'Cargo_Type_Alcohol', 'Border_Point_Malaba', 'Incident_Location_Border',
    'Route_Nairobi-Kisumu', 'Border_Point_Taveta', 'Client_Location_Kigali',
    'Client_Location_Nairobi', 'Incident_Location_Warehouse', 'Cargo_Type_Machinery',
    'Transporter_Transporter A', 'Client_Location_Arusha', 'Client_Type_Individual',
    'Client_Type_Business', 'Incident_Year'
]

def process_classifier_features(data):
    """Process input data for the classifier model."""
    features = np.zeros(len(CLASSIFIER_FEATURES))
    feature_dict = {name: i for i, name in enumerate(CLASSIFIER_FEATURES)}
    
    # Set default values first
    features[feature_dict['Client_Verification']] = 1  # Assuming verified
    features[feature_dict['KRA_Lock']] = 0  # Assuming no lock
    features[feature_dict['Incident_encoded']] = 0  # Default value
    features[feature_dict['Cargo_Risk_encoded']] = 1  # Default medium risk
    features[feature_dict['Predicted_Incident_Type']] = 0  # Default value
    
    # Process numerical variables
    if 'Cargo_Value' in data:
        features[feature_dict['Cargo_Value']] = float(data['Cargo_Value'])
    if 'Cargo_Weight' in data:
        features[feature_dict['Cargo_Weight']] = float(data['Cargo_Weight'])
    if 'KRA_Penalty' in data:
        features[feature_dict['KRA_Penalty']] = float(data['KRA_Penalty'])
    
    # Process categorical variables
    if 'Client_Type' in data:
        feature_name = f"Client_Type_{data['Client_Type']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    if 'Client_Location' in data:
        feature_name = f"Client_Location_{data['Client_Location']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    if 'Cargo_Type' in data:
        feature_name = f"Cargo_Type_{data['Cargo_Type']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    if 'Transporter' in data:
        feature_name = f"Transporter_{data['Transporter']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    if 'Route' in data:
        feature_name = f"Route_{data['Route']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    # Set current date-related features
    from datetime import datetime
    current_date = datetime.now()
    features[feature_dict['Incident_Year']] = current_date.year
    features[feature_dict['Incident_Month']] = current_date.month
    features[feature_dict['Incident_Day']] = current_date.day
    features[feature_dict['Incident_Weekday']] = current_date.weekday()
    features[feature_dict['Incident_Quarter']] = (current_date.month - 1) // 3 + 1
    
    # Set coordinates (default to Nairobi)
    features[feature_dict['latitude']] = -1.2921
    features[feature_dict['longitude']] = 36.8219
    
    # Set default location type
    features[feature_dict['Incident_Location_Transit']] = 1  # Default to transit
    
    return features.reshape(1, -1)

def process_regressor_features(data):
    """Process input data for the regressor model."""
    features = np.zeros(len(REGRESSOR_FEATURES))
    feature_dict = {name: i for i, name in enumerate(REGRESSOR_FEATURES)}
    
    # Copy the same processing logic as classifier but use REGRESSOR_FEATURES
    # Process categorical variables
    if 'Client_Type' in data:
        feature_name = f"Client_Type_{data['Client_Type']}"
        if feature_name in feature_dict:
            features[feature_dict[feature_name]] = 1
    
    # Add Predicted_Incident_Type from classifier
    if 'Predicted_Incident_Type' in data:
        features[feature_dict['Predicted_Incident_Type']] = data['Predicted_Incident_Type']
    
    
    if 'Cargo_Value' in data and 'Cargo_Value' in feature_dict:
        features[feature_dict['Cargo_Value']] = float(data['Cargo_Value'])
    
    if 'Cargo_Weight' in data and 'Cargo_Weight' in feature_dict:
        features[feature_dict['Cargo_Weight']] = float(data['Cargo_Weight'])
    
    if 'KRA_Penalty' in data and 'KRA_Penalty' in feature_dict:
        features[feature_dict['KRA_Penalty']] = float(data['KRA_Penalty'])
    
    return features.reshape(1, -1)

# Load models
try:
    with open(os.path.join(Config.MODEL_PATH, 'rf_classifier_model.pkl'), 'rb') as f:
        rf_classifier = pickle.load(f)
    with open(os.path.join(Config.MODEL_PATH, 'hybrid_rf_regressor_model.pkl'), 'rb') as f:
        hybrid_rf_regressor = pickle.load(f)
    logger.info("Models loaded successfully")
    
    # Print feature names from the model
    logger.info(f"Classifier features: {rf_classifier.feature_names_in_}")
    logger.info(f"Number of features expected: {len(rf_classifier.feature_names_in_)}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def check_auth_token(request):
    token = request.headers.get('Authorization')
    return token == f"Bearer {Config.AUTH_TOKEN}"

@pred_bp.route('/api/v1/predict', methods=['POST'])
def predict():
    if not check_auth_token(request):
        logger.warning("Unauthorized access attempt")
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json(force=True)
        
        # Validate input data
        required_fields = ['Cargo_Value', 'Cargo_Weight', 'KRA_Penalty']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

       
        classifier_features = process_classifier_features(data)
        predicted_incident_type = rf_classifier.predict(classifier_features)[0]
        
        
        data['Predicted_Incident_Type'] = predicted_incident_type
        
     
        regressor_features = process_regressor_features(data)
        predicted_loss = hybrid_rf_regressor.predict(regressor_features)[0]

        response = {
            'incident_type': int(predicted_incident_type),
            'estimated_loss': float(predicted_loss),
            'incident_type_mapping': {
                0: 'Accident',
                1: 'Theft',
                2: 'Fraud'
            }
        }
        
        logger.info(f"Successful prediction for cargo value: {data.get('Cargo_Value')}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500 


@pred_bp.route('/')
def index():
    return render_template('index.html')