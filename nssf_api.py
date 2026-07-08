from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from datetime import datetime
from typing import Dict, Any
import numpy as np

app = Flask(__name__)
CORS(app)

# Import your actual model class - FIXED IMPORT
from nssf_eligibility_model import ProductionNSSFEligibilityModel

class NSSFEligibilityAPI:
    def __init__(self, model_path='nssf_production_model.pkl'):
        self.model_path = model_path
        self.scorer = ProductionNSSFEligibilityModel()  # FIXED CLASS NAME
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.scorer.load_model(self.model_path)
                print("Model loaded successfully")
            else:
                print(f"Model file {self.model_path} not found. Training new model...")
                self.train_new_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.train_new_model()

    def train_new_model(self):
        try:
            print("Creating training data...")
            training_data = self.scorer.create_training_data(n_samples=1000)
            print("Training model...")
            self.scorer.train_model(training_data)
            print("Saving model...")
            self.scorer.save_model(self.model_path)
            print("New model trained and saved successfully")
        except Exception as e:
            print(f"Error training new model: {str(e)}")
            raise

    def validate_claim_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Required fields that must be present
        required_fields = ['nssf_number', 'age']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate and clean data according to the model's expectations
        validated_data = {}
        
        # Handle NSSF number - keep as string for validation
        validated_data['nssf_number'] = data.get('nssf_number', '')
        
        # Handle age
        try:
            validated_data['age'] = float(data['age'])
        except (ValueError, TypeError):
            raise ValueError("Age must be a valid number")

        # Handle other fields with defaults
        defaults = {
            'biometric_match': 0,
            'has_bank_details': 0,
            'has_id_document': 0,
            'has_photograph': 0,
            'name_match_score': 0,
            'parent_name_match': 0,
            'statement_cleanliness': 0.0
        }
        
        for key, default_value in defaults.items():
            if key in data:
                try:
                    if key == 'statement_cleanliness':
                        validated_data[key] = float(data[key])
                    else:
                        validated_data[key] = 1 if data[key] else 0
                except (ValueError, TypeError):
                    validated_data[key] = default_value
            else:
                validated_data[key] = default_value

        # Validate ranges
        if validated_data['age'] < 0 or validated_data['age'] > 150:
            raise ValueError("Age must be between 0 and 150")
        
        if validated_data['statement_cleanliness'] < 0 or validated_data['statement_cleanliness'] > 100:
            validated_data['statement_cleanliness'] = max(0, min(100, validated_data['statement_cleanliness']))

        return validated_data

    def predict(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.scorer or not self.scorer.trained:
            raise ValueError("Model not loaded or trained")

        validated_data = self.validate_claim_data(claim_data)
        
        # Use the model's validate_eligibility method
        result = self.scorer.validate_eligibility(validated_data)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['model_version'] = '1.0'
        result['original_nssf_number'] = claim_data.get('nssf_number', 'Not provided')
        
        return result


print("Initializing NSSF Eligibility API...")
api_instance = NSSFEligibilityAPI()

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the NSSF Eligibility API",
        "version": "1.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "predict": "POST /predict - Single prediction", 
            "batch": "POST /predict/batch - Batch predictions",
            "info": "GET /model/info - Model information",
            "retrain": "POST /model/retrain - Retrain model"
        },
        "required_fields": ["nssf_number", "age"],
        "optional_fields": [
            "biometric_match", "has_bank_details", "has_id_document", 
            "has_photograph", "name_match_score", "parent_name_match", 
            "statement_cleanliness"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': api_instance.scorer is not None and api_instance.scorer.trained,
        'model_path': api_instance.model_path,
        'features': api_instance.scorer.critical_fields if api_instance.scorer else []
    })

@app.route('/predict', methods=['POST'])
def predict_eligibility():
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Log the incoming request for debugging
        print(f"Received prediction request: {request.json}")
        
        result = api_instance.predict(request.json)
        return jsonify({
            'success': True,
            'data': result
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'validation_error'
        }), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'error_type': 'internal_error'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_eligibility_batch():
    try:
        if not request.json or 'claims' not in request.json:
            return jsonify({'error': 'No claims data provided. Expected format: {"claims": [...]'}), 400

        claims = request.json['claims']
        if not isinstance(claims, list):
            return jsonify({'error': 'Claims must be a list'}), 400

        results = []
        successful_predictions = 0

        for i, claim in enumerate(claims):
            try:
                result = api_instance.predict(claim)
                results.append({
                    'index': i,
                    'success': True,
                    'data': result
                })
                successful_predictions += 1
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'nssf_number': claim.get('nssf_number', 'unknown')
                })

        return jsonify({
            'success': True,
            'total_claims': len(claims),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(claims) - successful_predictions,
            'results': results
        })
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    try:
        if not api_instance.scorer or not api_instance.scorer.trained:
            return jsonify({'error': 'Model not loaded'}), 400

        return jsonify({
            'model_version': '1.0',
            'model_type': 'RandomForestClassifier',
            'features': api_instance.scorer.critical_fields,
            'feature_count': len(api_instance.scorer.critical_fields),
            'mandatory_thresholds': api_instance.scorer.mandatory_thresholds,
            'timestamp': datetime.now().isoformat(),
            'model_path': api_instance.model_path,
            'critical_requirements': [
                'NSSF Number must be provided',
                'Age must be >= 55',
                'Bank details required',
                'ID document required', 
                'Photograph required',
                'Biometric match required',
                'Name match required',
                'Parent name match required',
                'Statement cleanliness >= 75%'
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    try:
        print("Retraining model...")
        api_instance.train_new_model()
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'timestamp': datetime.now().isoformat(),
            'model_path': api_instance.model_path
        })
    except Exception as e:
        print(f"Error retraining model: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to retrain model',
            'details': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample data"""
    sample_data = {
        "nssf_number": "12345678",  # Valid NSSF number
        "age": 58,
        "biometric_match": 1,
        "has_bank_details": 1,
        "has_id_document": 1,
        "has_photograph": 1,
        "name_match_score": 1,
        "parent_name_match": 1,
        "statement_cleanliness": 85.0
    }
    
    try:
        result = api_instance.predict(sample_data)
        return jsonify({
            'success': True,
            'message': 'Test prediction successful',
            'sample_data': sample_data,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'sample_data': sample_data
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting NSSF Eligibility API...")
    print("Available endpoints:")
    print("  GET  / - API documentation")
    print("  GET  /health - Health check")
    print("  GET  /test - Test with sample data")
    print("  POST /predict - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("  GET  /model/info - Model information")
    print("  POST /model/retrain - Retrain model")
    print(f"Model file: {api_instance.model_path}")
    print("\nStarting server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)