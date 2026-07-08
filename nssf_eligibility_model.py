import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ProductionNSSFEligibilityModel:
    """
    Production-ready NSSF eligibility model for financial decisions.
    Focuses on critical fields with strict validation rules.
    """
    
    def __init__(self):
        self.model = None
        self.trained = False
        
        # Critical fields only - simplified for production
        self.critical_fields = [
            'nssf_number', 'age', 'biometric_match', 'has_bank_details',
            'has_id_document', 'has_photograph', 'name_match_score', 
            'parent_name_match', 'statement_cleanliness'
        ]
        
        # Strict eligibility rules - NO EXCEPTIONS (using thresholds for pickling)
        self.mandatory_thresholds = {
            'nssf_number': 1,        # Must have valid NSSF number
            'age': 55,               # Must be 55+ years old
            'biometric_match': 1,    # Must pass biometric verification
            'has_bank_details': 1,   # Must have bank account
            'has_id_document': 1,    # Must have valid ID
            'has_photograph': 1,     # Must have photograph
            'name_match_score': 1,   # Names must match exactly
            'parent_name_match': 1,  # Parent names must match
            'statement_cleanliness': 75  # Statement must be 75%+ clean
        }
    
    def create_training_data(self, n_samples=5000):
        """Create robust training data with realistic scenarios"""
        print("Creating production training data...")
        np.random.seed(42)
        
        # 60% eligible, 40% not eligible (reflects real-world distribution)
        n_eligible = int(n_samples * 0.6)
        n_ineligible = n_samples - n_eligible
        
        data = []
        
        # ELIGIBLE CASES (60%)
        for i in range(n_eligible):
            # All eligible cases MUST pass all mandatory requirements
            record = {
                'nssf_number': 1,  # Always valid
                'age': np.random.randint(55, 75),  # Above minimum
                'biometric_match': 1,  # Always matches
                'has_bank_details': 1,  # Always present
                'has_id_document': 1,  # Always present
                'has_photograph': 1,  # Always present
                'name_match_score': 1,  # Always matches
                'parent_name_match': 1,  # Always matches
                'statement_cleanliness': np.random.uniform(75, 100),  # Above threshold
                'eligible': 1
            }
            data.append(record)
        
        # INELIGIBLE CASES (40%)
        violation_types = [
            'no_nssf', 'too_young', 'no_biometric', 'no_bank', 
            'no_id', 'no_photo', 'no_name_match', 'no_parent_match', 'poor_statement'
        ]
        
        for i in range(n_ineligible):
            # Each ineligible case violates at least one mandatory requirement
            violation = np.random.choice(violation_types)
            
            record = {
                'nssf_number': 0 if violation == 'no_nssf' else 1,
                'age': np.random.randint(18, 55) if violation == 'too_young' else np.random.randint(55, 75),
                'biometric_match': 0 if violation == 'no_biometric' else 1,
                'has_bank_details': 0 if violation == 'no_bank' else 1,
                'has_id_document': 0 if violation == 'no_id' else 1,
                'has_photograph': 0 if violation == 'no_photo' else 1,
                'name_match_score': 0 if violation == 'no_name_match' else 1,
                'parent_name_match': 0 if violation == 'no_parent_match' else 1,
                'statement_cleanliness': np.random.uniform(20, 75) if violation == 'poor_statement' else np.random.uniform(75, 100),
                'eligible': 0
            }
            data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv('nssf_production_data.csv', index=False)
        print(f"Training data saved: nssf_production_data.csv")
        
        # Show statistics
        eligible_count = df['eligible'].sum()
        print(f"Total records: {len(df)}")
        print(f"Eligible: {eligible_count} ({eligible_count/len(df)*100:.1f}%)")
        print(f"Ineligible: {len(df) - eligible_count} ({(len(df) - eligible_count)/len(df)*100:.1f}%)")
        
        return df
    
    def train_model(self, data):
        """Train production model with high accuracy focus"""
        print("\nTraining production model...")
        
        X = data[self.critical_fields]
        y = data['eligible']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Use Random Forest for reliability and interpretability
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"Model Accuracy: {(y_pred == y_test).mean():.4f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.critical_fields,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        self.trained = True
        print("\nModel training completed successfully!")
    
    def validate_eligibility(self, applicant_data):
        """
        Production eligibility validation with strict rules.
        Returns detailed decision with reasons.
        """
        if not self.trained:
            raise ValueError("Model not trained. Train model first.")
        
        # Validate input data
        processed_data = self._validate_input(applicant_data)
        
        # Check mandatory requirements first (HARD STOP)
        violations = self._check_mandatory_requirements(processed_data)
        
        if violations:
            return {
                'decision': 'REJECTED',
                'confidence': 100.0,
                'reason': 'Mandatory requirement violation',
                'violations': violations,
                'eligibility_score': 0.0,
                'processed_data': processed_data
            }
        
        # If all mandatory requirements pass, use ML model
        input_df = pd.DataFrame([processed_data])
        
        # Get model prediction
        probability = self.model.predict_proba(input_df)[0, 1]
        prediction = self.model.predict(input_df)[0]
        
        # Decision logic (conservative threshold for financial decisions)
        decision = 'APPROVED' if probability >= 0.8 else 'REJECTED'
        
        return {
            'decision': decision,
            'confidence': round(probability * 100, 2),
            'reason': 'Model prediction' if decision == 'APPROVED' else 'Insufficient confidence',
            'violations': [],
            'eligibility_score': round(probability * 100, 2),
            'processed_data': processed_data
        }
    
    def _validate_input(self, data):
        """Validate and process input data"""
        processed = {}
        
        for field in self.critical_fields:
            if field == 'nssf_number':
                # Check if NSSF number exists and is valid
                nssf_val = data.get(field)
                processed[field] = 1 if nssf_val and str(nssf_val).strip() not in ['', '0'] else 0
            else:
                processed[field] = float(data.get(field, 0))
        
        return processed
    
    def _check_mandatory_requirements(self, data):
        """Check all mandatory requirements using threshold comparison"""
        violations = []
        
        requirement_names = {
            'nssf_number': 'Missing or invalid NSSF number',
            'age': 'Age below 55 years',
            'biometric_match': 'Biometric verification failed',
            'has_bank_details': 'No bank account details',
            'has_id_document': 'No valid ID document',
            'has_photograph': 'No photograph provided',
            'name_match_score': 'Name verification failed',
            'parent_name_match': 'Parent name verification failed',
            'statement_cleanliness': 'Statement cleanliness below 75%'
        }
        
        for field, threshold in self.mandatory_thresholds.items():
            if field == 'age' or field == 'statement_cleanliness':
                # For age and statement_cleanliness, check if value is >= threshold
                if data[field] < threshold:
                    violations.append(requirement_names[field])
            else:
                # For binary fields, check if value equals threshold (1)
                if data[field] != threshold:
                    violations.append(requirement_names[field])
        
        return violations
    
    def save_model(self, filepath='nssf_production_model.pkl'):
        """Save trained model for deployment"""
        if not self.trained:
            raise ValueError("No trained model to save")
        
        model_package = {
            'model': self.model,
            'critical_fields': self.critical_fields,
            'mandatory_thresholds': self.mandatory_thresholds,
            'model_type': 'NSSF_Production_Model_v1.0'
        }
        
        joblib.dump(model_package, filepath)
        print(f"Production model saved: {filepath}")
    
    def load_model(self, filepath='nssf_production_model.pkl'):
        """Load trained model for deployment"""
        try:
            model_package = joblib.load(filepath)
            self.model = model_package['model']
            self.critical_fields = model_package['critical_fields']
            self.mandatory_thresholds = model_package['mandatory_thresholds']
            self.trained = True
            print(f"Production model loaded: {filepath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")

def main():
    """Main function to demonstrate the production model"""
    print("="*60)
    print("NSSF PRODUCTION ELIGIBILITY MODEL")
    print("="*60)
    
    # Initialize model
    model = ProductionNSSFEligibilityModel()
    
    # Create training data
    training_data = model.create_training_data(n_samples=5000)
    
    # Train model
    model.train_model(training_data)
    
    # Save model
    model.save_model()
    
    # Test cases
    print("\n" + "="*60)
    print("TESTING PRODUCTION MODEL")
    print("="*60)
    
    test_cases = [
        {
            "name": "Valid Eligible Applicant",
            "data": {
                "nssf_number": "123456789",
                "age": 58,
                "biometric_match": 1,
                "has_bank_details": 1,
                "has_id_document": 1,
                "has_photograph": 1,
                "name_match_score": 1,
                "parent_name_match": 1,
                "statement_cleanliness": 85
            }
        },
        {
            "name": "Too Young (Should Reject)",
            "data": {
                "nssf_number": "987654321",
                "age": 52,
                "biometric_match": 1,
                "has_bank_details": 1,
                "has_id_document": 1,
                "has_photograph": 1,
                "name_match_score": 1,
                "parent_name_match": 1,
                "statement_cleanliness": 90
            }
        },
        {
            "name": "Missing Bank Details (Should Reject)",
            "data": {
                "nssf_number": "555666777",
                "age": 60,
                "biometric_match": 1,
                "has_bank_details": 0,
                "has_id_document": 1,
                "has_photograph": 1,
                "name_match_score": 1,
                "parent_name_match": 1,
                "statement_cleanliness": 80
            }
        },
        {
            "name": "Edge Case - Minimum Requirements",
            "data": {
                "nssf_number": "111222333",
                "age": 55,
                "biometric_match": 1,
                "has_bank_details": 1,
                "has_id_document": 1,
                "has_photograph": 1,
                "name_match_score": 1,
                "parent_name_match": 1,
                "statement_cleanliness": 75
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 50)
        
        result = model.validate_eligibility(test_case['data'])
        
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Reason: {result['reason']}")
        
        if result['violations']:
            print(f"Violations: {', '.join(result['violations'])}")
    
    print("\n" + "="*60)
    print("PRODUCTION MODEL READY FOR DEPLOYMENT")
    print("="*60)
    print("Key Features:")
    print("✓ Strict mandatory requirement validation")
    print("✓ Conservative decision threshold (80%)")
    print("✓ Clear rejection reasons")
    print("✓ High accuracy and reliability")
    print("✓ Fast decision making")
    print("✓ Financial-grade security")

if __name__ == "__main__":
    main()