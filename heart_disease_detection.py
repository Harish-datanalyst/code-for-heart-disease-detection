import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_prepare_data(file_path):
    """Load and prepare the heart disease dataset"""
    # Read CSV with proper encoding to handle special characters
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Clean column names - remove spaces and make lowercase
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    
    # Remove any missing values
    df = df.dropna()
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    
    results = {}
    
    # Model 1: Random Forest
    print("\n=== RANDOM FOREST CLASSIFIER ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importances:")
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    results['Random Forest'] = rf_accuracy
    
    # Model 2: Logistic Regression (with scaling)
    print("\n=== LOGISTIC REGRESSION ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    print(f"Accuracy: {lr_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_pred))
    
    results['Logistic Regression'] = lr_accuracy
    
    # Model 3: Support Vector Machine
    print("\n=== SUPPORT VECTOR MACHINE ===")
    svm = SVC(random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    print(f"Accuracy: {svm_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, svm_pred))
    
    results['SVM'] = svm_accuracy
    
    # Model comparison
    print("\n=== MODEL COMPARISON ===")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]:.4f}")
    
    # Confusion Matrix for Random Forest (usually the best)
    print("\n=== CONFUSION MATRIX (Random Forest) ===")
    cm = confusion_matrix(y_test, rf_pred)
    print("Confusion Matrix:")
    print(f"True Negatives: {cm[0,0]} (Correctly predicted no heart disease)")
    print(f"False Positives: {cm[0,1]} (Incorrectly predicted heart disease)")
    print(f"False Negatives: {cm[1,0]} (Missed heart disease cases)")
    print(f"True Positives: {cm[1,1]} (Correctly predicted heart disease)")
    
    return rf, scaler

def predict_heart_disease(model, scaler, patient_data):
    """Predict heart disease for new patient data"""
    # If patient_data is a single row, reshape it
    if len(patient_data.shape) == 1:
        patient_data = patient_data.reshape(1, -1)
    
    prediction = model.predict(patient_data)
    probability = model.predict_proba(patient_data)
    
    return prediction, probability

def main():
    """Main function to run the heart disease prediction system"""
    
    print("=== HEART DISEASE PREDICTION SYSTEM ===\n")
    
    # Step 1: Load and prepare data
    # IMPORTANT: Change this path to where your dataset.csv file is located
    file_path = 'D:\\intern projects uni\\intern projects\\dataset.csv'  # Update this path!
    
    try:
        df = load_and_prepare_data(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Please check the file path and make sure the file exists.")
        return
    except UnicodeDecodeError:
        print("Error: Could not read the file. Try changing encoding to 'utf-8' or 'latin1'")
        return
    
    # Step 2: Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Step 4: Train and evaluate models
    best_model, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Example prediction for a new patient
    print("\n=== EXAMPLE PREDICTION ===")
    # Example patient data (you can change these values)
    example_patient = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 3]])
    
    prediction, probability = predict_heart_disease(best_model, scaler, example_patient)
    
    print(f"Example patient prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
    print(f"Confidence: {probability[0][prediction[0]]:.4f}")
    
    print("\n=== SYSTEM READY FOR PREDICTIONS ===")

if __name__ == "__main__":
    main()

# USAGE INSTRUCTIONS:
# 1. Save this code as 'heart_disease_detection.py'
# 2. Make sure your dataset.csv file is in the same folder
# 3. Run: python heart_disease_detection.py

# TROUBLESHOOTING:
# - If you get FileNotFoundError: Update the file_path variable with the correct path
# - If you get UnicodeDecodeError: The file encoding might be different
# - Try adding encoding='latin1' or encoding='utf-8' to pd.read_csv()

# DATASET FEATURES:
# age: Age in years
# sex: 0 = female, 1 = male  
# chest_pain_type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
# resting_bp_s: Resting blood pressure in mm Hg
# cholesterol: Serum cholesterol in mg/dl
# fasting_blood_sugar: 1 = sugar > 120mg/dL, 0 = sugar < 120mg/dL
# resting_ecg: 0 = normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy
# max_heart_rate: Maximum heart rate achieved (71–202)
# exercise_angina: 0 = no, 1 = yes
# oldpeak: ST depression induced by exercise
# st_slope: 1 = upward, 2 = flat, 3 = downward
# target: 0 = Normal, 1 = Heart Disease