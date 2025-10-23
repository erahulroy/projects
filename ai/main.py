import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


# Load the dataset
def load_data():
    df = pd.read_csv('/content/heart.csv')
    return df


# Preprocess the data
def preprocess_data(df):
    print("\n--- Preprocessing Data ---")

    # Handle outliers using IQR method for numeric columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n--- Training and Evaluating Models ---")

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # Parameters for grid search
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    }

    best_models = {}
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        results[name] = {
            'model': best_model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'best_params': grid_search.best_params_
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

    # Find the best model based on accuracy
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

    return best_models, results


# Function to save the best model
def save_model(model, scaler, feature_names, model_name="heart_disease_model.pkl", scaler_name="scaler.pkl"):
    print(f"\nSaving model as {model_name}...")
    joblib.dump(model, model_name)
    joblib.dump(scaler, scaler_name)
    joblib.dump(feature_names, "feature_names.pkl")
    print("Model and scaler saved successfully!")

    return model_name, scaler_name


# Function to load the model
def load_model(model_name="heart_disease_model.pkl", scaler_name="scaler.pkl"):
    print(f"\nLoading model from {model_name}...")
    model = joblib.load(model_name)
    scaler = joblib.load(scaler_name)
    feature_names = joblib.load("feature_names.pkl")
    print("Model and scaler loaded successfully!")

    return model, scaler, feature_names


# Function to predict heart disease
def predict_heart_disease(model, scaler, feature_names):
    print("\n--- Heart Disease Prediction ---")
    print("Please enter the following information:")

    feature_descriptions = {
        'age': 'Age (in years)',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
        'trestbps': 'Resting blood pressure (in mm Hg)',
        'chol': 'Serum cholesterol (in mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect, 0 = unknown)'
    }

    user_data = {}

    for feature in feature_names:
        if feature in feature_descriptions:
            try:
                user_input = input(f"Enter {feature_descriptions[feature]}: ")
                user_data[feature] = float(user_input)
            except ValueError:
                print("Invalid input. Using default value.")
                user_data[feature] = 0

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Check if any features are missing
    missing_features = set(feature_names) - set(user_df.columns)
    for feature in missing_features:
        user_df[feature] = 0

    # Ensure correct order of features
    user_df = user_df[feature_names]

    # Scale features
    user_df_scaled = scaler.transform(user_df)

    # Make prediction
    prediction = model.predict(user_df_scaled)
    prediction_proba = model.predict_proba(user_df_scaled)

    print("\n--- Prediction Results ---")
    if prediction[0] == 1:
        print("Prediction: Heart Disease Detected")
        print(f"Confidence: {prediction_proba[0][1] * 100:.2f}%")
    else:
        print("Prediction: No Heart Disease Detected")
        print(f"Confidence: {prediction_proba[0][0] * 100:.2f}%")

    return prediction, prediction_proba


# Function to get feature importance (for Random Forest)
def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

        print("\nFeature Importance:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    else:
        print("This model doesn't support feature importance.")


# Modified main function to automatically train and go to prediction
def main():
    print("=== Heart Disease Prediction System ===")

    # Load data directly with fixed path
    df = load_data()
    
    try:
        # Try to load existing model first
        print("Checking for existing model...")
        model, scaler, feature_names = load_model()
        print("Existing model found!")
    except FileNotFoundError:
        print("No existing model found. Training new models...")
        # Preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)

        # Train and evaluate models
        best_models, results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = results[best_model_name]['model']

        # Save best model
        model_file, scaler_file = save_model(best_model, scaler, feature_names)
        
        # Show feature importance for Random Forest
        if best_model_name == 'Random Forest':
            get_feature_importance(best_model, feature_names)
        
        # Reload model to ensure consistency
        model, scaler, feature_names = load_model()
    
    # Make prediction
    predict_heart_disease(model, scaler, feature_names)
    
    print("Thank you for using the Heart Disease Prediction System. Goodbye!")



main()
