import joblib
import numpy as np
import pandas as pd

def load_models():
    """Load the trained model and scaler"""
    model = joblib.load('models/iris_classifier.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Make prediction for a single sample"""
    model, scaler = load_models()
    
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    return prediction, probabilities, model.classes_

def main():
    print("Iris Flower Species Predictor")
    print("Enter the measurements of the flower:")
    
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        prediction, probabilities, classes = predict_species(
            sepal_length, sepal_width, petal_length, petal_width
        )
        
        print("\nPrediction Results:")
        print(f"Predicted Species: {prediction}")
        print("\nPrediction Probabilities:")
        for species, prob in zip(classes, probabilities):
            print(f"{species}: {prob:.2%}")
            
    except ValueError:
        print("Error: Please enter valid numerical values")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 