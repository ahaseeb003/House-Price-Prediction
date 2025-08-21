import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def create_updated_model():
    """Create and train a model using number of rooms and house area"""
    # Load the updated dataset
    try:
        df = pd.read_csv('house_data_updated.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
    except FileNotFoundError:
        print("Error: house_data_updated.csv not found")
        return None
    
    # Prepare features and target
    X = df[['num_rooms', 'house_area_sqft']]
    y = df['price_usd']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Display model coefficients
    print(f"\nModel Coefficients:")
    print(f"Number of Rooms coefficient: {model.coef_[0]:,.2f}")
    print(f"House Area coefficient: {model.coef_[1]:,.2f}")
    print(f"Intercept: {model.intercept_:,.2f}")
    
    # Save the model
    with open('house_price_model_updated.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel trained and saved successfully!")
    return model

def predict_price_updated(num_rooms, house_area):
    """Predict house price based on number of rooms and house area"""
    try:
        with open('house_price_model_updated.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Create input array
        input_features = np.array([[num_rooms, house_area]])
        prediction = model.predict(input_features)
        return prediction[0]
    except FileNotFoundError:
        print("Model not found. Creating new model...")
        model = create_updated_model()
        if model:
            input_features = np.array([[num_rooms, house_area]])
            prediction = model.predict(input_features)
            return prediction[0]
        return None

def get_model_coefficients():
    """Get model coefficients for use in web interface"""
    try:
        with open('house_price_model_updated.pkl', 'rb') as f:
            model = pickle.load(f)
        return {
            'rooms_coef': model.coef_[0],
            'area_coef': model.coef_[1],
            'intercept': model.intercept_
        }
    except FileNotFoundError:
        # Default coefficients if model not found
        return {
            'rooms_coef': 15000,  # Approximate value per room
            'area_coef': 150,     # Approximate value per sq ft
            'intercept': 50000    # Base price
        }

if __name__ == '__main__':
    # Create and test the model
    model = create_updated_model()
    
    if model:
        # Test predictions
        test_cases = [
            (3, 1750),  # 3 rooms, 1750 sq ft
            (2, 1200),  # 2 rooms, 1200 sq ft
            (4, 2000),  # 4 rooms, 2000 sq ft
            (5, 2500),  # 5 rooms, 2500 sq ft
        ]
        
        print(f"\nTest Predictions:")
        for rooms, area in test_cases:
            predicted_price = predict_price_updated(rooms, area)
            print(f"{rooms} rooms, {area:,} sq ft: ${predicted_price:,.2f}")
        
        # Display model formula
        coeffs = get_model_coefficients()
        print(f"\nModel Formula:")
        print(f"Price = {coeffs['rooms_coef']:,.2f} × Rooms + {coeffs['area_coef']:,.2f} × Area + {coeffs['intercept']:,.2f}")

