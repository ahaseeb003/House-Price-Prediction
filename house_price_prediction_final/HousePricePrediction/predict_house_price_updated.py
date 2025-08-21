import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_house_price_updated():
    """Updated house price prediction with number of rooms and house area"""
    # Load the updated dataset
    try:
        df = pd.read_csv('house_data_updated.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset overview:")
        print(df.head())
        print(f"\nDataset statistics:")
        print(df.describe())
    except FileNotFoundError:
        print("Error: house_data_updated.csv not found. Please make sure the file is in the same directory.")
        return

    # Data Preprocessing
    # Features: 'num_rooms' and 'house_area_sqft', Target: 'price_usd'
    X = df[['num_rooms', 'house_area_sqft']]
    y = df['price_usd']
    
    print(f"\nFeatures used: {list(X.columns)}")
    print(f"Target variable: price_usd")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Mean Squared Error: ${mse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Display model coefficients
    print(f"\n{'='*50}")
    print(f"MODEL COEFFICIENTS")
    print(f"{'='*50}")
    print(f"Number of Rooms coefficient: ${model.coef_[0]:,.2f}")
    print(f"House Area coefficient: ${model.coef_[1]:,.2f}")
    print(f"Intercept: ${model.intercept_:,.2f}")
    
    # Model formula
    print(f"\n{'='*50}")
    print(f"MODEL FORMULA")
    print(f"{'='*50}")
    print(f"Price = {model.coef_[0]:,.2f} × Rooms + {model.coef_[1]:,.2f} × Area + {model.intercept_:,.2f}")

    # Example predictions for different house configurations
    print(f"\n{'='*50}")
    print(f"EXAMPLE PREDICTIONS")
    print(f"{'='*50}")
    
    example_houses = [
        (2, 1200),  # 2 rooms, 1200 sq ft
        (3, 1500),  # 3 rooms, 1500 sq ft
        (3, 1750),  # 3 rooms, 1750 sq ft
        (4, 2000),  # 4 rooms, 2000 sq ft
        (4, 2200),  # 4 rooms, 2200 sq ft
        (5, 2500),  # 5 rooms, 2500 sq ft
    ]
    
    for rooms, area in example_houses:
        predicted_price = model.predict([[rooms, area]])
        print(f"{rooms} rooms, {area:,} sq ft → ${predicted_price[0]:,.0f}")
    
    # Test with user input (optional)
    print(f"\n{'='*50}")
    print(f"INTERACTIVE PREDICTION")
    print(f"{'='*50}")
    
    try:
        user_rooms = input("Enter number of rooms (or press Enter to skip): ").strip()
        if user_rooms:
            user_rooms = int(user_rooms)
            user_area = float(input("Enter house area in sq ft: "))
            
            if user_rooms > 0 and user_area > 0:
                user_prediction = model.predict([[user_rooms, user_area]])
                print(f"\nPredicted price for {user_rooms} rooms, {user_area:,} sq ft: ${user_prediction[0]:,.0f}")
            else:
                print("Invalid input. Please enter positive values.")
    except (ValueError, KeyboardInterrupt):
        print("Skipping interactive prediction.")
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*50}")

if __name__ == '__main__':
    predict_house_price_updated()

