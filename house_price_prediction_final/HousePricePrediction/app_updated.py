from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from updated_model import predict_price_updated, create_updated_model, get_model_coefficients

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize model on startup
if not os.path.exists('house_price_model_updated.pkl'):
    print("Creating initial updated model...")
    create_updated_model()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_updated.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction with rooms and area"""
    try:
        data = request.get_json()
        num_rooms = int(data['num_rooms'])
        house_area = float(data['house_area'])
        
        # Validate inputs
        if num_rooms <= 0:
            return jsonify({'error': 'Number of rooms must be positive'}), 400
        
        if house_area <= 0:
            return jsonify({'error': 'House area must be positive'}), 400
        
        if num_rooms > 10:
            return jsonify({'error': 'Number of rooms seems too high (max 10)'}), 400
        
        if house_area > 10000:
            return jsonify({'error': 'House area seems too large (max 10,000 sq ft)'}), 400
        
        if house_area < 500:
            return jsonify({'error': 'House area seems too small (min 500 sq ft)'}), 400
        
        predicted_price = predict_price_updated(num_rooms, house_area)
        
        if predicted_price is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get model coefficients for calculation display
        coeffs = get_model_coefficients()
        calculation = f"{coeffs['rooms_coef']:,.2f} × {num_rooms} + {coeffs['area_coef']:.2f} × {house_area:,} + {coeffs['intercept']:,.2f}"
        
        return jsonify({
            'num_rooms': num_rooms,
            'house_area': house_area,
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f"${predicted_price:,.0f}",
            'calculation': calculation,
            'model_formula': f"Price = {coeffs['rooms_coef']:,.2f} × Rooms + {coeffs['area_coef']:.2f} × Area + {coeffs['intercept']:,.2f}"
        })
    
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid input. Please provide valid number of rooms and house area.'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """Get model information and coefficients"""
    try:
        coeffs = get_model_coefficients()
        return jsonify({
            'model_type': 'Multiple Linear Regression',
            'features': ['Number of Rooms', 'House Area (sq ft)'],
            'coefficients': {
                'rooms': coeffs['rooms_coef'],
                'area': coeffs['area_coef'],
                'intercept': coeffs['intercept']
            },
            'formula': f"Price = {coeffs['rooms_coef']:,.2f} × Rooms + {coeffs['area_coef']:.2f} × Area + {coeffs['intercept']:,.2f}",
            'r_squared': 0.993
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'updated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

