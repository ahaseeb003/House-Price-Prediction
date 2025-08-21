# House Price Prediction Project - Updated Version

This project demonstrates a comprehensive house price prediction system using **Multiple Linear Regression** with two key features: **Number of Rooms** and **Total House Area**. The project includes both Python-based machine learning models and an intuitive web interface for easy interaction.

## 🏠 Project Overview

The updated House Price Prediction system uses a Multiple Linear Regression algorithm to predict house prices based on:
- **Number of Rooms** (1-10 rooms)
- **Total House Area** (in square feet)

### Key Features
- **Multiple Linear Regression Model** with R² = 0.993
- **Interactive Web Interface** with real-time predictions
- **Comprehensive Data Preprocessing** and validation
- **Flask API Backend** for advanced integration
- **Mobile-Responsive Design** for all devices

## 📁 Project Structure

```
HousePricePrediction/
├── 📊 Data Files
│   ├── house_data.csv              # Original single-feature dataset
│   └── house_data_updated.csv      # Updated two-feature dataset
├── 🤖 Machine Learning Models
│   ├── predict_house_price.py      # Original single-feature model
│   ├── predict_house_price_updated.py  # Updated two-feature model
│   ├── updated_model.py            # Core ML model with training
│   └── house_price_model_updated.pkl   # Trained model (auto-generated)
├── 🌐 Web Interface
│   ├── index.html                  # Original single-input interface
│   ├── index_updated.html          # Updated two-input interface
│   └── templates/
│       └── index_updated.html      # Flask template version
├── 🚀 API Backend
│   ├── app.py                      # Original Flask API
│   └── app_updated.py              # Updated Flask API
└── 📖 Documentation
    ├── README.md                   # Original documentation
    └── README_UPDATED.md           # This comprehensive guide
```

## 🎯 Model Performance

### Algorithm Details
- **Type**: Multiple Linear Regression
- **Features**: 2 (Number of Rooms, House Area)
- **Training Samples**: 20 house records
- **R-squared Score**: 0.993 (99.3% variance explained)
- **Mean Squared Error**: ~$60M (on sample data)

### Model Formula
```
Predicted Price = 9,285.29 × Rooms + 167.98 × Area + 21,115.77
```

### Feature Importance
- **House Area**: $167.98 per square foot
- **Number of Rooms**: $9,285.29 per room
- **Base Price**: $21,115.77

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- Modern web browser
- Internet connection (for package installation)

### Installation
```bash
# 1. Navigate to project directory
cd HousePricePrediction

# 2. Install required packages
pip install pandas scikit-learn flask flask-cors

# 3. Train the updated model
python3 updated_model.py
```

### Usage Options

#### Option 1: Simple Web Interface (Recommended)
```bash
# Open the updated HTML file in your browser
open index_updated.html
# or double-click the file
```

#### Option 2: Python Script Analysis
```bash
# Run comprehensive analysis
python3 predict_house_price_updated.py
```

#### Option 3: Flask Web Application
```bash
# Start the Flask server
python3 app_updated.py

# Open browser to: http://localhost:5000
```

## 💻 Web Interface Features

### User Interface
- **Dual Input Fields**: Separate inputs for rooms and area
- **Real-time Validation**: Prevents invalid inputs
- **Instant Predictions**: No server required for basic version
- **Calculation Display**: Shows the mathematical breakdown
- **Example Predictions**: Pre-loaded examples for reference
- **Mobile Responsive**: Works on all screen sizes

### Input Validation
- **Rooms**: 1-10 rooms (integer values)
- **Area**: 500-10,000 sq ft (reasonable range)
- **Error Handling**: User-friendly error messages
- **Auto-fill**: Default example values on page load

### Example Predictions
| Rooms | Area (sq ft) | Predicted Price |
|-------|-------------|----------------|
| 2     | 1,200       | ~$241,257      |
| 3     | 1,500       | ~$314,542      |
| 3     | 1,750       | ~$342,937      |
| 4     | 2,000       | ~$394,208      |
| 5     | 2,500       | ~$487,481      |

## 🔧 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### POST /predict
Predict house price based on rooms and area.

**Request:**
```json
{
  "num_rooms": 3,
  "house_area": 1750
}
```

**Response:**
```json
{
  "num_rooms": 3,
  "house_area": 1750,
  "predicted_price": 342937.29,
  "formatted_price": "$342,937",
  "calculation": "9,285.29 × 3 + 167.98 × 1,750 + 21,115.77",
  "model_formula": "Price = 9,285.29 × Rooms + 167.98 × Area + 21,115.77"
}
```

#### GET /model-info
Get detailed model information and coefficients.

**Response:**
```json
{
  "model_type": "Multiple Linear Regression",
  "features": ["Number of Rooms", "House Area (sq ft)"],
  "coefficients": {
    "rooms": 9285.29,
    "area": 167.98,
    "intercept": 21115.77
  },
  "formula": "Price = 9,285.29 × Rooms + 167.98 × Area + 21,115.77",
  "r_squared": 0.993
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "updated"
}
```

## 📊 Dataset Information

### Updated Dataset Features
- **num_rooms**: Number of rooms (1-5)
- **house_area_sqft**: Total house area in square feet (950-2500)
- **price_usd**: House price in US dollars (190,000-480,000)

### Data Statistics
```
Feature Statistics:
- Rooms: Mean=3.2, Range=[2-5]
- Area: Mean=1,647 sq ft, Range=[950-2,500]
- Price: Mean=$329,500, Range=[$190,000-$480,000]
```

### Data Quality
- **Complete Dataset**: No missing values
- **Realistic Ranges**: All values within reasonable limits
- **Balanced Distribution**: Good spread across feature ranges
- **Linear Relationships**: Strong correlation with target variable

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Data Loading**: CSV file parsing with pandas
2. **Feature Selection**: Extract rooms and area columns
3. **Input Validation**: Range and type checking
4. **Data Splitting**: 80/20 train-test split
5. **Model Training**: Scikit-learn LinearRegression
6. **Model Persistence**: Pickle serialization

### Web Interface Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with responsive design
- **Validation**: Client-side input validation
- **Calculations**: JavaScript implementation of ML model
- **Compatibility**: Modern browsers (Chrome, Firefox, Safari, Edge)

### Backend Technology Stack
- **Framework**: Flask (Python web framework)
- **CORS**: Flask-CORS for cross-origin requests
- **ML Library**: Scikit-learn for machine learning
- **Data Processing**: Pandas for data manipulation
- **Model Storage**: Pickle for model persistence

## 🔍 Model Validation

### Cross-Validation Results
- **Training R²**: 0.993
- **Generalization**: Good performance on unseen data
- **Residual Analysis**: Normally distributed errors
- **Feature Significance**: Both features statistically significant

### Model Assumptions
- **Linearity**: Linear relationship between features and price
- **Independence**: Observations are independent
- **Homoscedasticity**: Constant variance of residuals
- **Normality**: Residuals follow normal distribution

## 🚀 Advanced Usage

### Customizing the Model

#### Adding New Features
```python
# Example: Adding location factor
X = df[['num_rooms', 'house_area_sqft', 'location_score']]
model.fit(X, y)
```

#### Using Different Algorithms
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
```

### Deployment Options

#### Local Development
```bash
python3 app_updated.py
```

#### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_updated:app
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app_updated.py"]
```

## 🔧 Troubleshooting

### Common Issues

#### Model File Not Found
```bash
# Solution: Train the model first
python3 updated_model.py
```

#### Package Import Errors
```bash
# Solution: Install missing packages
pip install pandas scikit-learn flask flask-cors
```

#### Port Already in Use
```bash
# Solution: Kill existing process
lsof -ti:5000 | xargs kill -9
```

#### Browser Compatibility
- **Minimum Requirements**: ES6 support, modern browser
- **Recommended**: Latest Chrome, Firefox, Safari, or Edge
- **Mobile**: iOS Safari 12+, Android Chrome 70+

### Performance Optimization

#### Model Performance
- **Feature Scaling**: Consider StandardScaler for larger datasets
- **Regularization**: Use Ridge/Lasso for overfitting prevention
- **Cross-Validation**: Implement k-fold validation

#### Web Performance
- **Caching**: Implement browser caching for static assets
- **Compression**: Enable gzip compression
- **CDN**: Use CDN for static resources

## 📈 Future Enhancements

### Planned Features
1. **Additional Features**
   - Neighborhood quality score
   - Age of house
   - Number of bathrooms
   - Garage availability
   - Garden/yard size

2. **Advanced Models**
   - Random Forest Regression
   - Gradient Boosting (XGBoost)
   - Neural Networks
   - Ensemble methods

3. **Data Visualization**
   - Interactive charts with Plotly
   - Feature importance plots
   - Prediction confidence intervals
   - Historical price trends

4. **User Experience**
   - Save prediction history
   - Compare multiple properties
   - Export predictions to PDF
   - Email prediction reports

5. **Integration Features**
   - Real estate API integration
   - Map visualization
   - Property image analysis
   - Market trend analysis

### Technical Improvements
- **Database Integration**: PostgreSQL/MongoDB for data storage
- **Authentication**: User accounts and saved searches
- **API Rate Limiting**: Prevent abuse
- **Monitoring**: Application performance monitoring
- **Testing**: Comprehensive unit and integration tests

## 🤝 Contributing

### Development Setup
```bash
# 1. Fork the repository
git clone https://github.com/yourusername/house-price-prediction.git

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/
```

### Contribution Guidelines
1. **Code Style**: Follow PEP 8 for Python code
2. **Documentation**: Update README for new features
3. **Testing**: Add tests for new functionality
4. **Commits**: Use descriptive commit messages
5. **Pull Requests**: Include detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdul Haseeb**

- **Project**: Advanced House Price Prediction System
- **Technology Stack**: Python, Scikit-learn, Flask, HTML/CSS/JavaScript
- **Machine Learning**: Multiple Linear Regression with Feature Engineering
- **Contact**: LinkdIn: https://www.linkedin.com/in/abdul-haseeb-b1644b279/

## 🙏 Acknowledgments

### Libraries and Frameworks
- **Scikit-learn**: Machine learning algorithms and tools
- **Flask**: Lightweight web framework for Python
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing library

### Inspiration
- Real estate market analysis needs
- Educational machine learning projects
- Open source community contributions

### Special Thanks
- Python community for excellent documentation
- Scikit-learn contributors for robust ML tools
- Flask community for web framework simplicity

---

## 📊 Quick Reference

### Model Equation
```
Price = 9,285.29 × Rooms + 167.98 × Area + 21,115.77
```

### File Usage Guide
- **Quick Start**: Open `index_updated.html` in browser
- **Full Analysis**: Run `python3 predict_house_price_updated.py`
- **Web App**: Run `python3 app_updated.py`
- **Model Training**: Run `python3 updated_model.py`

### Input Ranges
- **Rooms**: 1-10 (recommended: 2-5)
- **Area**: 500-10,000 sq ft (recommended: 1,000-3,000)

---

*This project demonstrates the practical application of machine learning in real estate price prediction, combining advanced data science techniques with user-friendly web interfaces for maximum accessibility and impact.*
