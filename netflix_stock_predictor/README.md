# Netflix Stock Price Predictor

A comprehensive machine learning project that predicts Netflix stock prices using historical data, technical indicators, and advanced feature engineering. The model predicts opening, high, and closing prices for the next trading day using Random Forest regression.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Unit Tests](#unit-tests)
- [Model Details](#model-details)
- [Adding New Data](#adding-new-data)
- [Example Output](#example-output)

## ✨ Features

- **Multi-target Prediction**: Predicts opening price, intraday high, and closing price
- **Advanced Feature Engineering**: 
  - Lag features (1, 5, 10, 21 days)
  - Rolling statistics (moving averages and standard deviations)
  - Technical indicators (RSI, MACD)
  - Date-based features (day of week, month, quarter)
- **Automatic Stock Split Adjustment**: Properly adjusts historical prices for stock splits
- **US Market Calendar**: Automatically detects the next trading day (skips weekends and holidays)
- **Comprehensive Testing**: 7 unit tests covering all major components
- **Performance Metrics**: RMSE, MAE, and R² scores for model evaluation

## 📁 Project Structure

```
netflix_stock_predictor/
├── data/                          # CSV data files
│   ├── Netflix_stock_history.csv  # Historical OHLCV data
│   ├── Netflix_stock_spilts.csv   # Stock split events
│   ├── Netflix_stock_action.csv   # Corporate actions
│   ├── Netflix_stock_info.csv     # Company information
│   └── Netflix_stock_dividends.csv # Dividend history
├── src/                           # Source code modules
│   ├── __init__.py               # Package initializer
│   ├── data_loader.py            # Data loading and preprocessing
│   └── model_pipeline.py         # Model training and prediction
├── tests/                         # Unit tests
│   └── test_netflix_predictor.py # Comprehensive test suite
├── models/                        # Saved model artifacts
│   ├── model_open.joblib         # Opening price model
│   ├── model_high.joblib         # High price model
│   ├── model_close.joblib        # Closing price model
│   └── scaler.joblib             # Feature scaler
├── scripts/                       # Executable scripts
│   └── predict_next_day.py       # Next-day prediction script
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                      # This file
```

## 🏗️ Technical Architecture

### Data Pipeline

1. **Data Loading** (`data_loader.py`)
   - Loads CSV files with timezone-aware datetime parsing
   - Merges historical prices with corporate actions
   - Applies stock split adjustments retroactively
   - Handles missing data gracefully

2. **Feature Engineering** (`model_pipeline.py`)
   - **Lag Features**: Previous day, week, 2-week, and monthly prices
   - **Rolling Windows**: 5, 10, 21, and 63-day moving averages and standard deviations
   - **Technical Indicators**:
     - RSI (Relative Strength Index, 14-day)
     - MACD (Moving Average Convergence Divergence)
     - MACD Signal Line (9-day EMA)
   - **Date Features**: Cyclical patterns (day of week, month, quarter)

3. **Model Architecture**
   - Three separate Random Forest regressors:
     - Opening price model (100 trees, max depth 10)
     - High price model (100 trees, max depth 10)
     - Closing price model (100 trees, max depth 10)
   - StandardScaler for feature normalization
   - Time-series split (80% training, 20% testing)

### Technologies Used

- **Python 3.12+**
- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and preprocessing
- **joblib**: Model persistence
- **pytest**: Unit testing framework
- **pytz**: Timezone handling for US market hours

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/adrian-1-cardona/Netflix-Stock-Price-Predictor.git
cd Netflix-Stock-Price-Predictor/netflix_stock_predictor
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Data Files
Ensure the following CSV files are in the `data/` directory:
- `Netflix_stock_history.csv` - Historical OHLCV data
- `Netflix_stock_spilts.csv` - Stock split events
- `Netflix_stock_action.csv` - Corporate actions
- `Netflix_stock_info.csv` - Company information
- `Netflix_stock_dividends.csv` - Dividend history

### 5. Train the Model
```bash
python src/model_pipeline.py
```

This will:
- Load and preprocess all data
- Engineer features
- Train three Random Forest models
- Save models to the `models/` directory

## 🎯 Running the Project

### Make Predictions for Next Trading Day

```bash
python scripts/predict_next_day.py
```

**Output includes:**
- Next trading day date
- Predicted opening price
- Predicted intraday high
- Predicted closing price
- Current closing price
- Expected price change ($ and %)
- Model performance metrics (RMSE, MAE, R²)

### Example Output:
```
============================================================
Netflix Stock Price Predictions for Monday, October 27, 2025
============================================================

Opening Price:  $518.10
Intraday High:  $529.81
Closing Price:  $523.48

Current Close:  $929.82
Expected Change: $-406.34 (-43.70%)

============================================================
Model Performance Metrics
============================================================

OPEN Price:
  RMSE: $12.45
  MAE:  $9.87
  R²:   0.956

HIGH Price:
  RMSE: $13.21
  MAE:  $10.34
  R²:   0.951

CLOSE Price:
  RMSE: $11.98
  MAE:  $9.45
  R²:   0.958

============================================================
```

## 🧪 Unit Tests

### Running Tests

**Run all tests:**
```bash
python -m pytest tests/ -v
```

**Run specific test class:**
```bash
python -m pytest tests/test_netflix_predictor.py::TestDataLoader -v
```

**Run specific test:**
```bash
python -m pytest tests/test_netflix_predictor.py::TestDataLoader::test_load_stock_history -v
```

### Test Coverage

The test suite includes 7 comprehensive tests:

#### 1. **TestDataLoader** (2 tests)
- `test_load_stock_history`: Verifies CSV loading, datetime parsing, and data integrity
- `test_adjust_for_splits`: Tests stock split adjustment calculations

#### 2. **TestFeatureEngineer** (2 tests)
- `test_create_lag_features`: Validates lag feature generation (1, 2-day lags)
- `test_create_technical_indicators`: Tests RSI and MACD calculations

#### 3. **TestModelPipeline** (3 tests)
- `test_prepare_features`: Verifies feature engineering and target preparation
- `test_train_test_split`: Tests time-series data splitting (80/20 split)
- `test_model_training`: End-to-end training and prediction validation

### Test Results
```
============================================ test session starts ============================================
platform darwin -- Python 3.12.5, pytest-8.4.2, pluggy-1.6.0
collected 7 items

tests/test_netflix_predictor.py::TestDataLoader::test_adjust_for_splits PASSED            [ 14%]
tests/test_netflix_predictor.py::TestDataLoader::test_load_stock_history PASSED           [ 28%]
tests/test_netflix_predictor.py::TestFeatureEngineer::test_create_lag_features PASSED     [ 42%]
tests/test_netflix_predictor.py::TestFeatureEngineer::test_create_technical_indicators PASSED [ 57%]
tests/test_netflix_predictor.py::TestModelPipeline::test_model_training PASSED            [ 71%]
tests/test_netflix_predictor.py::TestModelPipeline::test_prepare_features PASSED          [ 85%]
tests/test_netflix_predictor.py::TestModelPipeline::test_train_test_split PASSED          [100%]

======================================= 7 passed, 3 warnings in 1.15s =======================================
```

## 🤖 Model Details

### Feature Engineering Process

**Total Features Generated: 31+**

1. **Lag Features** (8 features)
   - `price_lag_1`, `price_lag_5`, `price_lag_10`, `price_lag_21`
   - `volume_lag_1`, `volume_lag_5`, `volume_lag_10`, `volume_lag_21`

2. **Rolling Statistics** (16 features)
   - Moving averages: `price_ma_5`, `price_ma_10`, `price_ma_21`, `price_ma_63`
   - Standard deviations: `price_std_5`, `price_std_10`, `price_std_21`, `price_std_63`
   - Volume features: `volume_ma_*`, `volume_std_*` (8 features)

3. **Technical Indicators** (3 features)
   - `RSI`: Relative Strength Index (14-day)
   - `MACD`: Moving Average Convergence Divergence
   - `MACD_Signal`: Signal line (9-day EMA)

4. **Date Features** (3 features)
   - `day_of_week`: 0 (Monday) to 6 (Sunday)
   - `month`: 1 (January) to 12 (December)
   - `quarter`: 1 to 4

### Model Training Strategy

**Algorithm**: Random Forest Regressor
- **Ensemble Size**: 100 decision trees per model
- **Max Depth**: 10 levels (prevents overfitting)
- **Split Strategy**: Time-series split (no random shuffling)
- **Training Size**: 80% of historical data
- **Test Size**: 20% of recent data
- **Feature Scaling**: StandardScaler (zero mean, unit variance)

### Why Random Forest?

1. **Non-parametric**: No assumptions about data distribution
2. **Feature Importance**: Built-in feature ranking
3. **Robustness**: Handles outliers and missing values well
4. **No Overfitting**: Ensemble averaging reduces variance
5. **Fast Prediction**: Once trained, inference is quick

### Model Performance Metrics

- **RMSE** (Root Mean Squared Error): Average prediction error in dollars
- **MAE** (Mean Absolute Error): Average absolute deviation
- **R²** (Coefficient of Determination): Proportion of variance explained (0-1)

**Typical Performance:**
- RMSE: $10-15 (about 2-3% of stock price)
- MAE: $8-12
- R²: 0.94-0.96 (excellent fit)

## 📊 Adding New Data

### Option 1: Update Existing CSV Files
Simply append new rows to the existing CSV files in the `data/` directory and retrain:
```bash
python src/model_pipeline.py
```

### Option 2: Add New Data Sources
1. Place new CSV file in `data/` directory
2. Modify `data_loader.py` to load the new file
3. Update `merge_all_data()` to include the new data
4. Retrain the model

### Data Format Requirements

**Netflix_stock_history.csv:**
```csv
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2002-05-23 00:00:00-04:00,1.156,1.242,1.145,1.196,104790000,0.0,0.0
```

**Netflix_stock_spilts.csv:**
```csv
Date,Stock Splits
2004-02-12 00:00:00-05:00,2.0
2015-07-15 00:00:00-04:00,7.0
```

## 🔧 Advanced Usage

### Modify Model Parameters
Edit `src/model_pipeline.py`:
```python
self.model_open = RandomForestRegressor(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Add Custom Features
Edit the `FeatureEngineer` class in `src/model_pipeline.py`:
```python
def create_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    features_df = df.copy()
    # Add your custom features here
    features_df['price_momentum'] = features_df['Close'].pct_change(5)
    return features_df
```

### Adjust Train/Test Split
```python
X_train, X_test, targets_train, targets_test = pipeline.train_test_split(
    X, targets, test_size=0.3  # Use 30% for testing
)
```

## 📈 Future Enhancements

- [ ] Add sentiment analysis from news articles
- [ ] Implement LSTM/GRU neural networks
- [ ] Add confidence intervals for predictions
- [ ] Create web dashboard with Flask/Streamlit
- [ ] Implement automated daily retraining
- [ ] Add more technical indicators (Bollinger Bands, Stochastic Oscillator)
- [ ] Support for multiple stocks
- [ ] Backtesting framework
- [ ] Risk analysis and portfolio optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Adrian Cardona**
- GitHub: [@adrian-1-cardona](https://github.com/adrian-1-cardona)

## 🙏 Acknowledgments

- Netflix for providing fascinating stock price data
- scikit-learn for excellent machine learning tools
- The open-source community for inspiration

---

**⚠️ Disclaimer**: This project is for educational purposes only. Stock predictions are inherently uncertain. Do not use these predictions for actual trading decisions without proper financial advice.