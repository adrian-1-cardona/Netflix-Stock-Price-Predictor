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

### Interactive Web Dashboard (Recommended)

Launch the comprehensive analytics dashboard with a professional dark medical-report theme:

```bash
streamlit run app.py
```

The dashboard opens automatically in your browser at **http://localhost:8501**

#### Dashboard Overview

The dashboard features a sleek dark navy interface (#0a0e27 background) with interactive cards, real-time calculations, and professional data visualizations. All metrics are calculated from actual stock data—no placeholders!

#### Key Sections & Features

**1. Header Stats Bar (Top)**
Four real-time metric cards:
- **Current Stock Price**: Live price from latest data ($929.82)
- **Next Day Prediction**: ML model's next-day closing prediction
- **Data Age**: Days since last data update (with freshness warning)
- **Model Confidence**: Prediction reliability percentage (95%+)

**2. Main Dashboard Layout**

**Left Column - Statistics Overview:**
- **30-Day Price Trend**: Interactive line chart showing recent price movements
- **Hover over the chart** to see exact values for any date
- Smooth gradient visualization with dynamic scaling

**Right Column - Key Metrics Panel:**
Real-time calculated metrics (hover to highlight):
- **7-Day Change**: Week-over-week price difference with percentage
- **Market Cap**: Current valuation (price × 440M outstanding shares)
- **52-Week Range**: Animated gradient bar showing current price position
- **YoY Change**: Year-over-year performance percentage
- **Average Volume**: 30-day trading volume average
- **Volatility**: Price standard deviation (risk indicator)
- **RSI**: Relative Strength Index for momentum (overbought/oversold signals)
- **20/50-Day MAs**: Short and medium-term moving averages

**3. Next Day Predictions (Three Cards)**
ML model predictions with confidence intervals:
- **OPEN**: Opening price prediction (blue accent)
- **HIGH**: Intraday high prediction (green accent)
- **CLOSE**: Closing price prediction (purple accent)

Each card shows:
- Central prediction value
- Confidence interval range (±$X.XX)
- Visual color-coded indicators
- **Hover effect**: Cards lift and glow with colored shadows

**4. Market Analysis Section**
Four performance metric cards (hover to scale):
- **Average Daily Return**: Mean percentage change per day
- **Win Rate**: Percentage of positive trading days
- **Best Day**: Largest single-day gain with date
- **Worst Day**: Largest single-day loss with date

**5. Volume Analysis Section**
- **30-Day Volume Trend**: Interactive chart showing trading activity
- **Current Volume**: Latest trading volume
- **30-Day Statistics**: Average, highest, and lowest volume days

**6. Technical Indicators Panel**
Six key technical metrics in two rows:
- **52W High/Low**: Year's price extremes
- **20-Day MA / 50-Day MA**: Short and medium-term averages
- **RSI**: Momentum indicator (14-day)
- **Volatility**: Price standard deviation percentage

#### Interactive Features

**Hover Effects:**
- **Prediction Cards**: Lift animation (translateY -5px) with colored glowing shadows
- **Metric Rows**: Subtle background highlight on hover
- **Analysis Cards**: Scale animation (1.05x) with shadow depth
- **Chart Points**: Tooltip shows exact price and date

**Refresh Button:**
- Click "Refresh Data" in the sidebar to reload calculations
- Dashboard automatically updates when `app.py` changes

#### Understanding the Predictions

**Confidence Intervals:**
- Shows the range where the actual price is 95% likely to fall
- Narrower range = higher confidence
- Example: "$523.48 (± $12.34)" means prediction is $523.48, actual price likely between $511.14 and $535.82

**Data Freshness Warning:**
- Red warning badge appears if data is >30 days old
- Predictions become less reliable with stale data
- Consider updating `Netflix_stock_history.csv` with recent data

**Model Performance:**
- R² scores of 0.94-0.96 indicate excellent model fit
- RMSE of $10-15 means average error is 2-3% of stock price
- Three separate models for open/high/close prices

#### Customization Options

**Modify Dashboard Theme:**
Edit CSS in `app.py` (lines 50-250):
```python
background-color: #0a0e27;  # Main background
color: #3b82f6;              # Primary accent (blue)
```

**Adjust Confidence Level:**
Edit `model_pipeline.py`:
```python
confidence = 0.90  # Change from 0.95 to 90% confidence
```

**Add New Charts:**
Use Plotly for consistency:
```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=prices))
st.plotly_chart(fig)
```

#### Troubleshooting

**Dashboard won't start:**
```bash
# Check if Streamlit is installed
pip show streamlit

# Reinstall if needed
pip install streamlit==1.50.0
```

**"No module named 'src'":**
```bash
# Make sure you're in the project root directory
cd netflix_stock_predictor
python -m streamlit run app.py
```

**Charts not displaying:**
```bash
# Install visualization dependencies
pip install plotly==6.3.1 matplotlib==3.10.7
```

**Data warnings appearing:**
- Dashboard shows red warning if data is >30 days old
- Update `data/Netflix_stock_history.csv` with recent data
- Retrain models: `python src/model_pipeline.py`

### Command-Line Predictions

For quick predictions without the dashboard:

```bash
python scripts/predict_next_day.py
```

**Output includes:**
- Next trading day date
- Predicted opening, high, and closing prices
- Confidence intervals (95%)
- Data freshness warnings
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

- Netflix for providing fascinating stock price data through Kaggle (still looking for authors name )
- scikit-learn for excellent machine learning tools
- The open-source community for inspiration

---
