# 🚀 Quick Start Guide - Netflix Stock Price Predictor Dashboard

## Launch the Dashboard in 3 Steps

### Step 1: Activate Virtual Environment
```bash
cd "/Users/adriancardina/Desktop/Netflix Stock Price Predictor/netflix_stock_predictor"
source venv/bin/activate
```

### Step 2: Run the Dashboard
```bash
streamlit run app.py
```

### Step 3: Open in Browser
The dashboard will automatically open at: **http://localhost:8501**

---

## 📊 Dashboard Features

### 1. 🎯 Predictions Tab
View next-day price predictions with:
- **Opening Price** - When market opens
- **Intraday High** - Highest price expected
- **Closing Price** - End-of-day price
- **Confidence Gauges** - Model certainty (90%+)
- **Price Ranges** - Upper and lower bounds

### 2. 📈 Historical Data Tab
Explore past performance:
- **Candlestick Charts** - Daily OHLC visualization
- **Moving Averages** - 20-day and 50-day trends
- **Volume Analysis** - Trading activity patterns
- **Time Filters** - 1M, 3M, 6M, 1Y, All Time

### 3. 📊 Analytics Tab
Technical analysis insights:
- **Returns Distribution** - Daily gain/loss patterns
- **Volatility Metrics** - Price fluctuation analysis
- **RSI Indicator** - Overbought/oversold signals
- **Statistics** - Max gains, losses, volatility

### 4. ℹ️ Model Info Tab
Understand the predictions:
- **Model Architecture** - Random Forest details
- **Feature Engineering** - 31+ features explained
- **Feature Importance** - Top 10 influential factors
- **Performance Metrics** - R² scores

---

## ⚠️ Data Freshness Alert

The dashboard automatically checks data age:
- **🟢 Green** - Data is recent (< 7 days old)
- **🟡 Yellow** - Data is aging (7-30 days old)
- **🔴 Red** - Data is outdated (> 30 days old)

**Current Status:** Data is **222 days old** (last update: March 18, 2025)

---

## 🔄 Updating Data

To update with fresh stock data:

1. **Download new CSV files** from your data provider
2. **Replace files** in the `data/` directory:
   - `Netflix_stock_history.csv`
   - Other CSV files as needed
3. **Retrain the model**:
   ```bash
   python src/model_pipeline.py
   ```
4. **Refresh dashboard** - Click "🔄 Refresh Data" in sidebar

---

## 🛠️ Troubleshooting

### Dashboard won't start?
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Port already in use?
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Model not found?
```bash
# Train the model first
python src/model_pipeline.py
```

---

## 💡 Tips for Best Experience

1. **Full Screen** - Use full browser window for best visualization
2. **Dark Mode** - Toggle in Streamlit settings (⋮ menu)
3. **Explore Tabs** - Each tab offers unique insights
4. **Hover Charts** - Hover over charts for detailed tooltips
5. **Time Ranges** - Switch between time periods for different perspectives

---

## 📱 Access from Other Devices

Dashboard is accessible on your local network:

1. Find your IP address:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

2. Access from other devices:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```

---

## 🎓 Understanding Predictions

### Confidence Levels
- **90-100%** 🟢 Very High - Strong confidence
- **80-89%** 🟡 High - Good confidence
- **70-79%** 🟠 Moderate - Reasonable confidence
- **Below 70%** 🔴 Low - Exercise caution

### Price Ranges
- **Prediction** - Most likely price
- **Upper Bound** - 95% confidence ceiling
- **Lower Bound** - 95% confidence floor

### Technical Indicators
- **RSI > 70** - Potentially overbought
- **RSI < 30** - Potentially oversold
- **MA Crossover** - Trend change signal

---

## 📚 Learn More

- **README.md** - Comprehensive documentation
- **Model Details** - See "Model Info" tab in dashboard
- **Run Tests** - `pytest tests/`

---

**Built with ❤️ using Streamlit, scikit-learn, and Python**
