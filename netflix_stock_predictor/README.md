# Netflix Stock Price Predictor

A machine learning project to predict Netflix stock prices using historical data and various market indicators.

## Project Structure

```
netflix_stock_predictor/
├── data/               # CSV data files
├── src/               # Source code
├── tests/             # Unit tests
├── models/            # Saved model files
├── scripts/           # Prediction scripts
└── README.md
```

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place CSV files in the `data/` directory:
   - Netflix_stock_history.csv
   - Netflix_stock_splits.csv
   - Netflix_stock_action.csv
   - Netflix_stock_info.csv
   - Netflix_stock_dividends.csv

## Running the Project

### Prediction Script
To get predictions for the next trading day:
```bash
python scripts/predict_next_day.py
```

### Running Tests
```bash
pytest tests/
```

## Example Output
```
Next Trading Day Predictions (2025-10-26):
Opening Price: $XXX.XX (Confidence: XX%)
Intraday High: $XXX.XX (Confidence: XX%)
Closing Price: $XXX.XX (Confidence: XX%)

Model Performance Metrics:
RMSE: X.XX
MAE: X.XX
R²: X.XX
```

## Adding New Data

1. Place new CSV files in the `data/` directory
2. Ensure the format matches existing files
3. Run data preprocessing script:
   ```bash
   python src/data_loader.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.