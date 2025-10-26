"""
Script to predict Netflix stock prices for the next trading day.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader
from src.model_pipeline import ModelPipeline

def is_market_holiday(date: datetime) -> bool:
    """
    Check if a given date is a US market holiday.
    This is a simplified version - in production, use a proper calendar library.
    """
    holidays = [
        # 2025 holidays - update yearly
        "2025-01-01",  # New Year's Day
        "2025-01-20",  # Martin Luther King Jr. Day
        "2025-02-17",  # Presidents Day
        "2025-04-18",  # Good Friday
        "2025-05-26",  # Memorial Day
        "2025-07-04",  # Independence Day
        "2025-09-01",  # Labor Day
        "2025-11-27",  # Thanksgiving Day
        "2025-12-25",  # Christmas Day
    ]
    return date.strftime('%Y-%m-%d') in holidays

def get_next_trading_day() -> datetime:
    """Get the next US market trading day."""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    
    # Start with tomorrow
    next_day = now + timedelta(days=1)
    
    # Keep incrementing until we find a trading day
    while (
        next_day.weekday() >= 5 or  # Weekend
        is_market_holiday(next_day)  # Holiday
    ):
        next_day += timedelta(days=1)
    
    return next_day

def get_prediction_intervals(
    predictions: np.ndarray,
    std_dev: float,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate prediction intervals.
    
    Args:
        predictions (np.ndarray): Model predictions
        std_dev (float): Standard deviation of predictions
        confidence (float): Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple[float, float]: Lower and upper bounds of prediction interval
    """
    from scipy import stats
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_dev
    
    return predictions - margin, predictions + margin

def format_price(price: float) -> str:
    """Format price with 2 decimal places and dollar sign."""
    return f"${price:.2f}"

def main():
    # Initialize data loader and model pipeline
    loader = DataLoader('data')
    pipeline = ModelPipeline('models')
    
    # Load saved model
    pipeline.load_model()
    
    # Get latest data
    data = loader.merge_all_data()
    processed_data = loader.preprocess_data(data)
    
    # Prepare features for prediction
    X, _ = pipeline.prepare_features(processed_data)
    latest_features = X.iloc[-1:]
    
    # Make predictions
    predictions = pipeline.predict(latest_features)
    
    # Get prediction intervals
    std_dev = np.std(predictions)
    lower_bound, upper_bound = get_prediction_intervals(predictions[0], std_dev)
    
    # Get next trading day
    next_trading_day = get_next_trading_day()
    
    # Print predictions
    print(f"\nPredictions for {next_trading_day.strftime('%Y-%m-%d')}:")
    print("-" * 50)
    print(f"Opening Price: {format_price(predictions[0])}")
    print(f"95% Confidence Interval: [{format_price(lower_bound)} - {format_price(upper_bound)}]")
    
    # Print model performance metrics (if available)
    try:
        X_train, X_test, y_train, y_test = pipeline.train_test_split(X, processed_data['Open'])
        test_predictions = pipeline.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        print("\nModel Performance Metrics:")
        print("-" * 50)
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.3f}")
    except Exception as e:
        print("\nNote: Could not calculate performance metrics:", str(e))

if __name__ == '__main__':
    main()