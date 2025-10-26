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
    
    # Make predictions with confidence intervals
    predictions = pipeline.predict_with_confidence(latest_features, confidence_level=0.95)
    
    # Get next trading day
    next_trading_day = get_next_trading_day()
    
    # Print predictions
    print(f"\n{'='*60}")
    print(f"Netflix Stock Price Predictions for {next_trading_day.strftime('%A, %B %d, %Y')}")
    print(f"{'='*60}\n")
    
    # Opening price
    print(f"Opening Price:  {format_price(predictions['open']['prediction'])}")
    print(f"  Confidence:   {predictions['open']['confidence']:.1f}%")
    print(f"  Range:        {format_price(predictions['open']['lower_bound'])} - {format_price(predictions['open']['upper_bound'])}")
    
    # High price
    print(f"\nIntraday High:  {format_price(predictions['high']['prediction'])}")
    print(f"  Confidence:   {predictions['high']['confidence']:.1f}%")
    print(f"  Range:        {format_price(predictions['high']['lower_bound'])} - {format_price(predictions['high']['upper_bound'])}")
    
    # Closing price
    print(f"\nClosing Price:  {format_price(predictions['close']['prediction'])}")
    print(f"  Confidence:   {predictions['close']['confidence']:.1f}%")
    print(f"  Range:        {format_price(predictions['close']['lower_bound'])} - {format_price(predictions['close']['upper_bound'])}")
    
    # Calculate expected price change
    current_close = processed_data['Close'].iloc[-1]
    predicted_close = predictions['close']['prediction']
    price_change = predicted_close - current_close
    price_change_pct = (price_change / current_close) * 100
    
    print(f"\n{'-'*60}")
    print(f"Current Close:     {format_price(current_close)}")
    print(f"Expected Change:   {format_price(price_change)} ({price_change_pct:+.2f}%)")
    
    # Overall model confidence (average of all three predictions)
    avg_confidence = (predictions['open']['confidence'] + 
                     predictions['high']['confidence'] + 
                     predictions['close']['confidence']) / 3
    
    print(f"\nOverall Model Confidence: {avg_confidence:.1f}%")
    
    # Interpretation
    if avg_confidence >= 90:
        confidence_text = "Very High - Model is very certain about these predictions"
    elif avg_confidence >= 80:
        confidence_text = "High - Model shows strong confidence in predictions"
    elif avg_confidence >= 70:
        confidence_text = "Moderate - Predictions have reasonable confidence"
    elif avg_confidence >= 60:
        confidence_text = "Low - Higher uncertainty in these predictions"
    else:
        confidence_text = "Very Low - Exercise caution with these predictions"
    
    print(f"Confidence Level: {confidence_text}")
    
    # Print model performance metrics (if available)
    try:
        X_train, X_test, targets_train, targets_test = pipeline.train_test_split(X, processed_data[['Open', 'High', 'Close']].shift(-1)[:-1].to_dict('series'))
        test_predictions = pipeline.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        print(f"\n{'='*60}")
        print("Model Performance Metrics (Test Set)")
        print(f"{'='*60}")
        
        for target in ['open', 'high', 'close']:
            # Only use matching indices
            common_idx = targets_test[target].index.intersection(pd.Index(range(len(test_predictions[target]))))
            y_true = targets_test[target].loc[common_idx]
            y_pred = test_predictions[target][:len(common_idx)]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print(f"\n{target.upper()} Price:")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE:  ${mae:.2f}")
            print(f"  R²:   {r2:.3f}")
        
        print(f"\n{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}\n")
        print(f"Note: Could not calculate performance metrics: {str(e)}\n")

if __name__ == '__main__':
    main()