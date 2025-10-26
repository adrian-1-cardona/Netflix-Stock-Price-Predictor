"""
Unit tests for Netflix stock price predictor.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader
from src.model_pipeline import ModelPipeline, FeatureEngineer

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.data_dir = Path(__file__).resolve().parent.parent / 'data'
        self.loader = DataLoader(str(self.data_dir))
        
    def test_load_stock_history(self):
        """Test loading historical stock data."""
        df = self.loader.load_stock_history()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertIn('Close', df.columns)
        
    def test_adjust_for_splits(self):
        """Test stock split adjustments."""
        # Create sample data
        history = pd.DataFrame({
            'Open': [100, 100, 100],
            'High': [105, 105, 105],
            'Low': [95, 95, 95],
            'Close': [102, 102, 102],
            'Volume': [1000, 1000, 1000]
        }, index=pd.date_range('2025-01-01', '2025-01-03'))
        
        splits = pd.DataFrame({
            'SplitRatio': [2.0]  # 2:1 split
        }, index=pd.DatetimeIndex(['2025-01-02']))
        
        adjusted = self.loader.adjust_for_splits(history, splits)
        
        # Check that prices before split date are halved
        self.assertEqual(adjusted.loc['2025-01-01', 'Close'], 51.0)
        self.assertEqual(adjusted.loc['2025-01-03', 'Close'], 102.0)

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        dates = pd.date_range('2025-01-01', '2025-01-10')
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000, 2000, len(dates))
        }, index=dates)
        
    def test_create_lag_features(self):
        """Test creation of lag features."""
        df = self.engineer.create_lag_features(self.sample_data, lags=[1, 2])
        
        self.assertIn('price_lag_1', df.columns)
        self.assertIn('price_lag_2', df.columns)
        self.assertIn('volume_lag_1', df.columns)
        self.assertIn('volume_lag_2', df.columns)
        
    def test_create_technical_indicators(self):
        """Test creation of technical indicators."""
        df = self.engineer.create_technical_indicators(self.sample_data)
        
        self.assertIn('RSI', df.columns)
        self.assertIn('MACD', df.columns)

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.model_dir = Path(__file__).resolve().parent.parent / 'models'
        self.pipeline = ModelPipeline(str(self.model_dir))
        
        # Create sample data with all required columns (need more rows for feature engineering)
        dates = pd.date_range('2025-01-01', '2025-06-30')  # 6 months of data
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(100, 200, len(dates)),
            'Low': np.random.uniform(100, 200, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000, 2000, len(dates))
        }, index=dates)
        
    def test_prepare_features(self):
        """Test feature preparation."""
        X, targets = self.pipeline.prepare_features(self.sample_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(targets, dict)
        self.assertIn('open', targets)
        self.assertIn('high', targets)
        self.assertIn('close', targets)
        self.assertEqual(len(X), len(targets['open']))
        
    def test_train_test_split(self):
        """Test data splitting."""
        X, targets = self.pipeline.prepare_features(self.sample_data)
        X_train, X_test, targets_train, targets_test = self.pipeline.train_test_split(X, targets, test_size=0.2)
        
        self.assertTrue(len(X_train) > len(X_test))
        self.assertEqual(len(X_train) + len(X_test), len(X))
        
    def test_model_training(self):
        """Test model training and prediction."""
        X, targets = self.pipeline.prepare_features(self.sample_data)
        X_train, X_test, targets_train, targets_test = self.pipeline.train_test_split(X, targets)
        
        self.pipeline.train(X_train, targets_train)
        predictions = self.pipeline.predict(X_test)
        
        self.assertIsInstance(predictions, dict)
        self.assertIn('open', predictions)
        self.assertIn('high', predictions)
        self.assertIn('close', predictions)
        self.assertEqual(len(predictions['open']), len(X_test))
        self.assertTrue(all(isinstance(pred, (float, np.float64)) for pred in predictions['open']))

if __name__ == '__main__':
    unittest.main()