"""
Model pipeline for Netflix stock price prediction.
Handles feature engineering, model training, and predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import joblib
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer with default parameters."""
        self.scaler = StandardScaler()
        
    def create_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Create lagged price and volume features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            lags (List[int]): List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        features_df = df.copy()
        
        for lag in lags:
            features_df[f'price_lag_{lag}'] = features_df['Close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['Volume'].shift(lag)
            
        return features_df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df (pd.DataFrame): Input dataframe
            windows (List[int]): List of window sizes
            
        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        features_df = df.copy()
        
        for window in windows:
            # Price features
            features_df[f'price_ma_{window}'] = features_df['Close'].rolling(window=window).mean()
            features_df[f'price_std_{window}'] = features_df['Close'].rolling(window=window).std()
            
            # Volume features
            features_df[f'volume_ma_{window}'] = features_df['Volume'].rolling(window=window).mean()
            features_df[f'volume_std_{window}'] = features_df['Volume'].rolling(window=window).std()
            
        return features_df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators (RSI, etc.).
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        features_df = df.copy()
        
        # RSI
        delta = features_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features_df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = features_df['Close'].ewm(span=26, adjust=False).mean()
        features_df['MACD'] = exp1 - exp2
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9, adjust=False).mean()
        
        return features_df
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create date-based features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with date features
        """
        features_df = df.copy()
        
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        
        return features_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df (pd.DataFrame): Raw input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        features_df = df.copy()
        
        # Create various features
        features_df = self.create_lag_features(features_df, lags=[1, 5, 10, 21])  # 1d, 1w, 2w, 1m
        features_df = self.create_rolling_features(features_df, windows=[5, 10, 21, 63])  # 1w, 2w, 1m, 3m
        features_df = self.create_technical_indicators(features_df)
        features_df = self.create_date_features(features_df)
        
        # Drop rows with NaN values (due to lagging/rolling operations)
        features_df = features_df.dropna()
        
        return features_df

class ModelPipeline:
    def __init__(self, model_dir: str):
        """
        Initialize model pipeline.
        
        Args:
            model_dir (str): Directory to save/load model artifacts
        """
        self.model_dir = Path(model_dir)
        self.feature_engineer = FeatureEngineer()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: X (features) and y (target)
        """
        # Engineer features
        features_df = self.feature_engineer.engineer_features(df)
        
        # Prepare target (next day's opening price)
        y = features_df['Open'].shift(-1)
        features_df = features_df.drop(columns=['Open'])  # Remove current day's open
        
        # Remove the last row (has NaN target)
        features_df = features_df[:-1]
        y = y[:-1]
        
        return features_df, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets by date.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
            pd.Series: Feature importance scores
        """
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
            name='importance'
        ).sort_values(ascending=False)
    
    def save_model(self):
        """Save model artifacts."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, self.model_dir / 'model.joblib')
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        
    def load_model(self):
        """Load saved model artifacts."""
        self.model = joblib.load(self.model_dir / 'model.joblib')
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')

if __name__ == '__main__':
    # Example usage
    from data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader('data')
    data = loader.merge_all_data()
    processed_data = loader.preprocess_data(data)
    
    # Initialize and train model
    pipeline = ModelPipeline('models')
    X, y = pipeline.prepare_features(processed_data)
    X_train, X_test, y_train, y_test = pipeline.train_test_split(X, y)
    
    # Train and save model
    pipeline.train(X_train, y_train)
    pipeline.save_model()
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    print(f"Made predictions for {len(predictions)} days")