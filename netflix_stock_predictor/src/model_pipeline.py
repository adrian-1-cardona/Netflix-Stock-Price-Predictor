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
        # Create separate models for open, high, and close predictions
        self.model_open = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model_high = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=43
        )
        self.model_close = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=44
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare features and target variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.Series]]: X (features) and y (targets for open, high, close)
        """
        # Engineer features
        features_df = self.feature_engineer.engineer_features(df)
        
        # Prepare targets (next day's open, high, and close)
        y_open = features_df['Open'].shift(-1)
        y_high = features_df['High'].shift(-1)
        y_close = features_df['Close'].shift(-1)
        
        # Remove target columns from features
        features_df = features_df.drop(columns=['Open', 'High', 'Low', 'Close'])
        
        # Remove the last row (has NaN targets)
        features_df = features_df[:-1]
        y_open = y_open[:-1]
        y_high = y_high[:-1]
        y_close = y_close[:-1]
        
        targets = {
            'open': y_open,
            'high': y_high,
            'close': y_close
        }
        
        return features_df, targets
    
    def train_test_split(self, X: pd.DataFrame, targets: Dict[str, pd.Series], test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets by date.
        
        Args:
            X (pd.DataFrame): Feature matrix
            targets (Dict[str, pd.Series]): Target variables
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple: (X_train, X_test, targets_train, targets_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        
        targets_train = {k: v.iloc[:split_idx] for k, v in targets.items()}
        targets_test = {k: v.iloc[split_idx:] for k, v in targets.items()}
        
        return X_train, X_test, targets_train, targets_test
    
    def train(self, X_train: pd.DataFrame, targets_train: Dict[str, pd.Series]):
        """
        Train the models.
        
        Args:
            X_train (pd.DataFrame): Training features
            targets_train (Dict[str, pd.Series]): Training targets
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train models
        print("Training opening price model...")
        self.model_open.fit(X_train_scaled, targets_train['open'])
        
        print("Training high price model...")
        self.model_high.fit(X_train_scaled, targets_train['high'])
        
        print("Training closing price model...")
        self.model_close.fit(X_train_scaled, targets_train['close'])
        
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            Dict[str, np.ndarray]: Predicted values for open, high, and close
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = {
            'open': self.model_open.predict(X_scaled),
            'high': self.model_high.predict(X_scaled),
            'close': self.model_close.predict(X_scaled)
        }
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
        """
        Make predictions with confidence intervals.
        
        Args:
            X (pd.DataFrame): Features to predict on
            confidence_level (float): Confidence level (default: 0.95 for 95%)
            
        Returns:
            Dict: Predictions with confidence intervals and uncertainty measures
        """
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        for target, model in [('open', self.model_open), ('high', self.model_high), ('close', self.model_close)]:
            # Get predictions from all trees in the forest
            tree_predictions = np.array([tree.predict(X_scaled) for tree in model.estimators_])
            
            # Calculate mean prediction and standard deviation
            mean_pred = np.mean(tree_predictions, axis=0)
            std_pred = np.std(tree_predictions, axis=0)
            
            # Calculate confidence interval
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * std_pred
            
            # Calculate confidence percentage (inverse of coefficient of variation)
            # Higher values = more confidence
            confidence_pct = 100 * (1 - (std_pred / mean_pred))
            confidence_pct = np.clip(confidence_pct, 0, 100)  # Ensure 0-100 range
            
            results[target] = {
                'prediction': mean_pred[0],
                'lower_bound': mean_pred[0] - margin[0],
                'upper_bound': mean_pred[0] + margin[0],
                'std_dev': std_pred[0],
                'confidence': confidence_pct[0]
            }
        
        return results
    
    def get_feature_importance(self, model_name: str = 'open') -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            model_name (str): Which model to get importance from ('open', 'high', 'close')
            
        Returns:
            pd.Series: Feature importance scores
        """
        model = {
            'open': self.model_open,
            'high': self.model_high,
            'close': self.model_close
        }[model_name]
        
        return pd.Series(
            model.feature_importances_,
            index=self.feature_names_,
            name='importance'
        ).sort_values(ascending=False)
    
    def save_model(self):
        """Save model artifacts."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model_open, self.model_dir / 'model_open.joblib')
        joblib.dump(self.model_high, self.model_dir / 'model_high.joblib')
        joblib.dump(self.model_close, self.model_dir / 'model_close.joblib')
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        
    def load_model(self):
        """Load saved model artifacts."""
        self.model_open = joblib.load(self.model_dir / 'model_open.joblib')
        self.model_high = joblib.load(self.model_dir / 'model_high.joblib')
        self.model_close = joblib.load(self.model_dir / 'model_close.joblib')
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
    X, targets = pipeline.prepare_features(processed_data)
    X_train, X_test, targets_train, targets_test = pipeline.train_test_split(X, targets)
    
    # Train and save model
    pipeline.train(X_train, targets_train)
    pipeline.save_model()
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    print(f"Made predictions for {len(predictions['open'])} days")
    