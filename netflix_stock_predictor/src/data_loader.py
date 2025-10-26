"""
Data loader module for Netflix stock price prediction.
Handles loading, merging, and preprocessing of stock data from various CSV sources.
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_dir (str): Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        
    def load_stock_history(self) -> pd.DataFrame:
        """Load and process historical stock price data."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_history.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_splits(self) -> pd.DataFrame:
        """Load stock split events."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_spilts.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_actions(self) -> pd.DataFrame:
        """Load corporate actions data."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_action.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_info(self) -> pd.DataFrame:
        """Load company information and metrics."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_info.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_dividends(self) -> pd.DataFrame:
        """Load dividend events."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_dividends.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def adjust_for_splits(self, df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust historical prices for stock splits.
        
        Args:
            df (pd.DataFrame): Historical price data
            splits (pd.DataFrame): Split events data
            
        Returns:
            pd.DataFrame: Split-adjusted price data
        """
        adjusted_df = df.copy()
        
        # Apply split adjustments from newest to oldest
        for date, row in splits.sort_index(ascending=False).iterrows():
            split_ratio = row['SplitRatio']
            mask = adjusted_df.index < date
            
            # Adjust prices and volume
            price_columns = ['Open', 'High', 'Low', 'Close']
            adjusted_df.loc[mask, price_columns] = adjusted_df.loc[mask, price_columns] / split_ratio
            adjusted_df.loc[mask, 'Volume'] = adjusted_df.loc[mask, 'Volume'] * split_ratio
            
        return adjusted_df
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Load and merge all data sources with proper adjustments.
        
        Returns:
            pd.DataFrame: Complete merged dataset
        """
        # Load all data sources
        history_df = self.load_stock_history()
        splits_df = self.load_stock_splits()
        actions_df = self.load_stock_actions()
        info_df = self.load_stock_info()
        dividends_df = self.load_stock_dividends()
        
        # Adjust historical prices for splits
        adjusted_history = self.adjust_for_splits(history_df, splits_df)
        
        # Merge with other data sources
        merged_df = adjusted_history.copy()
        
        # Add corporate actions
        if not actions_df.empty:
            merged_df = merged_df.join(actions_df, how='left')
            
        # Add company info
        if not info_df.empty:
            merged_df = merged_df.join(info_df, how='left')
            
        # Add dividend information
        if not dividends_df.empty:
            merged_df = merged_df.join(dividends_df, how='left')
        
        # Forward fill missing values for company info
        info_columns = info_df.columns if not info_df.empty else []
        merged_df[info_columns] = merged_df[info_columns].ffill()
        
        return merged_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the merged dataset.
        
        Args:
            df (pd.DataFrame): Merged dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = processed_df.fillna(method='ffill')  # Forward fill
        processed_df = processed_df.fillna(method='bfill')  # Backward fill remaining NaNs
        
        # Convert to Eastern Time (US Market)
        if not processed_df.index.tz:
            processed_df.index = processed_df.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        # Remove rows outside market hours (if timestamp information is available)
        if processed_df.index.time.min() != processed_df.index.time.max():
            market_hours = (processed_df.index.time >= pd.Timestamp('9:30').time()) & \
                         (processed_df.index.time <= pd.Timestamp('16:00').time())
            processed_df = processed_df[market_hours]
        
        return processed_df

if __name__ == '__main__':
    # Example usage
    loader = DataLoader('data')
    merged_data = loader.merge_all_data()
    processed_data = loader.preprocess_data(merged_data)
    print(f"Loaded and processed {len(processed_data)} rows of data")