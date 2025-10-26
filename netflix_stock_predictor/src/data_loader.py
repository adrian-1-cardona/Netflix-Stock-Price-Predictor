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
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_splits(self) -> pd.DataFrame:
        """Load stock split events."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_spilts.csv')
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['SplitRatio'] = df['Stock Splits']  # Column is named 'Stock Splits' in the file
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_actions(self) -> pd.DataFrame:
        """Load corporate actions data."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_action.csv')
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        return df
    
    def load_stock_info(self) -> pd.DataFrame:
        """Load company information and metrics."""
        # The info CSV doesn't have time series data, so we don't need to process it
        # We'll handle it differently in merge_all_data
        return pd.DataFrame()
    
    def load_stock_dividends(self) -> pd.DataFrame:
        """Load dividend events."""
        df = pd.read_csv(self.data_dir / 'Netflix_stock_dividends.csv')
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
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
        
        if splits.empty:
            return adjusted_df
            
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
        # Load historical price data
        merged_df = self.load_stock_history()
        
        # Load and apply splits
        splits_df = self.load_stock_splits()
        if not splits_df.empty:
            merged_df = self.adjust_for_splits(merged_df, splits_df)
        
        # We already have dividends and splits in the history file
        # No need to merge actions_df or dividends_df as they contain the same information
        
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
        processed_df = processed_df.fillna(0)  # Fill missing values with 0
        
        # Convert to Eastern Time (US Market)
        processed_df.index = processed_df.index.tz_convert('US/Eastern')
        
        # Sort index to ensure chronological order
        processed_df = processed_df.sort_index()
        
        return processed_df

if __name__ == '__main__':
    # Example usage
    loader = DataLoader('data')
    merged_data = loader.merge_all_data()
    processed_data = loader.preprocess_data(merged_data)
    print(f"Loaded and processed {len(processed_data)} rows of data")