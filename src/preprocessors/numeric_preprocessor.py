from .base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Any

class NumericPreprocessor(BasePreprocessor):
    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path)
        self.numeric_config = self.config['preprocessing']['numeric']
        self.scalers: Dict[str, Any] = {}
        self._setup_scalers()
    
    def _setup_scalers(self):
        """Initialize the appropriate scaler based on configuration."""
        scaling_method = self.numeric_config['scaling']
        if scaling_method == 'standard':
            self.scaler_class = StandardScaler
        elif scaling_method == 'minmax':
            self.scaler_class = MinMaxScaler
        elif scaling_method == 'robust':
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def _handle_missing_values(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Handle missing values in numeric columns."""
        method = self.numeric_config['handle_missing']
        df = data.copy()
        
        for col in columns:
            if method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                raise ValueError(f"Unknown missing value handling method: {method}")
        
        return df
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit scalers on numeric columns."""
        try:
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            numeric_columns = [col['name'] for col in self.config['dataset']['input_columns'] 
                             if col['type'] == 'numeric']
            
            # Handle missing values first
            data = self._handle_missing_values(data, numeric_columns)
            
            # Fit scalers for each numeric column
            for col in numeric_columns:
                self.scalers[col] = self.scaler_class()
                self.scalers[col].fit(data[col].values.reshape(-1, 1))
            
            self.logger.debug("Numeric preprocessor fit complete")
        except Exception as e:
            self.logger.error(f"Error in fit: {e}")
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric columns using fitted scalers."""
        try:
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            df = data.copy()
            numeric_columns = [col['name'] for col in self.config['dataset']['input_columns'] 
                             if col['type'] == 'numeric']
            
            # Handle missing values first
            df = self._handle_missing_values(df, numeric_columns)
            
            # Transform each numeric column
            for col in numeric_columns:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1))
                else:
                    self.logger.warning(f"No scaler found for column: {col}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            raise
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numeric data."""
        return self.fit_transform(data) 