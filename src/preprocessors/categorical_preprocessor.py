from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, Any

class CategoricalPreprocessor(BasePreprocessor):
    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path)
        self.categorical_config = self.config['preprocessing']['categorical']
        self.encoders: Dict[str, Any] = {}
        self.encoding_maps: Dict[str, Dict] = {}
    
    def _handle_missing_values(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Handle missing values in categorical columns."""
        method = self.categorical_config['handle_missing']
        df = data.copy()
        
        for col in columns:
            if method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('MISSING', inplace=True)
        
        return df
    
    def _setup_encoder(self, column_name: str) -> None:
        """Setup the appropriate encoder based on configuration."""
        encoding_method = self.categorical_config['encoding']
        
        if encoding_method == 'label':
            self.encoders[column_name] = LabelEncoder()
        elif encoding_method == 'onehot':
            self.encoders[column_name] = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit encoders on categorical columns."""
        try:
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            categorical_columns = [col['name'] for col in self.config['dataset']['input_columns'] 
                                if col['type'] == 'categorical']
            
            # Handle missing values first
            data = self._handle_missing_values(data, categorical_columns)
            
            # Fit encoders for each categorical column
            for col in categorical_columns:
                self._setup_encoder(col)
                
                if isinstance(self.encoders[col], LabelEncoder):
                    self.encoders[col].fit(data[col])
                    # Store encoding mapping
                    self.encoding_maps[col] = dict(zip(
                        self.encoders[col].classes_,
                        self.encoders[col].transform(self.encoders[col].classes_)
                    ))
                else:  # OneHotEncoder
                    self.encoders[col].fit(data[col].values.reshape(-1, 1))
                    # Store feature names
                    self.encoding_maps[col] = {
                        'feature_names': self.encoders[col].get_feature_names_out([col]).tolist()
                    }
            
            self.logger.debug("Categorical preprocessor fit complete")
        except Exception as e:
            self.logger.error(f"Error in fit: {e}")
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders."""
        try:
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            df = data.copy()
            categorical_columns = [col['name'] for col in self.config['dataset']['input_columns'] 
                                if col['type'] == 'categorical']
            
            # Handle missing values first
            df = self._handle_missing_values(df, categorical_columns)
            
            # Transform each categorical column
            for col in categorical_columns:
                if col not in self.encoders:
                    self.logger.warning(f"No encoder found for column: {col}")
                    continue
                
                if isinstance(self.encoders[col], LabelEncoder):
                    df[col] = self.encoders[col].transform(df[col])
                else:  # OneHotEncoder
                    encoded_features = self.encoders[col].transform(df[col].values.reshape(-1, 1))
                    encoded_df = pd.DataFrame(
                        encoded_features,
                        columns=self.encoding_maps[col]['feature_names'],
                        index=df.index
                    )
                    # Drop original column and add encoded columns
                    df = df.drop(columns=[col])
                    df = pd.concat([df, encoded_df], axis=1)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            raise
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess categorical data."""
        return self.fit_transform(data) 