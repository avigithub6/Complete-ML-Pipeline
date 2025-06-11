import pandas as pd
import yaml
from typing import Dict, List
from .text_preprocessor import TextPreprocessor
from .numeric_preprocessor import NumericPreprocessor
from .categorical_preprocessor import CategoricalPreprocessor

class PreprocessorFactory:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.preprocessors = self._initialize_preprocessors()
    
    def _load_config(self) -> dict:
        """Load configuration from yaml file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_preprocessors(self) -> Dict:
        """Initialize preprocessors based on column types in config."""
        preprocessors = {}
        column_types = set(col['type'] for col in self.config['dataset']['input_columns'])
        
        if 'text' in column_types:
            preprocessors['text'] = TextPreprocessor(self.config_path)
        if 'numeric' in column_types:
            preprocessors['numeric'] = NumericPreprocessor(self.config_path)
        if 'categorical' in column_types:
            preprocessors['categorical'] = CategoricalPreprocessor(self.config_path)
        
        return preprocessors
    
    def _get_columns_by_type(self, data_type: str) -> List[str]:
        """Get column names for a specific data type."""
        return [col['name'] for col in self.config['dataset']['input_columns'] 
                if col['type'] == data_type]
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit all preprocessors on their respective columns."""
        for preprocessor_type, preprocessor in self.preprocessors.items():
            columns = self._get_columns_by_type(preprocessor_type)
            if columns:
                preprocessor.fit(data[columns])
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using all preprocessors."""
        df = data.copy()
        
        # Process each type of data separately
        processed_dfs = []
        for preprocessor_type, preprocessor in self.preprocessors.items():
            columns = self._get_columns_by_type(preprocessor_type)
            if columns:
                processed_df = preprocessor.transform(df[columns])
                processed_dfs.append(processed_df)
        
        # Combine all processed dataframes
        if processed_dfs:
            return pd.concat(processed_dfs, axis=1)
        return df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit all preprocessors and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def get_preprocessor(self, data_type: str):
        """Get a specific preprocessor by type."""
        return self.preprocessors.get(data_type) 