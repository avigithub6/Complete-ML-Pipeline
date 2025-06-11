from abc import ABC, abstractmethod
import pandas as pd
import yaml
import os

class BasePreprocessor(ABC):
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_logging()
    
    def load_config(self, config_path):
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration."""
        import logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel('DEBUG')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel('DEBUG')
        
        # File handler
        log_file = os.path.join(log_dir, f'{self.__class__.__name__}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel('DEBUG')
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the preprocessor on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted preprocessor."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data against configuration."""
        expected_columns = [col['name'] for col in self.config['dataset']['input_columns']]
        missing_columns = [col for col in expected_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            return False
        return True 