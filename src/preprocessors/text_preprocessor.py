from .base_preprocessor import BasePreprocessor
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

class TextPreprocessor(BasePreprocessor):
    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path)
        self.text_config = self.config['preprocessing']['text']
        self._download_nltk_resources()
        self.stemmer = PorterStemmer() if self.text_config['stemming'] else None
        
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.logger.debug("NLTK resources downloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to download NLTK resources: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Process a single text string."""
        try:
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            if self.text_config['lowercase']:
                text = text.lower()
            
            # Remove punctuation
            if self.text_config['remove_punctuation']:
                text = text.translate(str.maketrans("", "", string.punctuation))
            
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords
            if self.text_config['remove_stopwords']:
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            
            # Apply stemming
            if self.text_config['stemming']:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            return " ".join(tokens)
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return ""
    
    def fit(self, data: pd.DataFrame) -> None:
        """No fitting required for basic text preprocessing."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        self.logger.debug("Text preprocessor fit complete")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the text data."""
        try:
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            df = data.copy()
            text_columns = [col['name'] for col in self.config['dataset']['input_columns'] 
                          if col['type'] == 'text']
            
            for col in text_columns:
                self.logger.debug(f"Processing text column: {col}")
                df[col] = df[col].apply(self._preprocess_text)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            raise
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the text data."""
        return self.fit_transform(data) 