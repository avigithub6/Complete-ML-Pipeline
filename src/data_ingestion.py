import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
from preprocessors.preprocessor_factory import PreprocessorFactory

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('data_ingestion')
    logger.setLevel('DEBUG')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')
    
    log_file_path = os.path.join(log_dir, 'data_ingestion.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel('DEBUG')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_data(data_path: str, logger) -> pd.DataFrame:
    """Load data from specified path."""
    try:
        df = pd.read_csv(data_path)
        logger.debug(f'Data loaded from {data_path}')
        return df
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def process_data(df: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """Process data according to configuration."""
    try:
        # Initialize preprocessor factory
        preprocessor = PreprocessorFactory(config_path='config.yaml')
        
        # Process the data
        processed_df = preprocessor.fit_transform(df)
        logger.debug('Data preprocessing completed')
        
        return processed_df
    except Exception as e:
        logger.error(f'Error processing data: {e}')
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int, logger) -> tuple:
    """Split data into train and test sets."""
    try:
        train_data, test_data = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state
        )
        logger.debug('Data split into train and test sets')
        return train_data, test_data
    except Exception as e:
        logger.error(f'Error splitting data: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str, logger) -> None:
    """Save processed data."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.debug(f'Data saved to {output_dir}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

def main():
    """Main function to orchestrate data ingestion process."""
    try:
        # Setup logging
        logger = setup_logging()
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get parameters
        data_path = config.get('data_path', 'data/raw/input.csv')
        test_size = config['dataset'].get('test_size', 0.2)
        random_state = config['dataset'].get('random_state', 42)
        output_dir = config.get('output_dir', 'data/processed')
        
        # Load data
        df = load_data(data_path, logger)
        
        # Process data
        processed_df = process_data(df, config, logger)
        
        # Split data
        train_data, test_data = split_data(processed_df, test_size, random_state, logger)
        
        # Save processed data
        save_data(train_data, test_data, output_dir, logger)
        
        logger.info('Data ingestion completed successfully')
        
    except Exception as e:
        logger.error(f'Data ingestion failed: {e}')
        raise

if __name__ == '__main__':
    main()