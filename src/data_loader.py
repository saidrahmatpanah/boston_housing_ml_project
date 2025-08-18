"""
Data Loader Module for Boston Housing Dataset
This module handles loading and basic exploration of the Boston Housing dataset.
"""

import pandas as pd
import numpy as np
import warnings

# Suppress warnings for deprecated load_boston
warnings.filterwarnings('ignore')

class BostonHousingDataLoader:
    """Class for loading and managing Boston Housing dataset."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.target_name = 'MEDV'
        
    def load_data(self):
        """
        Load California Housing dataset (reliable alternative to Boston Housing).
        
        Returns:
            tuple: (features, target, feature_names)
        """
        try:
            # Use California Housing dataset (more reliable and ethical)
            from sklearn.datasets import fetch_california_housing
            california = fetch_california_housing()
            
            # Create DataFrame
            self.features = pd.DataFrame(california.data, columns=california.feature_names)
            self.target = pd.Series(california.target, name='MEDV')
            self.feature_names = california.feature_names
            
            # Combine features and target
            self.data = pd.concat([self.features, self.target], axis=1)
            
            print(f"California Housing dataset loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Target: {self.target_name}")
            
            return self.features, self.target, self.feature_names
            
        except Exception as e:
            print(f"Error loading California Housing dataset: {e}")
            print("Trying Boston Housing dataset from original source...")
            
            # Fallback: Try Boston Housing dataset from original source
            try:
                data_url = "http://lib.stat.cmu.edu/datasets/boston"
                raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
                data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
                target = raw_df.values[1::2, 2]
                
                # Feature names for Boston Housing dataset
                feature_names = [
                    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
                ]
                
                # Create DataFrame
                self.features = pd.DataFrame(data, columns=feature_names)
                self.target = pd.Series(target, name=self.target_name)
                self.feature_names = feature_names
                
                # Combine features and target
                self.data = pd.concat([self.features, self.target], axis=1)
                
                print(f"Boston Housing dataset loaded from original source!")
                print(f"Shape: {self.data.shape}")
                print(f"Features: {len(self.feature_names)}")
                print(f"Target: {self.target_name}")
                
                return self.features, self.target, self.feature_names
                
            except Exception as e2:
                print(f"Error loading Boston Housing dataset: {e2}")
                return None, None, None
    
    def get_data_info(self):
        """
        Get basic information about the dataset.
        
        Returns:
            dict: Dictionary containing dataset information
        """
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return None
            
        info = {
            'shape': self.data.shape,
            'feature_names': list(self.feature_names),
            'target_name': self.target_name,
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def get_feature_descriptions(self):
        """
        Get descriptions of all features.
        
        Returns:
            dict: Dictionary mapping feature names to descriptions
        """
        descriptions = {
            'MedInc': 'میانگین درآمد خانوار در بلوک',
            'HouseAge': 'میانگین سن خانه در بلوک',
            'AveRooms': 'میانگین تعداد اتاق‌ها در هر خانه',
            'AveBedrms': 'میانگین تعداد اتاق خواب در هر خانه',
            'Population': 'جمعیت بلوک',
            'AveOccup': 'میانگین تعداد ساکنان در هر خانه',
            'Latitude': 'عرض جغرافیایی مرکز بلوک',
            'Longitude': 'طول جغرافیایی مرکز بلوک',
            'MEDV': 'ارزش متوسط خانه‌ها در 100,000 دلار (متغیر هدف)'
        }
        
        return descriptions
    
    def save_to_csv(self, filepath='data/boston_housing.csv'):
        """
        Save dataset to CSV file.
        
        Args:
            filepath (str): Path to save the CSV file
        """
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
            
        try:
            self.data.to_csv(filepath, index=False)
            print(f"Dataset saved to {filepath}")
        except Exception as e:
            print(f"Error saving dataset: {e}")
    
    def load_from_csv(self, filepath='data/boston_housing.csv'):
        """
        Load dataset from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            tuple: (features, target, feature_names)
        """
        try:
            self.data = pd.read_csv(filepath)
            self.features = self.data.drop(columns=[self.target_name])
            self.target = self.data[self.target_name]
            self.feature_names = self.features.columns.tolist()
            
            print(f"Dataset loaded from {filepath}")
            print(f"Shape: {self.data.shape}")
            
            return self.features, self.target, self.feature_names
            
        except Exception as e:
            print(f"Error loading from CSV: {e}")
            return None, None, None

def main():
    """Main function to demonstrate data loading."""
    # Create data loader instance
    loader = BostonHousingDataLoader()
    
    # Load data
    features, target, feature_names = loader.load_data()
    
    if features is not None:
        # Get dataset information
        info = loader.get_data_info()
        print("\nDataset Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Save to CSV
        loader.save_to_csv()
        
        # Get feature descriptions
        descriptions = loader.get_feature_descriptions()
        print("\nFeature Descriptions:")
        for feature, desc in descriptions.items():
            print(f"{feature}: {desc}")

if __name__ == "__main__":
    main()
