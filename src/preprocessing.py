"""
Data Preprocessing Module for Boston Housing Dataset
This module handles data cleaning, scaling, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Class for preprocessing Boston Housing dataset."""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard', 'robust', 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
        # Initialize scaler based on type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'minmax'")
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='mean')
    
    def check_missing_values(self, data):
        """
        Check for missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Dictionary with missing value information
        """
        missing_info = {
            'total_missing': data.isnull().sum().sum(),
            'missing_per_column': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
        }
        
        return missing_info
    
    def check_outliers(self, data, method='iqr', threshold=1.5):
        """
        Check for outliers in numerical columns.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to detect outliers ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            dict: Dictionary with outlier information
        """
        outlier_info = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers = data[z_scores > threshold]
            
            outlier_info[column] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outlier_info
    
    def handle_outliers(self, data, method='iqr', threshold=1.5, strategy='clip'):
        """
        Handle outliers in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to detect outliers
            threshold (float): Threshold for outlier detection
            strategy (str): Strategy to handle outliers ('clip', 'remove', 'transform')
            
        Returns:
            pd.DataFrame: Data with handled outliers
        """
        data_clean = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == 'zscore':
                mean_val = data[column].mean()
                std_val = data[column].std()
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
            
            if strategy == 'clip':
                data_clean[column] = data_clean[column].clip(lower=lower_bound, upper=upper_bound)
            elif strategy == 'remove':
                mask = (data_clean[column] >= lower_bound) & (data_clean[column] <= upper_bound)
                data_clean = data_clean[mask]
            elif strategy == 'transform':
                # Log transformation for positive values
                if (data_clean[column] > 0).all():
                    data_clean[column] = np.log1p(data_clean[column])
        
        return data_clean
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features using the selected scaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            fit (bool): Whether to fit the scaler on training data
            
        Returns:
            tuple: Scaled training and test features
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.feature_names = X_train.columns.tolist()
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, features, target, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            features (pd.DataFrame): Feature matrix
            target (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features.
        
        Args:
            X (pd.DataFrame): Input features
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Features with polynomial terms
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Create feature names for polynomial features
        feature_names = poly.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_poly, columns=feature_names)
    
    def get_preprocessing_summary(self, original_data, processed_data):
        """
        Get a summary of preprocessing steps.
        
        Args:
            original_data (pd.DataFrame): Original data
            processed_data (pd.DataFrame): Processed data
            
        Returns:
            dict: Preprocessing summary
        """
        summary = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'scaler_type': self.scaler_type,
            'features_removed': len(original_data.columns) - len(processed_data.columns),
            'samples_removed': len(original_data) - len(processed_data)
        }
        
        return summary
    
    def plot_feature_distributions(self, data, columns=None, figsize=(15, 10)):
        """
        Plot distributions of numerical features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): List of columns to plot (if None, plot all numerical)
            figsize (tuple): Figure size
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if i < len(axes):
                axes[i].hist(data[column], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {column}')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data, figsize=(12, 10)):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Input data
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        correlation_matrix = data.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate preprocessing."""
    from data_loader import BostonHousingDataLoader
    
    # Load data
    loader = BostonHousingDataLoader()
    features, target, feature_names = loader.load_data()
    
    if features is not None:
        # Create preprocessor
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        # Check missing values
        missing_info = preprocessor.check_missing_values(features)
        print("Missing Values Info:")
        print(f"Total missing: {missing_info['total_missing']}")
        
        # Check outliers
        outlier_info = preprocessor.check_outliers(features)
        print("\nOutlier Info:")
        for col, info in outlier_info.items():
            print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        # Handle outliers
        features_clean = preprocessor.handle_outliers(features, strategy='clip')
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(features_clean, target)
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Get preprocessing summary
        summary = preprocessor.get_preprocessing_summary(features, features_clean)
        print("\nPreprocessing Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Plot distributions
        preprocessor.plot_feature_distributions(features_clean)
        
        # Plot correlation matrix
        preprocessor.plot_correlation_matrix(features_clean)

if __name__ == "__main__":
    main()
