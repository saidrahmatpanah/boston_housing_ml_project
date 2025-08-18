"""
Boston Housing Machine Learning Project

This package contains modules for:
- Data loading and preprocessing
- Model training and evaluation
- Comprehensive analysis and visualization

Modules:
- data_loader: Load and manage Boston Housing dataset
- preprocessing: Data cleaning, scaling, and preparation
- models: Machine learning model training and management
- evaluation: Model evaluation and visualization
- main: Main pipeline execution

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .data_loader import BostonHousingDataLoader
from .preprocessing import DataPreprocessor
from .models import ModelTrainer
from .evaluation import ModelEvaluator

__all__ = [
    'BostonHousingDataLoader',
    'DataPreprocessor', 
    'ModelTrainer',
    'ModelEvaluator'
]
