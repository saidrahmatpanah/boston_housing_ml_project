"""
Configuration file for Boston Housing ML Project
Contains various settings and parameters for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATASET_NAME = "boston_housing"
TARGET_COLUMN = "MEDV"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing configuration
SCALER_TYPE = "standard"  # Options: "standard", "robust", "minmax"
OUTLIER_METHOD = "iqr"    # Options: "iqr", "zscore"
OUTLIER_THRESHOLD = 1.5
OUTLIER_STRATEGY = "clip"  # Options: "clip", "remove", "transform"

# Model configuration
MODELS_TO_TRAIN = [
    "Linear Regression",
    "Ridge Regression", 
    "Lasso Regression",
    "Elastic Net",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "SVR",
    "KNN",
    "Decision Tree"
]

# Hyperparameter tuning configuration
CROSS_VALIDATION_FOLDS = 5
N_JOBS = -1  # Use all available CPU cores

# Random Forest hyperparameters
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost hyperparameters
XGB_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Gradient Boosting hyperparameters
GB_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Evaluation configuration
EVALUATION_METRICS = [
    'r2', 'rmse', 'mae', 'mape', 
    'explained_variance', 'max_error'
]

# Visualization configuration
FIGURE_SIZE = (12, 8)
DPI = 100
COLOR_PALETTE = "husl"
STYLE = "seaborn-v0_8"

# Feature importance configuration
TOP_N_FEATURES = 15

# File paths
DATA_FILE = DATA_DIR / "boston_housing.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
TRAINING_RESULTS_FILE = RESULTS_DIR / "training_results.json"
EVALUATION_REPORT_FILE = RESULTS_DIR / "evaluation_report.txt"
EXPLORATION_RESULTS_FILE = RESULTS_DIR / "exploration_results.json"
PREPROCESSED_DATA_FILE = RESULTS_DIR / "preprocessed_data.pkl"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = RESULTS_DIR / "project.log"

# Performance thresholds
EXCELLENT_R2_THRESHOLD = 0.8
GOOD_R2_THRESHOLD = 0.6
FAIR_R2_THRESHOLD = 0.4

# Feature descriptions (Persian/Farsi)
FEATURE_DESCRIPTIONS = {
    'CRIM': 'نرخ جرم و جنایت در هر شهر',
    'ZN': 'نسبت زمین‌های مسکونی با قطعات بزرگتر از 25,000 فوت مربع',
    'INDUS': 'نسبت کسب و کار‌های غیر خرده‌فروشی در هر شهر',
    'CHAS': 'متغیر مجازی رودخانه چارلز (1 اگر در کنار رودخانه، 0 در غیر این صورت)',
    'NOX': 'غلظت اکسیدهای نیتروژن (قسمت در 10 میلیون)',
    'RM': 'میانگین تعداد اتاق‌ها در هر خانه',
    'AGE': 'نسبت خانه‌های اشغال شده قبل از 1940',
    'DIS': 'میانگین فاصله تا پنج مرکز اشتغال بوستون',
    'RAD': 'شاخص دسترسی به بزرگراه‌های شعاعی',
    'TAX': 'نرخ مالیات بر دارایی کامل در هر 10,000 دلار',
    'PTRATIO': 'نسبت دانش‌آموز به معلم در هر شهر',
    'B': '1000(Bk - 0.63)²، که Bk نسبت سیاه‌پوستان در هر شهر است',
    'LSTAT': 'درصد پایین‌تر از وضعیت در جامعه',
    'MEDV': 'ارزش متوسط خانه‌ها در 1000 دلار (متغیر هدف)'
}

# Model performance categories
PERFORMANCE_CATEGORIES = {
    'excellent': 'R² ≥ 0.8 - عملکرد عالی',
    'good': '0.6 ≤ R² < 0.8 - عملکرد خوب',
    'fair': '0.4 ≤ R² < 0.6 - عملکرد متوسط',
    'poor': 'R² < 0.4 - عملکرد ضعیف'
}

# Environment variables
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Export configuration
def get_config():
    """Get all configuration as a dictionary."""
    config = {}
    for key, value in globals().items():
        if not key.startswith('_') and key.isupper():
            config[key] = value
    return config

def print_config():
    """Print current configuration."""
    print("🔧 Boston Housing ML Project Configuration:")
    print("=" * 50)
    
    config = get_config()
    for key, value in config.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_config()
