"""
Test file for Boston Housing ML Project
Tests the functionality of various modules
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import BostonHousingDataLoader
from preprocessing import DataPreprocessor
from models import ModelTrainer
from evaluation import ModelEvaluator

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = BostonHousingDataLoader()
    
    def test_load_data(self):
        """Test data loading functionality."""
        features, target, feature_names = self.loader.load_data()
        
        self.assertIsNotNone(features)
        self.assertIsNotNone(target)
        self.assertIsNotNone(feature_names)
        self.assertEqual(features.shape[1], len(feature_names))
        self.assertEqual(len(features), len(target))
    
    def test_get_data_info(self):
        """Test data info retrieval."""
        self.loader.load_data()
        info = self.loader.get_data_info()
        
        self.assertIsNotNone(info)
        self.assertIn('shape', info)
        self.assertIn('feature_names', info)
        self.assertIn('target_name', info)

class TestPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.loader = BostonHousingDataLoader()
        self.features, self.target, _ = self.loader.load_data()
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        missing_info = self.preprocessor.check_missing_values(self.features)
        
        self.assertIsNotNone(missing_info)
        self.assertIn('total_missing', missing_info)
        self.assertIn('missing_per_column', missing_info)
    
    def test_check_outliers(self):
        """Test outlier detection."""
        outlier_info = self.preprocessor.check_outliers(self.features)
        
        self.assertIsNotNone(outlier_info)
        self.assertIsInstance(outlier_info, dict)
    
    def test_split_data(self):
        """Test data splitting functionality."""
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            self.features, self.target, test_size=0.2
        )
        
        self.assertEqual(len(X_train) + len(X_test), len(self.features))
        self.assertEqual(len(y_train) + len(y_test), len(self.target))
        self.assertAlmostEqual(len(X_test) / len(self.features), 0.2, places=1)

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        self.loader = BostonHousingDataLoader()
        self.preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        self.features, self.target, _ = self.loader.load_data()
        self.features_clean = self.preprocessor.handle_outliers(self.features, strategy='clip')
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split_data(
            self.features_clean, self.target
        )
        self.X_train_scaled, self.X_test_scaled = self.preprocessor.scale_features(
            self.X_train, self.X_test
        )
    
    def test_get_models(self):
        """Test model retrieval."""
        models = self.trainer.get_models()
        
        self.assertIsNotNone(models)
        self.assertGreater(len(models), 0)
        self.assertIn('Linear Regression', models)
        self.assertIn('Random Forest', models)
    
    def test_train_models(self):
        """Test model training."""
        # Train only a few models for testing
        test_models = ['Linear Regression', 'Random Forest']
        results = self.trainer.train_models(
            self.X_train_scaled, self.y_train, 
            self.X_test_scaled, self.y_test,
            models_to_train=test_models
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        for model_name, result in results.items():
            self.assertIn('r2', result)
            self.assertIn('rmse', result)
            self.assertIn('mae', result)

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred, 'Test Model')
        
        self.assertIsNotNone(metrics)
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
    
    def test_print_metrics(self):
        """Test metrics printing (should not raise errors)."""
        try:
            self.evaluator.calculate_metrics(self.y_true, self.y_pred, 'Test Model')
            self.evaluator.print_metrics('Test Model')
            # If we reach here, no errors occurred
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"print_metrics raised an exception: {e}")

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestModelTrainer))
    test_suite.addTest(unittest.makeSuite(TestModelEvaluator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 Running Boston Housing ML Project Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
