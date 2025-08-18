#!/usr/bin/env python3
"""
Boston Housing ML Project - Demo Script
This script demonstrates the main functionality of the project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    """Main demo function."""
    print("🏠 Boston Housing Machine Learning Project - Demo")
    print("=" * 60)
    
    try:
        # Import modules
        from data_loader import BostonHousingDataLoader
        from preprocessing import DataPreprocessor
        from models import ModelTrainer
        from evaluation import ModelEvaluator
        
        print("✅ All modules imported successfully!")
        
        # Step 1: Load Data
        print("\n📊 Step 1: Loading Boston Housing Dataset...")
        loader = BostonHousingDataLoader()
        features, target, feature_names = loader.load_data()
        
        if features is None:
            print("❌ Failed to load dataset")
            return False
        
        print(f"✅ Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Step 2: Data Preprocessing
        print("\n🔧 Step 2: Data Preprocessing...")
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        # Check data quality
        missing_info = preprocessor.check_missing_values(features)
        print(f"Missing values: {missing_info['total_missing']}")
        
        # Handle outliers
        features_clean = preprocessor.handle_outliers(features, strategy='clip')
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(features_clean, target)
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        print("✅ Data preprocessing completed!")
        
        # Step 3: Model Training (Quick Demo)
        print("\n🤖 Step 3: Training Models (Quick Demo)...")
        trainer = ModelTrainer()
        
        # Train only a few models for demo
        demo_models = ['Linear Regression', 'Random Forest', 'XGBoost']
        results = {}
        
        for model_name in demo_models:
            if model_name in trainer.get_models():
                print(f"  Training {model_name}...")
                model = trainer.get_models()[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate basic metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"    R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        if not results:
            print("❌ No models were trained successfully")
            return False
        
        # Step 4: Model Evaluation
        print("\n📈 Step 4: Model Evaluation...")
        evaluator = ModelEvaluator()
        
        for model_name, result in results.items():
            evaluator.calculate_metrics(y_test, result['y_pred'], model_name)
        
        # Step 5: Results Summary
        print("\n🏆 Demo Results Summary:")
        print("-" * 40)
        
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"Best Model: {best_model[0]}")
        print(f"R² Score: {best_model[1]['r2']:.4f}")
        print(f"RMSE: {best_model[1]['rmse']:.4f}")
        
        print(f"\nAll Models Performance:")
        for model_name, result in results.items():
            print(f"  {model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
