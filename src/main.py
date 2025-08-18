"""
Main Pipeline for Boston Housing Machine Learning Project
This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import os
import sys
import warnings
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from data_loader import BostonHousingDataLoader
from preprocessing import DataPreprocessor
from models import ModelTrainer
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'models', 'results', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory '{directory}' created/verified")

def main():
    """Main pipeline execution."""
    print("🏠 Boston Housing Machine Learning Project")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    try:
        # Step 1: Load Data
        print("\n📊 Step 1: Loading Boston Housing Dataset...")
        loader = BostonHousingDataLoader()
        features, target, feature_names = loader.load_data()
        
        if features is None:
            print("❌ Failed to load dataset. Exiting...")
            return
        
        # Save dataset to CSV
        loader.save_to_csv('data/boston_housing.csv')
        
        # Step 2: Data Preprocessing
        print("\n🔧 Step 2: Data Preprocessing...")
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        # Check data quality
        missing_info = preprocessor.check_missing_values(features)
        print(f"Missing values: {missing_info['total_missing']}")
        
        outlier_info = preprocessor.check_outliers(features)
        print(f"Outliers detected in {len([k for k, v in outlier_info.items() if v['count'] > 0])} features")
        
        # Handle outliers
        features_clean = preprocessor.handle_outliers(features, strategy='clip')
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(features_clean, target)
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Step 3: Model Training
        print("\n🤖 Step 3: Training Machine Learning Models...")
        trainer = ModelTrainer()
        
        # Train all models
        results = trainer.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        if not results:
            print("❌ No models were trained successfully. Exiting...")
            return
        
        # Step 4: Hyperparameter Tuning
        print("\n⚙️ Step 4: Hyperparameter Tuning...")
        
        # Tune Random Forest
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        trainer.hyperparameter_tuning(X_train_scaled, y_train, 'Random Forest', rf_param_grid)
        
        # Tune XGBoost
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        trainer.hyperparameter_tuning(X_train_scaled, y_train, 'XGBoost', xgb_param_grid)
        
        # Step 5: Feature Importance Analysis
        print("\n🔍 Step 5: Feature Importance Analysis...")
        
        # Get feature importance for tree-based models
        for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            if model_name in trainer.models:
                importance = trainer.get_feature_importance(model_name, feature_names)
                if importance:
                    print(f"\nTop 5 features for {model_name}:")
                    for i, (feature, score) in enumerate(list(importance.items())[:5]):
                        print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Step 6: Model Evaluation
        print("\n📈 Step 6: Comprehensive Model Evaluation...")
        evaluator = ModelEvaluator()
        
        # Evaluate all models
        for model_name, result in results.items():
            y_pred = result['y_pred']
            metrics = evaluator.calculate_metrics(y_test, y_pred, model_name)
            evaluator.print_metrics(model_name)
        
        # Step 7: Visualization and Analysis
        print("\n📊 Step 7: Creating Visualizations...")
        
        # Model comparison
        evaluator.compare_models(results)
        
        # Detailed analysis for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"R² Score: {best_result['r2']:.4f}")
        print(f"RMSE: {best_result['rmse']:.4f}")
        
        # Plot predictions vs actual for best model
        evaluator.plot_predictions_vs_actual(y_test, best_result['y_pred'], best_model_name)
        
        # Learning curves for best model
        evaluator.plot_learning_curves(best_result['model'], X_train_scaled, y_train)
        
        # Feature importance plot for best tree-based model
        if best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            trainer.plot_feature_importance(best_model_name)
        
        # Step 8: Save Results
        print("\n💾 Step 8: Saving Results...")
        
        # Save best model
        trainer.save_model(best_model_name, 'models/best_model.pkl')
        
        # Save evaluation report
        evaluator.create_evaluation_report('results/evaluation_report.txt')
        
        # Save model comparison results
        import json
        comparison_data = {}
        for model_name, result in results.items():
            comparison_data[model_name] = {
                'r2': float(result['r2']),
                'rmse': float(result['rmse']),
                'mae': float(result['mae'])
            }
        
        with open('results/model_comparison.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Step 9: Final Summary
        print("\n🎉 Pipeline Execution Completed Successfully!")
        print("=" * 50)
        print("📁 Generated Files:")
        print("  - data/boston_housing.csv")
        print("  - models/best_model.pkl")
        print("  - results/evaluation_report.txt")
        print("  - results/model_comparison.json")
        print("\n📊 Model Performance Summary:")
        
        # Print final comparison
        for model_name, result in results.items():
            print(f"  {model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {best_result['r2']:.4f}")
        print(f"   RMSE: {best_result['rmse']:.4f}")
        
        if best_result['r2'] > 0.8:
            print("   🎯 Excellent performance!")
        elif best_result['r2'] > 0.6:
            print("   👍 Good performance")
        else:
            print("   ⚠️ Performance needs improvement")
        
        print("\n🚀 Next Steps:")
        print("  - Review the evaluation report")
        print("  - Analyze feature importance")
        print("  - Consider feature engineering")
        print("  - Test on new data")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_quick_demo():
    """Run a quick demonstration of the pipeline."""
    print("🚀 Quick Demo Mode")
    print("=" * 30)
    
    try:
        # Load data
        loader = BostonHousingDataLoader()
        features, target, feature_names = loader.load_data()
        
        if features is None:
            return False
        
        # Quick preprocessing
        preprocessor = DataPreprocessor(scaler_type='standard')
        features_clean = preprocessor.handle_outliers(features, strategy='clip')
        X_train, X_test, y_train, y_test = preprocessor.split_data(features_clean, target)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Train only a few models for demo
        trainer = ModelTrainer()
        demo_models = ['Linear Regression', 'Random Forest', 'XGBoost']
        
        results = {}
        for model_name in demo_models:
            if model_name in trainer.get_models():
                model = trainer.get_models()[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                results[model_name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
        
        # Quick evaluation
        print("\nQuick Demo Results:")
        for model_name, result in results.items():
            print(f"  {model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Boston Housing ML Pipeline')
    parser.add_argument('--demo', action='store_true', help='Run quick demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        success = run_quick_demo()
    else:
        success = main()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
