"""
Model Evaluation Module for Boston Housing Dataset
This module provides comprehensive evaluation and visualization capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Class for comprehensive model evaluation and visualization."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'model_name': model_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, model_name):
        """
        Print evaluation metrics for a specific model.
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results for {model_name}")
            return
        
        metrics = self.evaluation_results[model_name]
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Explained Variance Score: {metrics['explained_variance']:.4f}")
        print(f"Maximum Error: {metrics['max_error']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"{'='*50}")
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, figsize=(10, 8)):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Scatter plot: Predicted vs Actual
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Predicted vs Actual - {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'Residuals Plot - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax3.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Residuals Distribution - {model_name}')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title(f'Q-Q Plot of Residuals - {model_name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print(f"\nResiduals Statistics for {model_name}:")
        print(f"Mean: {np.mean(residuals):.4f}")
        print(f"Std: {np.std(residuals):.4f}")
        print(f"Skewness: {stats.skew(residuals):.4f}")
        print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    def plot_learning_curves(self, model, X_train, y_train, cv=5, figsize=(12, 5)):
        """
        Plot learning curves for a model.
        
        Args:
            model: Trained model
            X_train (np.array): Training features
            y_train (np.array): Training target
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Examples')
        plt.ylabel('R² Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_validation_curves(self, model, X_train, y_train, param_name, param_range, cv=5, figsize=(10, 6)):
        """
        Plot validation curves for hyperparameter tuning.
        
        Args:
            model: Model to evaluate
            X_train (np.array): Training features
            y_train (np.array): Training target
            param_name (str): Name of the parameter to vary
            param_range (list): Range of parameter values
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
        """
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name, param_range=param_range,
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('R² Score')
        plt.title(f'Validation Curves for {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_models(self, model_results, figsize=(15, 10)):
        """
        Compare multiple models using various metrics.
        
        Args:
            model_results (dict): Dictionary containing model results
            figsize (tuple): Figure size
        """
        if not model_results:
            print("No model results to compare")
            return
        
        # Extract metrics for comparison
        models = list(model_results.keys())
        metrics = ['r2', 'rmse', 'mae', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            
            if metric == 'r2':
                # Higher is better for R²
                colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
            else:
                # Lower is better for error metrics
                colors = ['green' if v < np.mean(values) else 'orange' if v < np.mean(values) * 1.2 else 'red' for v in values]
            
            bars = axes[i].bar(models, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def create_evaluation_report(self, output_file='evaluation_report.txt'):
        """
        Create a comprehensive evaluation report.
        
        Args:
            output_file (str): Path to save the report
        """
        if not self.evaluation_results:
            print("No evaluation results to report")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Boston Housing Dataset - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary table
            f.write("Model Performance Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'MAPE':<8}\n")
            f.write("-" * 60 + "\n")
            
            for model_name, metrics in self.evaluation_results.items():
                f.write(f"{model_name:<20} {metrics['r2']:<8.4f} {metrics['rmse']:<8.4f} "
                       f"{metrics['mae']:<8.4f} {metrics['mape']:<8.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Detailed results for each model
            for model_name, metrics in self.evaluation_results.items():
                f.write(f"Detailed Results for {model_name}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in metrics.items():
                    if metric != 'model_name':
                        f.write(f"{metric.upper()}: {value:.4f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            
            best_r2_model = max(self.evaluation_results.items(), key=lambda x: x[1]['r2'])
            best_rmse_model = min(self.evaluation_results.items(), key=lambda x: x[1]['rmse'])
            
            f.write(f"Best R² Score: {best_r2_model[0]} ({best_r2_model[1]['r2']:.4f})\n")
            f.write(f"Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['rmse']:.4f})\n")
            
            if best_r2_model[1]['r2'] > 0.8:
                f.write("Excellent model performance!\n")
            elif best_r2_model[1]['r2'] > 0.6:
                f.write("Good model performance.\n")
            else:
                f.write("Model performance needs improvement.\n")
        
        print(f"Evaluation report saved to {output_file}")
    
    def plot_residual_analysis(self, y_true, y_pred, model_name, figsize=(15, 10)):
        """
        Comprehensive residual analysis plots.
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Actual
        axes[1].scatter(y_true, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Actual')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        axes[2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[3])
        axes[3].set_title('Q-Q Plot of Residuals')
        axes[3].grid(True, alpha=0.3)
        
        # 5. Residuals vs Index
        axes[4].plot(residuals, alpha=0.6, color='purple')
        axes[4].axhline(y=0, color='red', linestyle='--')
        axes[4].set_xlabel('Sample Index')
        axes[4].set_ylabel('Residuals')
        axes[4].set_title('Residuals vs Index')
        axes[4].grid(True, alpha=0.3)
        
        # 6. Residuals boxplot
        axes[5].boxplot(residuals, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[5].set_ylabel('Residuals')
        axes[5].set_title('Residuals Boxplot')
        axes[5].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate evaluation capabilities."""
    from data_loader import BostonHousingDataLoader
    from preprocessing import DataPreprocessor
    from models import ModelTrainer
    
    # Load and preprocess data
    loader = BostonHousingDataLoader()
    features, target, feature_names = loader.load_data()
    
    if features is not None:
        # Preprocess data
        preprocessor = DataPreprocessor(scaler_type='standard')
        features_clean = preprocessor.handle_outliers(features, strategy='clip')
        X_train, X_test, y_train, y_test = preprocessor.split_data(features_clean, target)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        
        for model_name, result in results.items():
            y_pred = result['y_pred']
            metrics = evaluator.calculate_metrics(y_test, y_pred, model_name)
            evaluator.print_metrics(model_name)
            
            # Plot predictions vs actual
            evaluator.plot_predictions_vs_actual(y_test, y_pred, model_name)
            
            # Plot learning curves for Random Forest
            if model_name == 'Random Forest':
                evaluator.plot_learning_curves(result['model'], X_train_scaled, y_train)
        
        # Compare all models
        evaluator.compare_models(results)
        
        # Create evaluation report
        evaluator.create_evaluation_report()
        
        # Comprehensive residual analysis for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        evaluator.plot_residual_analysis(y_test, best_result['y_pred'], best_model_name)

if __name__ == "__main__":
    main()
