"""
Machine Learning Models Module for Boston Housing Dataset
This module implements various regression models and their training.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class for training and managing machine learning models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.model_scores = {}
        self.feature_importance = {}
        
    def get_models(self):
        """
        Get a dictionary of all available models.
        
        Returns:
            dict: Dictionary of model names and instances
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        return models
    
    def train_models(self, X_train, y_train, X_test, y_test, models_to_train=None):
        """
        Train multiple models and evaluate their performance.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features
            y_test (np.array): Test target
            models_to_train (list): List of model names to train (if None, train all)
            
        Returns:
            dict: Dictionary with model performance metrics
        """
        models = self.get_models()
        
        if models_to_train:
            models = {k: v for k, v in models.items() if k in models_to_train}
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'y_pred': y_pred
                }
                
                # Store model
                self.models[name] = model
                
                # Update best model
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = model
                
                print(f"  R² Score: {r2:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        self.model_scores = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name, param_grid, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            model_name (str): Name of the model to tune
            param_grid (dict): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and score
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Train it first.")
            return None
        
        model = self.models[model_name]
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            
        Returns:
            dict: Feature importance dictionary
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print(f"Model {model_name} doesn't support feature importance.")
            return None
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        self.feature_importance[model_name] = feature_importance
        return feature_importance
    
    def plot_model_comparison(self, figsize=(12, 8)):
        """
        Plot comparison of model performances.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.model_scores:
            print("No models trained yet.")
            return
        
        # Extract metrics
        models = list(self.model_scores.keys())
        r2_scores = [self.model_scores[model]['r2'] for model in models]
        rmse_scores = [self.model_scores[model]['rmse'] for model in models]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # R² scores
        bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model R² Scores Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE scores
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model RMSE Scores Comparison')
        ax2.set_ylabel('RMSE')
        
        # Add value labels on bars
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}.")
            return
        
        importance = self.feature_importance[model_name]
        
        # Get top N features
        top_features = dict(list(importance.items())[:top_n])
        
        plt.figure(figsize=figsize)
        
        # Create horizontal bar plot
        features = list(top_features.keys())
        scores = list(top_features.values())
        
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores, color='lightgreen', alpha=0.7)
        
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, score in enumerate(scores):
            plt.text(score + 0.001, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        try:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, model_name, filepath):
        """
        Load a model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to the saved model
        """
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            print(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, model_name, X):
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model to use
            X (np.array): Features for prediction
            
        Returns:
            np.array: Predictions
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        try:
            predictions = self.models[model_name].predict(X)
            return predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def get_best_model_info(self):
        """
        Get information about the best performing model.
        
        Returns:
            dict: Information about the best model
        """
        if self.best_model is None:
            return None
        
        # Find the name of the best model
        best_model_name = None
        for name, model in self.models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        if best_model_name:
            return {
                'name': best_model_name,
                'model': self.best_model,
                'r2_score': self.best_score,
                'metrics': self.model_scores[best_model_name]
            }
        
        return None

def main():
    """Main function to demonstrate model training."""
    from data_loader import BostonHousingDataLoader
    from preprocessing import DataPreprocessor
    
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
        
        # Hyperparameter tuning for Random Forest
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        trainer.hyperparameter_tuning(X_train_scaled, y_train, 'Random Forest', rf_param_grid)
        
        # Get feature importance
        trainer.get_feature_importance('Random Forest', feature_names)
        
        # Plot results
        trainer.plot_model_comparison()
        trainer.plot_feature_importance('Random Forest')
        
        # Get best model info
        best_info = trainer.get_best_model_info()
        if best_info:
            print(f"\nBest Model: {best_info['name']}")
            print(f"R² Score: {best_info['r2_score']:.4f}")
        
        # Save best model
        trainer.save_model('Random Forest', 'models/best_model.pkl')

if __name__ == "__main__":
    main()
