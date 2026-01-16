import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockSelectionModel:
    def __init__(self, model_type: str = 'random_forest', model_dir: str = 'models'):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model: Any = None
        self.feature_importance: Optional[np.ndarray] = None
        self.scaler: Any = None

    def create_model(self, params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {}

        default_params: Dict[str, Any] = {}
        
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **params}
            self.model = RandomForestClassifier(**params)

        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **params}
            self.model = xgb.XGBClassifier(**params)

        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            params = {**default_params, **params}
            self.model = lgb.LGBMClassifier(**params)

        elif self.model_type == 'logistic':
            default_params = {
                'random_state': 42,
                'max_iter': 1000
            }
            params = {**default_params, **params}
            self.model = LogisticRegression(**params)

        elif self.model_type == 'svm':
            default_params = {
                'random_state': 42,
                'probability': True
            }
            params = {**default_params, **params}
            self.model = SVC(**params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info(f"Created {self.model_type} model with params: {params}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        if self.model is None:
            self.create_model()

        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability prediction")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        y_pred = self.predict(X_test)
        
        metrics: Dict[str, Any] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }

        try:
            y_proba = self.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics['roc_auc'] = None

        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        if self.model is None:
            self.create_model()

        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }

    def save_model(self, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = f"{self.model_type}_model.pkl"

        filepath = self.model_dir / filename
        joblib.dump(self.model, filepath)
        
        if self.feature_importance is not None:
            importance_path = self.model_dir / f"{self.model_type}_feature_importance.pkl"
            joblib.dump(self.feature_importance, importance_path)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = f"{self.model_type}_model.pkl"

        filepath = self.model_dir / filename
        if filepath.exists():
            self.model = joblib.load(filepath)
            
            importance_path = self.model_dir / f"{self.model_type}_feature_importance.pkl"
            if importance_path.exists():
                self.feature_importance = joblib.load(importance_path)

            logger.info(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file {filepath} not found")

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(self.feature_importance)),
            'importance': self.feature_importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, top_n: int = 20) -> plt.Figure:
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")

        importance_df = self.get_feature_importance(feature_names)
        importance_df = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        return plt.gcf()


class EnsembleModel:
    def __init__(self, models: Optional[List[Dict[str, Any]]] = None, voting: str = 'soft'):
        self.models = models if models else []
        self.voting = voting

    def add_model(self, model: Any, weight: float = 1.0) -> None:
        self.models.append({'model': model, 'weight': weight})

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble")
            
        if self.voting == 'hard':
            # Hard voting with weights
            # Sum of weights for each class
            # Assuming binary classification for simplicity for now, or use max voting
            
            predictions = []
            weights = []
            
            for model_info in self.models:
                model = model_info['model']
                weight = model_info['weight']
                predictions.append(model.predict(X))
                weights.append(weight)
                
            predictions = np.array(predictions) # (n_models, n_samples)
            weights = np.array(weights) # (n_models,)
            
            # Weighted vote for each sample
            n_samples = X.shape[0]
            final_predictions = np.zeros(n_samples, dtype=int)
            
            # This is a simple implementation for binary/multiclass
            # Iterate over samples is safest but slow. Vectorized:
            
            # One-hot encode predictions? No, too complex.
            # Use bincount per column?
            
            for i in range(n_samples):
                sample_preds = predictions[:, i] # (n_models,)
                # bincount needs non-negative ints
                # We use weights.
                count = np.bincount(sample_preds.astype(int), weights=weights)
                final_predictions[i] = count.argmax()
                
            return final_predictions
            
        else:
            probas = []
            weights = []
            for model_info in self.models:
                model = model_info['model']
                weight = model_info['weight']
                # Check if predict_proba returns (N, 2)
                p = model.predict_proba(X)
                if p.ndim == 2:
                    p = p[:, 1] # Probability of class 1
                probas.append(p * weight)
                weights.append(weight)

            # Sum of weighted probabilities / sum of weights
            weighted_probas = np.array(probas).sum(axis=0) / sum(weights)
            final_prediction = (weighted_probas > 0.5).astype(int)

        return final_prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.voting == 'hard':
            raise ValueError("Probability prediction not available for hard voting")

        probas = []
        weights = []

        for model_info in self.models:
            model = model_info['model']
            weight = model_info['weight']
            p = model.predict_proba(X)
            if p.ndim == 2:
                p = p[:, 1]
            probas.append(p * weight)
            weights.append(weight)

        weighted_probas = np.array(probas).sum(axis=0) / sum(weights)

        # Return (N, 2) format
        return np.column_stack([1 - weighted_probas, weighted_probas])


class HyperparameterTuner:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127]
            }
        else:
            return {}

    def tune(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 3, scoring: str = 'accuracy', n_jobs: int = -1) -> Any:
        model = StockSelectionModel(model_type=self.model_type)
        model.create_model()

        param_grid = self.get_param_grid()

        if not param_grid:
            logger.warning(f"No parameter grid defined for {self.model_type}")
            return model.model

        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")

        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return grid_search.best_estimator_
