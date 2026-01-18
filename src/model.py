"""
Machine Learning Model for H3N2 Antigenic Prediction
=====================================================
XGBoost-based classifier for predicting antigenic drift patterns.

Author: PKM-RE Team (Syifa & Rofi)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier
import joblib


class H3N2Predictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.classes_ = None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split and scale data"""
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Encode labels if string
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
        else:
            y_encoded = y
            self.classes_ = np.unique(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, **kwargs):
        """Train XGBoost model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        default_params.update(kwargs)
        
        self.model = XGBClassifier(**default_params)
        self.model.fit(X_train, y_train)
        return self
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        X_scaled = self.scaler.fit_transform(X)
        
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
        
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }

    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance ranking"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importance))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance[:top_n])
        
        return importance
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        data = joblib.load(filepath)
        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.label_encoder = data.get('label_encoder', LabelEncoder())
        predictor.feature_names = data['feature_names']
        predictor.classes_ = data.get('classes', None)
        return predictor
    
    def predict(self, X):
        """Predict for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if self.classes_ is not None and hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
