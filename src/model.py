"""
Modul training dan evaluasi model ML
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib


class H3N2Predictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names
 = None
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split dan scale data"""
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, **kwargs):
        """Training model XGBoost"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = XGBClassifier(**default_params)
        self.model.fit(X_train, y_train
)
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluasi model"""
        y_pred = self.model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions
': y_pred
        }
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model belum di-train")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, filepath):
        """Simpan model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names

        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model dari file"""
        data = joblib.load(filepath)
        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        return predictor
    
    def predict(self, X):
        """Prediksi untuk data baru"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)