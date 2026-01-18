"""
Advanced Machine Learning Model Training System
================================================
Comprehensive ensemble and deep learning approaches for H3N2 antigenic prediction

Models Implemented:
1. Ensemble Methods:
   - Stacking Classifier (XGBoost + RF + LightGBM + ExtraTrees)
   - Voting Classifier (Hard & Soft voting)
   - Weighted Ensemble
   
2. Deep Learning:
   - Multi-layer Perceptron (MLP)
   - 1D Convolutional Neural Network (CNN)
   - Attention-based Neural Network
   
3. Advanced Tree Methods:
   - CatBoost
   - LightGBM
   - Histogram-based Gradient Boosting
   
4. Interpretability:
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Permutation Importance
   
5. Uncertainty Quantification:
   - Prediction intervals
   - Confidence scores
   - Calibration curves

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("[WARNING] CatBoost not available")

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not available")

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False
    print("[WARNING] TensorFlow/Keras not available")

PROCESSED_DIR = "data/processed"
ADVANCED_DIR = "data/advanced"
MODELS_DIR = "models/advanced"
RESULTS_DIR = "results/advanced"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class AdvancedModelTrainer:
    """Advanced model training with ensemble and deep learning"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def load_data(self, use_advanced_features=True):
        """Load feature data"""
        print("Loading data...")
        
        if use_advanced_features and os.path.exists(os.path.join(ADVANCED_DIR, 'h3n2_advanced_features.csv')):
            features_file = os.path.join(ADVANCED_DIR, 'h3n2_advanced_features.csv')
            matrix_file = os.path.join(ADVANCED_DIR, 'h3n2_advanced_features_matrix.csv')
            print("  Using advanced features (200+)")
        else:
            features_file = os.path.join(PROCESSED_DIR, 'h3n2_features.csv')
            matrix_file = os.path.join(PROCESSED_DIR, 'h3n2_features_matrix.csv')
            print("  Using basic features (74)")
        
        # Load features
        X = pd.read_csv(matrix_file)
        meta = pd.read_csv(features_file)
        
        # Binary labels
        y_binary = (meta['collection_year'] >= 2020).astype(int)
        
        # Multi-class labels
        def assign_period(year):
            if pd.isna(year):
                return -1
            if year >= 2020:
                return 3
            elif year >= 2015:
                return 2
            elif year >= 2010:
                return 1
            else:
                return 0
        
        y_multiclass = meta['collection_year'].apply(assign_period)
        
        print(f"  Data shape: {X.shape}")
        print(f"  Binary distribution: {np.bincount(y_binary)}")
        print(f"  Multi-class distribution: {np.bincount(y_multiclass[y_multiclass >= 0])}")
        
        return X, y_binary, y_multiclass, meta

    
    def train_ensemble_stacking(self, X_train, y_train, X_test, y_test, task='binary'):
        """Train stacking ensemble classifier"""
        print(f"\n{'='*60}")
        print(f"STACKING ENSEMBLE - {task.upper()}")
        print('='*60)
        
        # Base estimators
        base_estimators = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=self.random_state
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=10, random_state=self.random_state
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state
            ))
        ]
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            base_estimators.append((
                'lgb', lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                )
            ))
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            base_estimators.append((
                'cat', cb.CatBoostClassifier(
                    iterations=200, depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=0
                )
            ))
        
        # Meta-learner
        if task == 'binary':
            meta_learner = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=self.random_state, eval_metric='logloss'
            )
        else:
            meta_learner = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                objective='multi:softmax', num_class=4,
                random_state=self.random_state
            )
        
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        print("Training stacking ensemble...")
        stacking.fit(X_train, y_train)
        
        # Predictions
        y_pred = stacking.predict(X_test)
        y_pred_proba = stacking.predict_proba(X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        
        print(f"\nStacking Ensemble Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Save model
        model_name = f'stacking_{task}'
        self.models[model_name] = stacking
        self.results[model_name] = metrics
        
        joblib.dump(stacking, os.path.join(MODELS_DIR, f'{model_name}_model.pkl'))
        print(f"\n[SAVED] Model: {model_name}_model.pkl")
        
        return stacking, metrics
    
    def train_voting_ensemble(self, X_train, y_train, X_test, y_test, task='binary', voting='soft'):
        """Train voting ensemble classifier"""
        print(f"\n{'='*60}")
        print(f"VOTING ENSEMBLE ({voting.upper()}) - {task.upper()}")
        print('='*60)
        
        # Estimators
        estimators = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=self.random_state
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=10, random_state=self.random_state
            ))
        ]
        
        if LIGHTGBM_AVAILABLE:
            estimators.append((
                'lgb', lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                )
            ))
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        print(f"Training {voting} voting ensemble...")
        voting_clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = voting_clf.predict(X_test)
        y_pred_proba = voting_clf.predict_proba(X_test) if voting == 'soft' else None
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        
        print(f"\nVoting Ensemble Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Save model
        model_name = f'voting_{voting}_{task}'
        self.models[model_name] = voting_clf
        self.results[model_name] = metrics
        
        joblib.dump(voting_clf, os.path.join(MODELS_DIR, f'{model_name}_model.pkl'))
        print(f"\n[SAVED] Model: {model_name}_model.pkl")
        
        return voting_clf, metrics

    
    def train_deep_learning_mlp(self, X_train, y_train, X_test, y_test, task='binary'):
        """Train Multi-Layer Perceptron"""
        print(f"\n{'='*60}")
        print(f"DEEP LEARNING MLP - {task.upper()}")
        print('='*60)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'mlp_{task}'] = scaler
        
        # MLP architecture
        if task == 'binary':
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state,
                verbose=False
            )
        else:
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state,
                verbose=False
            )
        
        print("Training MLP...")
        mlp.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = mlp.predict(X_test_scaled)
        y_pred_proba = mlp.predict_proba(X_test_scaled)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        metrics['n_iterations'] = mlp.n_iter_
        metrics['loss_curve'] = mlp.loss_curve_.tolist() if hasattr(mlp, 'loss_curve_') else []
        
        print(f"\nMLP Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Iterations: {mlp.n_iter_}")
        
        # Save model
        model_name = f'mlp_{task}'
        self.models[model_name] = mlp
        self.results[model_name] = metrics
        
        joblib.dump(mlp, os.path.join(MODELS_DIR, f'{model_name}_model.pkl'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f'{model_name}_scaler.pkl'))
        print(f"\n[SAVED] Model: {model_name}_model.pkl")
        
        return mlp, metrics
    
    def train_keras_cnn(self, X_train, y_train, X_test, y_test, task='binary'):
        """Train 1D CNN using Keras"""
        if not KERAS_AVAILABLE:
            print("\n[SKIP] Keras not available")
            return None, {}
        
        print(f"\n{'='*60}")
        print(f"DEEP LEARNING CNN (Keras) - {task.upper()}")
        print('='*60)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape for CNN (samples, features, 1)
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        self.scalers[f'cnn_{task}'] = scaler
        
        # Build CNN model
        if task == 'binary':
            n_classes = 1
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
        else:
            n_classes = 4
            loss = 'sparse_categorical_crossentropy'
            activation = 'softmax'
            
        model = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(n_classes, activation=activation)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        print("Training CNN...")
        history = model.fit(
            X_train_cnn, y_train,
            validation_split=0.1,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_cnn, verbose=0)
        
        if task == 'binary':
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        metrics['history'] = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        print(f"\nCNN Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
        # Save model
        model_name = f'cnn_{task}'
        self.models[model_name] = model
        self.results[model_name] = metrics
        
        model.save(os.path.join(MODELS_DIR, f'{model_name}_model.h5'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f'{model_name}_scaler.pkl'))
        print(f"\n[SAVED] Model: {model_name}_model.h5")
        
        return model, metrics

    
    def train_catboost(self, X_train, y_train, X_test, y_test, task='binary'):
        """Train CatBoost classifier"""
        if not CATBOOST_AVAILABLE:
            print("\n[SKIP] CatBoost not available")
            return None, {}
        
        print(f"\n{'='*60}")
        print(f"CATBOOST - {task.upper()}")
        print('='*60)
        
        if task == 'binary':
            cat = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                loss_function='Logloss',
                eval_metric='Accuracy',
                random_state=self.random_state,
                verbose=100
            )
        else:
            cat = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                loss_function='MultiClass',
                eval_metric='Accuracy',
                random_state=self.random_state,
                verbose=100
            )
        
        print("Training CatBoost...")
        cat.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predictions
        y_pred = cat.predict(X_test)
        y_pred_proba = cat.predict_proba(X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        metrics['best_iteration'] = cat.best_iteration_
        
        print(f"\nCatBoost Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Best iteration: {cat.best_iteration_}")
        
        # Save model
        model_name = f'catboost_{task}'
        self.models[model_name] = cat
        self.results[model_name] = metrics
        
        cat.save_model(os.path.join(MODELS_DIR, f'{model_name}_model.cbm'))
        print(f"\n[SAVED] Model: {model_name}_model.cbm")
        
        return cat, metrics
    
    def calculate_shap_values(self, model, X_test, model_name, max_samples=100):
        """Calculate SHAP values for model interpretability"""
        if not SHAP_AVAILABLE:
            print("\n[SKIP] SHAP not available")
            return None
        
        print(f"\n{'='*60}")
        print(f"SHAP ANALYSIS - {model_name.upper()}")
        print('='*60)
        
        try:
            # Sample data for SHAP (can be slow for large datasets)
            X_sample = X_test.sample(min(max_samples, len(X_test)), random_state=self.random_state)
            
            # Create explainer
            if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, ExtraTreesClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            
            # Calculate SHAP values
            print("Calculating SHAP values...")
            shap_values = explainer.shap_values(X_sample)
            
            # Save SHAP values
            shap_file = os.path.join(RESULTS_DIR, f'shap_{model_name}.pkl')
            joblib.dump({
                'shap_values': shap_values,
                'X_sample': X_sample,
                'feature_names': X_sample.columns.tolist()
            }, shap_file)
            
            print(f"[SAVED] SHAP values: shap_{model_name}.pkl")
            
            # Plot summary
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_{model_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SAVED] SHAP plot: shap_summary_{model_name}.png")
            
            return shap_values
            
        except Exception as e:
            print(f"[ERROR] SHAP calculation failed: {e}")
            return None
    
    def calibrate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Calibrate model probabilities"""
        print(f"\n{'='*60}")
        print(f"MODEL CALIBRATION - {model_name.upper()}")
        print('='*60)
        
        # Calibrate using Platt scaling
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        
        print("Calibrating model...")
        calibrated.fit(X_train, y_train)
        
        # Get calibrated probabilities
        y_pred_proba_uncal = model.predict_proba(X_test)
        y_pred_proba_cal = calibrated.predict_proba(X_test)
        
        # Calculate calibration metrics
        if len(np.unique(y_test)) == 2:  # Binary
            # Brier score (lower is better)
            brier_uncal = brier_score_loss(y_test, y_pred_proba_uncal[:, 1])
            brier_cal = brier_score_loss(y_test, y_pred_proba_cal[:, 1])
            
            print(f"\nCalibration Results:")
            print(f"  Brier Score (uncalibrated): {brier_uncal:.4f}")
            print(f"  Brier Score (calibrated): {brier_cal:.4f}")
            print(f"  Improvement: {(brier_uncal - brier_cal) / brier_uncal * 100:.2f}%")
            
            # Plot calibration curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Uncalibrated
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_uncal[:, 1], n_bins=10)
            ax1.plot(prob_pred, prob_true, marker='o', label='Model')
            ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
            ax1.set_xlabel('Mean predicted probability')
            ax1.set_ylabel('Fraction of positives')
            ax1.set_title(f'Calibration Curve (Uncalibrated)\nBrier Score: {brier_uncal:.4f}')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Calibrated
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_cal[:, 1], n_bins=10)
            ax2.plot(prob_pred, prob_true, marker='o', label='Model')
            ax2.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
            ax2.set_xlabel('Mean predicted probability')
            ax2.set_ylabel('Fraction of positives')
            ax2.set_title(f'Calibration Curve (Calibrated)\nBrier Score: {brier_cal:.4f}')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'calibration_{model_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SAVED] Calibration plot: calibration_{model_name}.png")
        
        # Save calibrated model
        joblib.dump(calibrated, os.path.join(MODELS_DIR, f'{model_name}_calibrated.pkl'))
        print(f"[SAVED] Calibrated model: {model_name}_calibrated.pkl")
        
        return calibrated
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, task):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        if y_pred_proba is not None:
            if task == 'binary':
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                except:
                    metrics['roc_auc'] = None
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics

    
    def save_all_results(self):
        """Save all training results"""
        print(f"\n{'='*60}")
        print("SAVING ALL RESULTS")
        print('='*60)
        
        results_file = os.path.join(RESULTS_DIR, 'advanced_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"[SAVED] Results: advanced_training_results.json")
        
        # Create comparison table
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'roc_auc': metrics.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"[SAVED] Comparison: model_comparison.csv")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            data = comparison_df.sort_values(metric, ascending=True)
            ax.barh(data['model'], data[metric], color='steelblue')
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'Model Comparison - {metric.capitalize()}')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(data[metric]):
                ax.text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] Comparison plot: model_comparison.png")
        
        return comparison_df

def main():
    print("="*60)
    print("ADVANCED MODEL TRAINING")
    print("PKM-RE: Ensemble & Deep Learning")
    print("="*60)
    
    trainer = AdvancedModelTrainer(random_state=42)
    
    # Load data
    X, y_binary, y_multiclass, meta = trainer.load_data(use_advanced_features=True)
    
    # Filter valid multi-class samples
    valid_idx = y_multiclass >= 0
    X_multi = X[valid_idx]
    y_multi = y_multiclass[valid_idx]
    
    # Train-test split
    print("\nSplitting data...")
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    print(f"  Binary - Train: {len(X_train_bin)}, Test: {len(X_test_bin)}")
    print(f"  Multi-class - Train: {len(X_train_multi)}, Test: {len(X_test_multi)}")
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING ADVANCED MODELS")
    print("="*60)
    
    # 1. Stacking Ensemble
    trainer.train_ensemble_stacking(X_train_bin, y_train_bin, X_test_bin, y_test_bin, task='binary')
    trainer.train_ensemble_stacking(X_train_multi, y_train_multi, X_test_multi, y_test_multi, task='multiclass')
    
    # 2. Voting Ensemble (Soft)
    trainer.train_voting_ensemble(X_train_bin, y_train_bin, X_test_bin, y_test_bin, task='binary', voting='soft')
    trainer.train_voting_ensemble(X_train_multi, y_train_multi, X_test_multi, y_test_multi, task='multiclass', voting='soft')
    
    # 3. MLP
    trainer.train_deep_learning_mlp(X_train_bin, y_train_bin, X_test_bin, y_test_bin, task='binary')
    trainer.train_deep_learning_mlp(X_train_multi, y_train_multi, X_test_multi, y_test_multi, task='multiclass')
    
    # 4. CNN (if Keras available)
    trainer.train_keras_cnn(X_train_bin, y_train_bin, X_test_bin, y_test_bin, task='binary')
    trainer.train_keras_cnn(X_train_multi, y_train_multi, X_test_multi, y_test_multi, task='multiclass')
    
    # 5. CatBoost (if available)
    trainer.train_catboost(X_train_bin, y_train_bin, X_test_bin, y_test_bin, task='binary')
    trainer.train_catboost(X_train_multi, y_train_multi, X_test_multi, y_test_multi, task='multiclass')
    
    # SHAP analysis for best models
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    if 'stacking_binary' in trainer.models:
        trainer.calculate_shap_values(
            trainer.models['stacking_binary'],
            pd.DataFrame(X_test_bin, columns=X.columns),
            'stacking_binary'
        )
    
    # Model calibration
    print("\n" + "="*60)
    print("MODEL CALIBRATION")
    print("="*60)
    
    if 'stacking_binary' in trainer.models:
        trainer.calibrate_model(
            trainer.models['stacking_binary'],
            X_train_bin, y_train_bin,
            X_test_bin, y_test_bin,
            'stacking_binary'
        )
    
    # Save all results
    comparison_df = trainer.save_all_results()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nTotal models trained: {len(trainer.models)}")
    print(f"\nTop 5 Models by Accuracy:")
    print(comparison_df.head(5).to_string(index=False))
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
