"""
Model Training Pipeline for H3N2 Antigenic Prediction
======================================================
Trains XGBoost classifier to predict antigenic drift patterns.

Since we don't have actual antigenic distance labels, we use temporal drift
as a proxy - sequences from different time periods are assumed to have
different antigenic properties due to antigenic drift.

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import H3N2Predictor

# ============== CONFIGURATION ==============
FEATURES_FILE = "data/processed/h3n2_features.csv"
OUTPUT_DIR = "models"
RESULTS_DIR = "results"

# ============== LABEL CREATION ==============

def create_temporal_labels(df, year_col='collection_year'):
    """
    Create labels based on temporal periods (antigenic drift proxy)
    
    Periods based on WHO vaccine strain updates:
    - Period 1: 2009-2013 (Perth/16/2009 era)
    - Period 2: 2014-2016 (Switzerland/Hong Kong era)
    - Period 3: 2017-2019 (Singapore/Kansas era)
    - Period 4: 2020-2024 (Darwin/Cambodia era)
    """
    def assign_period(year):
        if pd.isna(year):
            return None
        year = int(year)
        if year <= 2013:
            return 'period_1_2009_2013'
        elif year <= 2016:
            return 'period_2_2014_2016'
        elif year <= 2019:
            return 'period_3_2017_2019'
        else:
            return 'period_4_2020_2024'
    
    df['antigenic_period'] = df[year_col].apply(assign_period)
    return df


def create_binary_labels(df, year_col='collection_year', threshold_year=2020):
    """
    Create binary labels: pre-pandemic vs post-pandemic
    """
    def assign_binary(year):
        if pd.isna(year):
            return None
        return 'recent' if int(year) >= threshold_year else 'historical'
    
    df['binary_label'] = df[year_col].apply(assign_binary)
    return df


def get_feature_columns(df):
    """Get only numeric feature columns (exclude metadata)"""
    exclude_cols = ['accession', 'collection_year', 'location', 'strain_name',
                   'quality_score', 'source_database', 'is_human', 'ncbi_url',
                   'antigenic_period', 'binary_label']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols 
                   and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    return feature_cols


def plot_confusion_matrix(cm, classes, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(importance_dict, output_path, top_n=20):
    """Plot and save feature importance"""
    # Sort and get top N
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    plt.barh(range(len(features)), values, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_year_distribution(df, output_path):
    """Plot year distribution of training data"""
    plt.figure(figsize=(12, 5))
    year_counts = df['collection_year'].value_counts().sort_index()
    plt.bar(year_counts.index.astype(int), year_counts.values, color='steelblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Sequences')
    plt.title('Year Distribution of Training Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_multiclass_model(df, feature_cols):
    """Train model for multi-class (temporal period) prediction"""
    print("\n" + "="*60)
    print("TRAINING MULTI-CLASS MODEL (Temporal Periods)")
    print("="*60)
    
    # Filter rows with valid labels
    df_valid = df[df['antigenic_period'].notna()].copy()
    print(f"Training samples: {len(df_valid)}")
    print(f"Class distribution:")
    print(df_valid['antigenic_period'].value_counts())
    
    X = df_valid[feature_cols]
    y = df_valid['antigenic_period']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Initialize predictor
    predictor = H3N2Predictor()
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = predictor.cross_validate(X, y, cv=5)
    print(f"CV Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y, test_size=0.2)
    predictor.train(X_train, y_train, n_estimators=150, max_depth=8, learning_rate=0.05)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    
    return predictor, metrics, cv_results


def train_binary_model(df, feature_cols):
    """Train model for binary (recent vs historical) prediction"""
    print("\n" + "="*60)
    print("TRAINING BINARY MODEL (Recent vs Historical)")
    print("="*60)
    
    # Filter rows with valid labels
    df_valid = df[df['binary_label'].notna()].copy()
    print(f"Training samples: {len(df_valid)}")
    print(f"Class distribution:")
    print(df_valid['binary_label'].value_counts())
    
    X = df_valid[feature_cols]
    y = df_valid['binary_label']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Initialize predictor
    predictor = H3N2Predictor()
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = predictor.cross_validate(X, y, cv=5)
    print(f"CV Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y, test_size=0.2)
    predictor.train(X_train, y_train, n_estimators=150, max_depth=6, learning_rate=0.05)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return predictor, metrics, cv_results


def main():
    print("="*60)
    print("H3N2 ANTIGENIC PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load features
    print("\n[1] Loading features...")
    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: {FEATURES_FILE} not found! Run extract_features.py first.")
        return
    
    df = pd.read_csv(FEATURES_FILE)
    print(f"    Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # 2. Create labels
    print("\n[2] Creating labels...")
    df = create_temporal_labels(df)
    df = create_binary_labels(df)
    
    # 3. Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"    Using {len(feature_cols)} features")
    
    # 4. Plot data distribution
    print("\n[3] Plotting data distribution...")
    plot_year_distribution(df, os.path.join(RESULTS_DIR, 'year_distribution.png'))
    
    # 5. Train multi-class model
    mc_predictor, mc_metrics, mc_cv = train_multiclass_model(df, feature_cols)
    
    # Save multi-class model
    mc_predictor.save(os.path.join(OUTPUT_DIR, 'h3n2_multiclass_model.pkl'))
    
    # Plot multi-class results
    classes = sorted(df['antigenic_period'].dropna().unique())
    plot_confusion_matrix(mc_metrics['confusion_matrix'], classes,
                         os.path.join(RESULTS_DIR, 'multiclass_confusion_matrix.png'))
    
    mc_importance = mc_predictor.get_feature_importance(top_n=20)
    plot_feature_importance(mc_importance, 
                           os.path.join(RESULTS_DIR, 'multiclass_feature_importance.png'))
    
    # 6. Train binary model
    bin_predictor, bin_metrics, bin_cv = train_binary_model(df, feature_cols)
    
    # Save binary model
    bin_predictor.save(os.path.join(OUTPUT_DIR, 'h3n2_binary_model.pkl'))
    
    # Plot binary results
    plot_confusion_matrix(bin_metrics['confusion_matrix'], ['historical', 'recent'],
                         os.path.join(RESULTS_DIR, 'binary_confusion_matrix.png'))
    
    bin_importance = bin_predictor.get_feature_importance(top_n=20)
    plot_feature_importance(bin_importance,
                           os.path.join(RESULTS_DIR, 'binary_feature_importance.png'))
    
    # 7. Save training results
    print("\n[4] Saving results...")
    results = {
        'training_date': datetime.now().isoformat(),
        'total_samples': len(df),
        'features_used': len(feature_cols),
        'multiclass_model': {
            'task': 'Temporal Period Classification',
            'classes': classes,
            'cv_accuracy': float(mc_cv['mean_accuracy']),
            'cv_std': float(mc_cv['std_accuracy']),
            'test_accuracy': float(mc_metrics['accuracy']),
            'test_f1': float(mc_metrics['f1_score']),
            'test_precision': float(mc_metrics['precision']),
            'test_recall': float(mc_metrics['recall']),
            'top_features': mc_importance
        },
        'binary_model': {
            'task': 'Recent vs Historical Classification',
            'classes': ['historical', 'recent'],
            'cv_accuracy': float(bin_cv['mean_accuracy']),
            'cv_std': float(bin_cv['std_accuracy']),
            'test_accuracy': float(bin_metrics['accuracy']),
            'test_f1': float(bin_metrics['f1_score']),
            'test_precision': float(bin_metrics['precision']),
            'test_recall': float(bin_metrics['recall']),
            'test_roc_auc': float(bin_metrics.get('roc_auc', 0)),
            'top_features': bin_importance
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 8. Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nMulti-class Model (4 periods):")
    print(f"  - CV Accuracy: {mc_cv['mean_accuracy']:.4f} (+/- {mc_cv['std_accuracy']:.4f})")
    print(f"  - Test Accuracy: {mc_metrics['accuracy']:.4f}")
    print(f"  - Test F1-Score: {mc_metrics['f1_score']:.4f}")
    
    print(f"\nBinary Model (recent vs historical):")
    print(f"  - CV Accuracy: {bin_cv['mean_accuracy']:.4f} (+/- {bin_cv['std_accuracy']:.4f})")
    print(f"  - Test Accuracy: {bin_metrics['accuracy']:.4f}")
    print(f"  - Test F1-Score: {bin_metrics['f1_score']:.4f}")
    if 'roc_auc' in bin_metrics:
        print(f"  - Test ROC-AUC: {bin_metrics['roc_auc']:.4f}")
    
    print(f"\nFiles saved:")
    print(f"  - models/h3n2_multiclass_model.pkl")
    print(f"  - models/h3n2_binary_model.pkl")
    print(f"  - results/training_results.json")
    print(f"  - results/*.png (visualizations)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
