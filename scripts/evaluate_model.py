"""
Comprehensive Model Evaluation & Validation
============================================
- Cross-validation with multiple metrics
- ROC curves and AUC scores
- Precision-Recall curves
- Confusion matrices with detailed stats
- Learning curves
- Model comparison

Author: PKM-RE Team
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Directories
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_and_models():
    """Load processed data and trained models"""
    print("Loading data and models...")
    
    # Load feature matrix
    X = pd.read_csv(os.path.join(PROCESSED_DIR, 'h3n2_features_matrix.csv'))
    
    # Load metadata for labels
    meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'h3n2_features.csv'))
    
    # Binary labels (recent vs historical)
    y_binary = (meta['collection_year'] >= 2020).astype(int)
    
    # Multi-class labels (temporal periods)
    def assign_period(year):
        if pd.isna(year):
            return -1
        if year >= 2020:
            return 3  # Recent
        elif year >= 2015:
            return 2  # Mid-recent
        elif year >= 2010:
            return 1  # Mid-historical
        else:
            return 0  # Historical
    
    y_multiclass = meta['collection_year'].apply(assign_period)
    
    # Load models
    binary_model = joblib.load(os.path.join(MODELS_DIR, 'h3n2_binary_model.pkl'))
    multiclass_model = joblib.load(os.path.join(MODELS_DIR, 'h3n2_multiclass_model.pkl'))
    
    print(f"  Data shape: {X.shape}")
    print(f"  Binary classes: {np.bincount(y_binary)}")
    print(f"  Multi-class distribution: {np.bincount(y_multiclass[y_multiclass >= 0])}")
    
    return X, y_binary, y_multiclass, binary_model, multiclass_model, meta

def cross_validation_analysis(X, y, model, model_name, cv=5):
    """Perform comprehensive cross-validation"""
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name}")
    print('='*60)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
    }
    
    cv_results = cross_validate(
        model, X, y, 
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    results = {}
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        results[metric] = {
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
        print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
    
    return results

def plot_roc_curves(X, y_binary, binary_model):
    """Plot ROC curves for binary classification"""
    print("\nGenerating ROC curves...")
    
    from sklearn.model_selection import StratifiedKFold
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (train, test) in enumerate(cv.split(X, y_binary)):
        binary_model.fit(X.iloc[train], y_binary.iloc[train])
        y_pred_proba = binary_model.predict_proba(X.iloc[test])[:, 1]
        
        fpr, tpr, _ = roc_curve(y_binary.iloc[test], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        ax.plot(fpr, tpr, lw=1, alpha=0.3, 
                label=f'ROC fold {i+1} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8,
            label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                     alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Binary Classification (Recent vs Historical)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    return mean_auc, std_auc

def plot_precision_recall_curves(X, y_binary, binary_model):
    """Plot Precision-Recall curves"""
    print("\nGenerating Precision-Recall curves...")
    
    from sklearn.model_selection import StratifiedKFold
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (train, test) in enumerate(cv.split(X, y_binary)):
        binary_model.fit(X.iloc[train], y_binary.iloc[train])
        y_pred_proba = binary_model.predict_proba(X.iloc[test])[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_binary.iloc[test], y_pred_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, lw=1, alpha=0.3,
                label=f'PR fold {i+1} (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Binary Classification', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Precision-Recall curves saved")

def plot_learning_curves(X, y, model, model_name):
    """Plot learning curves to detect overfitting/underfitting"""
    print(f"\nGenerating learning curves for {model_name}...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='r')
    
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='g')
    
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f"learning_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Learning curves saved: {filename}")

def detailed_classification_report(X, y, model, model_name, class_names=None):
    """Generate detailed classification report"""
    print(f"\n{'='*60}")
    print(f"Detailed Classification Report: {model_name}")
    print('='*60)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, 
                                   output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filename = f"confusion_matrix_detailed_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return report, cm

def save_evaluation_results(results_dict):
    """Save all evaluation results to JSON"""
    output_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\n✅ Evaluation results saved: {output_file}")

def main():
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("PKM-RE: H3N2 Antigenic Prediction")
    print("="*60)
    
    # Load data and models
    X, y_binary, y_multiclass, binary_model, multiclass_model, meta = load_data_and_models()
    
    # Filter out invalid labels for multiclass
    valid_idx = y_multiclass >= 0
    X_multi = X[valid_idx]
    y_multi = y_multiclass[valid_idx]
    
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'data_shape': X.shape,
        'binary_distribution': np.bincount(y_binary).tolist(),
        'multiclass_distribution': np.bincount(y_multi).tolist(),
    }
    
    # 1. Cross-validation analysis
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    results['binary_cv'] = cross_validation_analysis(
        X, y_binary, binary_model, "Binary Model", cv=5
    )
    
    results['multiclass_cv'] = cross_validation_analysis(
        X_multi, y_multi, multiclass_model, "Multi-class Model", cv=5
    )
    
    # 2. ROC curves (binary only)
    mean_auc, std_auc = plot_roc_curves(X, y_binary, binary_model)
    results['roc_auc'] = {'mean': float(mean_auc), 'std': float(std_auc)}
    
    # 3. Precision-Recall curves
    plot_precision_recall_curves(X, y_binary, binary_model)
    
    # 4. Learning curves
    plot_learning_curves(X, y_binary, binary_model, "Binary Model")
    plot_learning_curves(X_multi, y_multi, multiclass_model, "Multi-class Model")
    
    # 5. Detailed classification reports
    binary_report, binary_cm = detailed_classification_report(
        X, y_binary, binary_model, "Binary Model",
        class_names=['Historical (<2020)', 'Recent (≥2020)']
    )
    
    multiclass_report, multiclass_cm = detailed_classification_report(
        X_multi, y_multi, multiclass_model, "Multi-class Model",
        class_names=['<2010', '2010-2014', '2015-2019', '≥2020']
    )
    
    results['binary_report'] = binary_report
    results['multiclass_report'] = multiclass_report
    
    # 6. Save all results
    save_evaluation_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\n📊 Generated files:")
    print("  - results/roc_curves.png")
    print("  - results/precision_recall_curves.png")
    print("  - results/learning_curves_binary_model.png")
    print("  - results/learning_curves_multi-class_model.png")
    print("  - results/confusion_matrix_detailed_binary_model.png")
    print("  - results/confusion_matrix_detailed_multi-class_model.png")
    print("  - results/evaluation_results.json")

if __name__ == "__main__":
    main()
