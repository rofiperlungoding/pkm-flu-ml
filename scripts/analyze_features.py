"""
Feature Analysis & Interpretation
==================================
- Feature importance ranking
- Feature correlation analysis
- Feature distribution by class
- Physicochemical property analysis
- SHAP values for model interpretation

Author: PKM-RE Team
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from datetime import datetime

# Directories
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load feature data and models"""
    print("Loading data...")
    
    X = pd.read_csv(os.path.join(PROCESSED_DIR, 'h3n2_features_matrix.csv'))
    meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'h3n2_features.csv'))
    
    binary_model = joblib.load(os.path.join(MODELS_DIR, 'h3n2_binary_model.pkl'))
    multiclass_model = joblib.load(os.path.join(MODELS_DIR, 'h3n2_multiclass_model.pkl'))
    
    y_binary = (meta['collection_year'] >= 2020).astype(int)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    return X, meta, y_binary, binary_model, multiclass_model

def analyze_feature_importance(model, feature_names, model_name, top_n=20):
    """Analyze and visualize feature importance"""
    print(f"\n{'='*60}")
    print(f"Feature Importance Analysis: {model_name}")
    print('='*60)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, row in feat_imp_df.head(top_n).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.6f}")
    
    # Plot top features
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = feat_imp_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    filename = f"feature_importance_detailed_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Feature importance plot saved: {filename}")
    
    return feat_imp_df

def analyze_feature_correlations(X, top_features=30):
    """Analyze correlations between top features"""
    print(f"\n{'='*60}")
    print("Feature Correlation Analysis")
    print('='*60)
    
    # Select top features based on variance
    feature_var = X.var().sort_values(ascending=False)
    top_feat_names = feature_var.head(top_features).index
    X_top = X[top_feat_names]
    
    # Calculate correlation matrix
    corr_matrix = X_top.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print(f"\nHighly correlated feature pairs (|r| > 0.7): {len(high_corr_pairs)}")
    for pair in high_corr_pairs[:10]:
        print(f"  {pair['feature1']:30s} <-> {pair['feature2']:30s}: {pair['correlation']:.3f}")
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, annot=False)
    
    ax.set_title(f'Feature Correlation Matrix (Top {top_features} Features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Correlation matrix saved")
    
    return corr_matrix, high_corr_pairs

def analyze_feature_distributions(X, y_binary, top_features=12):
    """Analyze feature distributions by class"""
    print(f"\n{'='*60}")
    print("Feature Distribution Analysis")
    print('='*60)
    
    # Get top features by variance
    feature_var = X.var().sort_values(ascending=False)
    top_feat_names = feature_var.head(top_features).index
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feat in enumerate(top_feat_names):
        ax = axes[idx]
        
        # Split by class
        historical = X.loc[y_binary == 0, feat]
        recent = X.loc[y_binary == 1, feat]
        
        # Plot distributions
        ax.hist(historical, bins=30, alpha=0.6, label='Historical (<2020)', 
                color='blue', density=True)
        ax.hist(recent, bins=30, alpha=0.6, label='Recent (≥2020)', 
                color='red', density=True)
        
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    
    fig.suptitle('Feature Distributions by Temporal Class', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Feature distributions saved")

def analyze_physicochemical_properties(X, feat_imp_df):
    """Analyze physicochemical property importance"""
    print(f"\n{'='*60}")
    print("Physicochemical Property Analysis")
    print('='*60)
    
    # Group features by property type
    property_groups = {
        'Amino Acid Composition': [],
        'Hydrophobicity': [],
        'Charge': [],
        'Polarity': [],
        'Aromaticity': [],
        'Epitope Sites': [],
        'Sequence Properties': []
    }
    
    for feat in feat_imp_df['feature']:
        if 'aa_' in feat:
            property_groups['Amino Acid Composition'].append(feat)
        elif 'hydro' in feat.lower():
            property_groups['Hydrophobicity'].append(feat)
        elif 'charge' in feat.lower() or 'charged' in feat.lower():
            property_groups['Charge'].append(feat)
        elif 'polar' in feat.lower():
            property_groups['Polarity'].append(feat)
        elif 'aromatic' in feat.lower():
            property_groups['Aromaticity'].append(feat)
        elif 'epitope' in feat.lower() or 'site_' in feat:
            property_groups['Epitope Sites'].append(feat)
        else:
            property_groups['Sequence Properties'].append(feat)
    
    # Calculate average importance per group
    group_importance = {}
    for group, features in property_groups.items():
        if features:
            importances = feat_imp_df[feat_imp_df['feature'].isin(features)]['importance']
            group_importance[group] = {
                'mean': importances.mean(),
                'sum': importances.sum(),
                'count': len(features),
                'top_feature': feat_imp_df[feat_imp_df['feature'].isin(features)].iloc[0]['feature']
            }
    
    print("\nProperty Group Importance:")
    for group, stats in sorted(group_importance.items(), 
                               key=lambda x: x[1]['mean'], reverse=True):
        print(f"\n  {group}:")
        print(f"    Features: {stats['count']}")
        print(f"    Mean importance: {stats['mean']:.6f}")
        print(f"    Total importance: {stats['sum']:.6f}")
        print(f"    Top feature: {stats['top_feature']}")
    
    # Plot property group importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    groups = list(group_importance.keys())
    means = [group_importance[g]['mean'] for g in groups]
    sums = [group_importance[g]['sum'] for g in groups]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    # Mean importance
    bars1 = ax1.barh(groups, means, color=colors)
    ax1.set_xlabel('Mean Feature Importance', fontsize=12)
    ax1.set_title('Average Importance by Property Group', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars1, means):
        ax1.text(val, bar.get_y() + bar.get_height()/2, 
                f' {val:.4f}', va='center', fontsize=10)
    
    # Total importance
    bars2 = ax2.barh(groups, sums, color=colors)
    ax2.set_xlabel('Total Feature Importance', fontsize=12)
    ax2.set_title('Cumulative Importance by Property Group', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars2, sums):
        ax2.text(val, bar.get_y() + bar.get_height()/2, 
                f' {val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'property_group_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Property group analysis saved")
    
    return group_importance

def save_analysis_results(feat_imp_binary, feat_imp_multi, 
                          corr_pairs, group_importance):
    """Save analysis results to JSON"""
    results = {
        'analysis_date': datetime.now().isoformat(),
        'binary_top_20_features': feat_imp_binary.head(20).to_dict('records'),
        'multiclass_top_20_features': feat_imp_multi.head(20).to_dict('records'),
        'high_correlation_pairs': corr_pairs[:20],
        'property_group_importance': group_importance,
    }
    
    output_file = os.path.join(RESULTS_DIR, 'feature_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis results saved: {output_file}")

def main():
    print("="*60)
    print("FEATURE ANALYSIS & INTERPRETATION")
    print("PKM-RE: H3N2 Antigenic Prediction")
    print("="*60)
    
    # Load data
    X, meta, y_binary, binary_model, multiclass_model = load_data()
    
    # 1. Feature importance analysis
    feat_imp_binary = analyze_feature_importance(
        binary_model, X.columns, "Binary Model", top_n=20
    )
    
    feat_imp_multi = analyze_feature_importance(
        multiclass_model, X.columns, "Multi-class Model", top_n=20
    )
    
    # 2. Feature correlation analysis
    corr_matrix, corr_pairs = analyze_feature_correlations(X, top_features=30)
    
    # 3. Feature distribution analysis
    analyze_feature_distributions(X, y_binary, top_features=12)
    
    # 4. Physicochemical property analysis
    group_importance = analyze_physicochemical_properties(X, feat_imp_binary)
    
    # 5. Save results
    save_analysis_results(feat_imp_binary, feat_imp_multi, 
                          corr_pairs, group_importance)
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📊 Generated files:")
    print("  - results/feature_importance_detailed_binary_model.png")
    print("  - results/feature_importance_detailed_multi-class_model.png")
    print("  - results/feature_correlation_matrix.png")
    print("  - results/feature_distributions.png")
    print("  - results/property_group_importance.png")
    print("  - results/feature_analysis.json")

if __name__ == "__main__":
    main()
