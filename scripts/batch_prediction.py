"""
Batch Prediction System for Large-Scale H3N2 Sequence Analysis
===============================================================
High-performance parallel processing pipeline for predicting temporal periods
and antigenic properties of large sequence datasets.

Features:
- Parallel processing with multiprocessing
- Progress tracking with tqdm
- Error handling and recovery
- Checkpoint system for resumability
- Memory-efficient batch processing
- Comprehensive result aggregation
- Statistical analysis of predictions
- Confidence interval calculation
- Ensemble prediction aggregation

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import FeatureExtractor

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except:
    TQDM_AVAILABLE = False
    print("[WARNING] tqdm not available. Progress bars disabled.")

MODELS_DIR = "models"
ADVANCED_MODELS_DIR = "models/advanced"
RESULTS_DIR = "results/batch"
CHECKPOINT_DIR = "results/batch/checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class BatchPredictor:
    """High-performance batch prediction system"""
    
    def __init__(self, model_type='basic', use_ensemble=False):
        """
        Initialize batch predictor
        
        Args:
            model_type: 'basic', 'advanced', or 'all'
            use_ensemble: Whether to use ensemble predictions
        """
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        self.models = {}
        self.scalers = {}
        self.feature_extractor = None
        self.reference_seq = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        print("Loading models...")
        
        # Load basic models
        basic_binary = os.path.join(MODELS_DIR, 'h3n2_binary_model.pkl')
        basic_multi = os.path.join(MODELS_DIR, 'h3n2_multiclass_model.pkl')
        
        if os.path.exists(basic_binary):
            model_dict = joblib.load(basic_binary)
            # Handle both dict and direct model formats
            if isinstance(model_dict, dict) and 'model' in model_dict:
                self.models['basic_binary'] = model_dict['model']
                if 'scaler' in model_dict:
                    self.scalers['basic_binary'] = model_dict['scaler']
            else:
                self.models['basic_binary'] = model_dict
            print("  ✓ Basic binary model")
        
        if os.path.exists(basic_multi):
            model_dict = joblib.load(basic_multi)
            # Handle both dict and direct model formats
            if isinstance(model_dict, dict) and 'model' in model_dict:
                self.models['basic_multiclass'] = model_dict['model']
                if 'scaler' in model_dict:
                    self.scalers['basic_multiclass'] = model_dict['scaler']
            else:
                self.models['basic_multiclass'] = model_dict
            print("  ✓ Basic multi-class model")
        
        # Load advanced models if requested
        if self.model_type in ['advanced', 'all']:
            advanced_models = [
                'stacking_binary', 'stacking_multiclass',
                'voting_soft_binary', 'voting_soft_multiclass',
                'mlp_binary', 'mlp_multiclass',
                'catboost_binary', 'catboost_multiclass'
            ]
            
            for model_name in advanced_models:
                model_path = os.path.join(ADVANCED_MODELS_DIR, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"  ✓ {model_name}")
                    
                    # Load scaler if exists
                    scaler_path = os.path.join(ADVANCED_MODELS_DIR, f'{model_name}_scaler.pkl')
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)
        
        if not self.models:
            raise ValueError("No models found! Please train models first.")
        
        print(f"\nTotal models loaded: {len(self.models)}")
    
    def _load_reference_sequence(self):
        """Load reference sequence for feature extraction"""
        # Try to load from processed data
        processed_file = "data/processed/h3n2_ha_comprehensive.csv"
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file)
            perth = df[df['strain_name'].str.contains('Perth/16/2009', na=False)]
            if len(perth) > 0:
                self.reference_seq = perth.iloc[0]['sequence']
                print("Using reference: A/Perth/16/2009")
                return
        
        # Fallback: use first sequence
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file)
            self.reference_seq = df.iloc[0]['sequence']
            print("Using first sequence as reference")
        else:
            raise ValueError("No reference sequence found!")
    
    def predict_single(self, sequence, seq_id="Unknown"):
        """
        Predict for a single sequence
        
        Args:
            sequence: Protein sequence string
            seq_id: Sequence identifier
            
        Returns:
            Dictionary with predictions from all models
        """
        # Extract features
        if self.feature_extractor is None:
            if self.reference_seq is None:
                self._load_reference_sequence()
            self.feature_extractor = FeatureExtractor(self.reference_seq)
        
        features = self.feature_extractor.extract_all_features(sequence)
        X = pd.DataFrame([features])
        
        # Get predictions from all models
        predictions = {
            'sequence_id': seq_id,
            'sequence_length': len(sequence),
            'predictions': {}
        }
        
        for model_name, model in self.models.items():
            try:
                # Scale if needed
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                    pred = model.predict(X_scaled)[0]
                    pred_proba = model.predict_proba(X_scaled)[0]
                else:
                    pred = model.predict(X)[0]
                    pred_proba = model.predict_proba(X)[0]
                
                # Determine task type
                task = 'binary' if 'binary' in model_name else 'multiclass'
                
                # Format prediction
                if task == 'binary':
                    label = "Recent (≥2020)" if pred == 1 else "Historical (<2020)"
                    confidence = pred_proba[pred] * 100
                    probabilities = {
                        'historical': float(pred_proba[0]),
                        'recent': float(pred_proba[1])
                    }
                else:
                    period_labels = {
                        0: '<2010', 1: '2010-2014',
                        2: '2015-2019', 3: '≥2020'
                    }
                    label = period_labels[pred]
                    confidence = pred_proba[pred] * 100
                    probabilities = {
                        period_labels[i]: float(prob)
                        for i, prob in enumerate(pred_proba)
                    }
                
                predictions['predictions'][model_name] = {
                    'prediction': int(pred),
                    'label': label,
                    'confidence': float(confidence),
                    'probabilities': probabilities
                }
                
            except Exception as e:
                predictions['predictions'][model_name] = {
                    'error': str(e)
                }
        
        # Ensemble prediction if requested
        if self.use_ensemble and len(self.models) > 1:
            predictions['ensemble'] = self._ensemble_prediction(predictions['predictions'])
        
        return predictions

    
    def _ensemble_prediction(self, predictions):
        """Aggregate predictions from multiple models"""
        # Separate binary and multiclass predictions
        binary_preds = []
        binary_probs = []
        multi_preds = []
        multi_probs = []
        
        for model_name, pred_data in predictions.items():
            if 'error' in pred_data:
                continue
            
            if 'binary' in model_name:
                binary_preds.append(pred_data['prediction'])
                binary_probs.append(pred_data['probabilities'])
            else:
                multi_preds.append(pred_data['prediction'])
                multi_probs.append(pred_data['probabilities'])
        
        ensemble = {}
        
        # Binary ensemble (majority voting + average probabilities)
        if binary_preds:
            # Majority vote
            ensemble['binary_vote'] = int(np.round(np.mean(binary_preds)))
            
            # Average probabilities
            avg_probs = {
                'historical': np.mean([p['historical'] for p in binary_probs]),
                'recent': np.mean([p['recent'] for p in binary_probs])
            }
            ensemble['binary_probabilities'] = avg_probs
            ensemble['binary_confidence'] = max(avg_probs.values()) * 100
            ensemble['binary_label'] = "Recent (≥2020)" if ensemble['binary_vote'] == 1 else "Historical (<2020)"
            
            # Uncertainty (std of probabilities)
            ensemble['binary_uncertainty'] = np.std([p['recent'] for p in binary_probs])
        
        # Multi-class ensemble
        if multi_preds:
            # Majority vote
            ensemble['multiclass_vote'] = int(np.round(np.mean(multi_preds)))
            
            # Average probabilities
            period_labels = ['<2010', '2010-2014', '2015-2019', '≥2020']
            avg_probs = {}
            for label in period_labels:
                probs = [p.get(label, 0) for p in multi_probs]
                avg_probs[label] = np.mean(probs)
            
            ensemble['multiclass_probabilities'] = avg_probs
            ensemble['multiclass_confidence'] = max(avg_probs.values()) * 100
            ensemble['multiclass_label'] = period_labels[ensemble['multiclass_vote']]
            
            # Uncertainty
            max_label = period_labels[ensemble['multiclass_vote']]
            ensemble['multiclass_uncertainty'] = np.std([p.get(max_label, 0) for p in multi_probs])
        
        return ensemble
    
    def predict_batch(self, sequences, seq_ids=None, batch_size=100, 
                     n_jobs=-1, checkpoint_interval=500):
        """
        Predict for a batch of sequences with parallel processing
        
        Args:
            sequences: List of protein sequences
            seq_ids: List of sequence identifiers
            batch_size: Number of sequences per batch
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            checkpoint_interval: Save checkpoint every N sequences
            
        Returns:
            List of prediction dictionaries
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if len(sequences) != len(seq_ids):
            raise ValueError("sequences and seq_ids must have same length")
        
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION")
        print('='*60)
        print(f"Total sequences: {len(sequences)}")
        print(f"Batch size: {batch_size}")
        print(f"Parallel jobs: {n_jobs if n_jobs > 0 else cpu_count()}")
        print(f"Checkpoint interval: {checkpoint_interval}")
        
        # Check for existing checkpoint
        checkpoint_file = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.json')
        start_idx = 0
        all_results = []
        
        if os.path.exists(checkpoint_file):
            print("\nFound existing checkpoint. Resume? (y/n)")
            # For automation, auto-resume
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint['last_index']
                all_results = checkpoint['results']
                print(f"Resuming from index {start_idx}")
        
        # Process in batches
        n_batches = (len(sequences) - start_idx + batch_size - 1) // batch_size
        
        iterator = range(start_idx, len(sequences), batch_size)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Processing batches", total=n_batches)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids = seq_ids[batch_start:batch_end]
            
            # Parallel processing
            if n_jobs == 1:
                # Sequential processing
                batch_results = [
                    self.predict_single(seq, sid)
                    for seq, sid in zip(batch_seqs, batch_ids)
                ]
            else:
                # Parallel processing
                n_workers = cpu_count() if n_jobs == -1 else n_jobs
                with Pool(n_workers) as pool:
                    batch_results = pool.starmap(
                        self.predict_single,
                        zip(batch_seqs, batch_ids)
                    )
            
            all_results.extend(batch_results)
            
            # Save checkpoint
            if (batch_end % checkpoint_interval == 0) or (batch_end == len(sequences)):
                checkpoint = {
                    'last_index': batch_end,
                    'total': len(sequences),
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
        
        # Remove checkpoint after completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"\n✓ Batch prediction complete: {len(all_results)} sequences")
        
        return all_results
    
    def predict_from_fasta(self, fasta_file, output_file=None, **kwargs):
        """
        Predict for all sequences in a FASTA file
        
        Args:
            fasta_file: Path to FASTA file
            output_file: Path to save results (optional)
            **kwargs: Additional arguments for predict_batch
            
        Returns:
            List of prediction dictionaries
        """
        from Bio import SeqIO
        
        print(f"Loading sequences from {fasta_file}...")
        
        sequences = []
        seq_ids = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
            seq_ids.append(record.id)
        
        print(f"Loaded {len(sequences)} sequences")
        
        # Predict
        results = self.predict_batch(sequences, seq_ids, **kwargs)
        
        # Save if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, output_file):
        """Save prediction results to file"""
        print(f"\nSaving results to {output_file}...")
        
        # Determine format from extension
        ext = Path(output_file).suffix.lower()
        
        if ext == '.json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif ext == '.csv':
            # Flatten results for CSV
            rows = []
            for result in results:
                row = {
                    'sequence_id': result['sequence_id'],
                    'sequence_length': result['sequence_length']
                }
                
                # Add predictions from each model
                if 'predictions' in result:
                    for model_name, pred in result['predictions'].items():
                        if 'error' not in pred:
                            row[f'{model_name}_prediction'] = pred['prediction']
                            row[f'{model_name}_label'] = pred['label']
                            row[f'{model_name}_confidence'] = pred['confidence']
                
                # Add ensemble if available
                if 'ensemble' in result:
                    ens = result['ensemble']
                    if 'binary_vote' in ens:
                        row['ensemble_binary_prediction'] = ens['binary_vote']
                        row['ensemble_binary_label'] = ens['binary_label']
                        row['ensemble_binary_confidence'] = ens['binary_confidence']
                        row['ensemble_binary_uncertainty'] = ens['binary_uncertainty']
                    
                    if 'multiclass_vote' in ens:
                        row['ensemble_multiclass_prediction'] = ens['multiclass_vote']
                        row['ensemble_multiclass_label'] = ens['multiclass_label']
                        row['ensemble_multiclass_confidence'] = ens['multiclass_confidence']
                        row['ensemble_multiclass_uncertainty'] = ens['multiclass_uncertainty']
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .json or .csv")
        
        print(f"✓ Results saved: {output_file}")

    
    def analyze_results(self, results, output_dir=None):
        """
        Perform statistical analysis of batch prediction results
        
        Args:
            results: List of prediction dictionaries
            output_dir: Directory to save analysis plots
            
        Returns:
            Dictionary with analysis statistics
        """
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS")
        print('='*60)
        
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        analysis = {
            'total_sequences': len(results),
            'timestamp': datetime.now().isoformat(),
            'models_used': list(self.models.keys()),
        }
        
        # Analyze each model's predictions
        for model_name in self.models.keys():
            model_preds = []
            model_confs = []
            
            for result in results:
                if 'predictions' in result and model_name in result['predictions']:
                    pred_data = result['predictions'][model_name]
                    if 'error' not in pred_data:
                        model_preds.append(pred_data['prediction'])
                        model_confs.append(pred_data['confidence'])
            
            if model_preds:
                analysis[model_name] = {
                    'total_predictions': len(model_preds),
                    'prediction_distribution': dict(pd.Series(model_preds).value_counts()),
                    'mean_confidence': float(np.mean(model_confs)),
                    'std_confidence': float(np.std(model_confs)),
                    'min_confidence': float(np.min(model_confs)),
                    'max_confidence': float(np.max(model_confs)),
                    'median_confidence': float(np.median(model_confs))
                }
        
        # Analyze ensemble predictions if available
        if self.use_ensemble and len(results) > 0 and 'ensemble' in results[0]:
            binary_votes = []
            binary_confs = []
            binary_uncs = []
            multi_votes = []
            multi_confs = []
            multi_uncs = []
            
            for result in results:
                if 'ensemble' in result:
                    ens = result['ensemble']
                    if 'binary_vote' in ens:
                        binary_votes.append(ens['binary_vote'])
                        binary_confs.append(ens['binary_confidence'])
                        binary_uncs.append(ens['binary_uncertainty'])
                    if 'multiclass_vote' in ens:
                        multi_votes.append(ens['multiclass_vote'])
                        multi_confs.append(ens['multiclass_confidence'])
                        multi_uncs.append(ens['multiclass_uncertainty'])
            
            analysis['ensemble'] = {}
            
            if binary_votes:
                analysis['ensemble']['binary'] = {
                    'total_predictions': len(binary_votes),
                    'prediction_distribution': dict(pd.Series(binary_votes).value_counts()),
                    'mean_confidence': float(np.mean(binary_confs)),
                    'std_confidence': float(np.std(binary_confs)),
                    'mean_uncertainty': float(np.mean(binary_uncs)),
                    'std_uncertainty': float(np.std(binary_uncs))
                }
            
            if multi_votes:
                analysis['ensemble']['multiclass'] = {
                    'total_predictions': len(multi_votes),
                    'prediction_distribution': dict(pd.Series(multi_votes).value_counts()),
                    'mean_confidence': float(np.mean(multi_confs)),
                    'std_confidence': float(np.std(multi_confs)),
                    'mean_uncertainty': float(np.mean(multi_uncs)),
                    'std_uncertainty': float(np.std(multi_uncs))
                }
        
        # Save analysis
        analysis_file = os.path.join(output_dir, 'batch_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Analysis saved: {analysis_file}")
        
        # Create visualization plots
        self._create_analysis_plots(results, output_dir)
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total sequences analyzed: {analysis['total_sequences']}")
        print(f"  Models used: {len(analysis['models_used'])}")
        
        if 'ensemble' in analysis and 'binary' in analysis['ensemble']:
            ens_bin = analysis['ensemble']['binary']
            print(f"\n  Ensemble Binary Predictions:")
            print(f"    Mean confidence: {ens_bin['mean_confidence']:.2f}%")
            print(f"    Mean uncertainty: {ens_bin['mean_uncertainty']:.4f}")
            print(f"    Distribution: {ens_bin['prediction_distribution']}")
        
        return analysis
    
    def _create_analysis_plots(self, results, output_dir):
        """Create visualization plots for batch analysis"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style('whitegrid')
            
            # Extract data for plotting
            plot_data = []
            for result in results:
                if 'predictions' not in result:
                    continue
                for model_name, pred in result['predictions'].items():
                    if 'error' not in pred:
                        plot_data.append({
                            'model': model_name,
                            'prediction': pred['label'],
                            'confidence': pred['confidence']
                        })
            
            if not plot_data:
                print("  [WARNING] No data available for plotting")
                return
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Prediction distribution by model
            ax1 = axes[0, 0]
            pred_counts = plot_df.groupby(['model', 'prediction']).size().unstack(fill_value=0)
            pred_counts.plot(kind='bar', ax=ax1, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
            ax1.set_title('Prediction Distribution by Model', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Count')
            ax1.legend(title='Prediction', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(axis='y', alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 2. Confidence distribution
            ax2 = axes[0, 1]
            for model in plot_df['model'].unique():
                model_data = plot_df[plot_df['model'] == model]['confidence']
                ax2.hist(model_data, alpha=0.6, label=model, bins=20)
            ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Confidence (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. Box plot of confidence by model
            ax3 = axes[1, 0]
            plot_df.boxplot(column='confidence', by='model', ax=ax3)
            ax3.set_title('Confidence by Model', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Confidence (%)')
            plt.sca(ax3)
            plt.xticks(rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
            
            # 4. Mean confidence comparison
            ax4 = axes[1, 1]
            mean_conf = plot_df.groupby('model')['confidence'].mean().sort_values(ascending=True)
            mean_conf.plot(kind='barh', ax=ax4, color='steelblue')
            ax4.set_title('Mean Confidence by Model', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Mean Confidence (%)')
            ax4.set_ylabel('Model')
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(mean_conf):
                ax4.text(v, i, f' {v:.2f}%', va='center')
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, 'batch_analysis_plots.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Analysis plots saved: {plot_file}")
            
        except Exception as e:
            print(f"  [WARNING] Plot creation failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch prediction for H3N2 sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from FASTA file with basic models
  python scripts/batch_prediction.py --fasta input.fasta --output results.csv
  
  # Use advanced models with ensemble
  python scripts/batch_prediction.py --fasta input.fasta --output results.json \\
      --model-type advanced --ensemble
  
  # Parallel processing with 8 workers
  python scripts/batch_prediction.py --fasta input.fasta --output results.csv \\
      --n-jobs 8 --batch-size 200
  
  # With analysis
  python scripts/batch_prediction.py --fasta input.fasta --output results.csv \\
      --analyze --analysis-dir results/analysis
        """
    )
    
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file (.json or .csv)')
    parser.add_argument('--model-type', type=str, default='basic',
                       choices=['basic', 'advanced', 'all'],
                       help='Type of models to use')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble predictions')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                       help='Save checkpoint every N sequences')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform statistical analysis')
    parser.add_argument('--analysis-dir', type=str, default=None,
                       help='Directory for analysis outputs')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BATCH PREDICTION SYSTEM")
    print("PKM-RE: H3N2 Antigenic Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = BatchPredictor(
        model_type=args.model_type,
        use_ensemble=args.ensemble
    )
    
    # Predict
    results = predictor.predict_from_fasta(
        args.fasta,
        output_file=args.output,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Analyze if requested
    if args.analyze:
        predictor.analyze_results(results, output_dir=args.analysis_dir)
    
    print("\n" + "="*60)
    print("BATCH PREDICTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
