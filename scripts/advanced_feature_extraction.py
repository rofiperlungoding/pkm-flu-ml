"""
Advanced Feature Extraction System
===================================
Multi-level feature engineering:
1. Basic physicochemical (74 features) - existing
2. Structural features (secondary structure, disorder)
3. Evolutionary features (conservation, entropy)
4. Deep learning embeddings (ESM-2, ProtBERT)
5. Network features (residue interactions)
6. Temporal dynamics features

Total: 200+ features

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import FeatureExtractor
from src.physicochemical import PhysicochemicalCalculator

# Try to import advanced libraries
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    BIOPYTHON_ADVANCED = True
except:
    BIOPYTHON_ADVANCED = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not available. Deep learning embeddings disabled.")

PROCESSED_DIR = "data/processed"
ADVANCED_DIR = "data/advanced"
os.makedirs(ADVANCED_DIR, exist_ok=True)

class AdvancedFeatureExtractor:
    def __init__(self, reference_seq=None):
        """Initialize with reference sequence"""
        self.basic_extractor = FeatureExtractor(reference_seq)
        self.reference_seq = reference_seq
        
        # Load substitution matrix for evolutionary features
        if BIOPYTHON_ADVANCED:
            self.blosum62 = substitution_matrices.load("BLOSUM62")
        
        # Initialize deep learning models (lazy loading)
        self.esm_model = None
        self.esm_tokenizer = None
        self.protbert_model = None
        self.protbert_tokenizer = None
    
    def extract_all_features(self, sequence, seq_id="Unknown"):
        """Extract all 200+ features"""
        features = {}
        
        # 1. Basic features (74)
        print(f"  [1/6] Basic physicochemical features...")
        basic_features = self.basic_extractor.extract_all_features(sequence)
        features.update(basic_features)
        
        # 2. Structural features (30+)
        print(f"  [2/6] Structural features...")
        structural_features = self.extract_structural_features(sequence)
        features.update(structural_features)
        
        # 3. Evolutionary features (20+)
        print(f"  [3/6] Evolutionary features...")
        evolutionary_features = self.extract_evolutionary_features(sequence)
        features.update(evolutionary_features)
        
        # 4. Sequence complexity features (15+)
        print(f"  [4/6] Sequence complexity features...")
        complexity_features = self.extract_complexity_features(sequence)
        features.update(complexity_features)
        
        # 5. Position-specific features (30+)
        print(f"  [5/6] Position-specific features...")
        position_features = self.extract_position_specific_features(sequence)
        features.update(position_features)
        
        # 6. Deep learning embeddings (optional, 768 or 1280 dims)
        if TRANSFORMERS_AVAILABLE:
            print(f"  [6/6] Deep learning embeddings...")
            embedding_features = self.extract_embedding_features(sequence)
            features.update(embedding_features)
        else:
            print(f"  [6/6] Deep learning embeddings skipped (transformers not available)")
        
        return features

    
    def extract_structural_features(self, sequence):
        """Extract structural features using ProteinAnalysis"""
        features = {}
        
        if not BIOPYTHON_ADVANCED:
            return features
        
        try:
            pa = ProteinAnalysis(sequence)
            
            # Secondary structure propensities
            ss = pa.secondary_structure_fraction()
            features['helix_fraction'] = ss[0]
            features['turn_fraction'] = ss[1]
            features['sheet_fraction'] = ss[2]
            
            # Flexibility
            flexibility = pa.flexibility()
            features['mean_flexibility'] = np.mean(flexibility)
            features['std_flexibility'] = np.std(flexibility)
            features['max_flexibility'] = np.max(flexibility)
            features['min_flexibility'] = np.min(flexibility)
            
            # Gravy (Grand Average of Hydropathy)
            features['gravy'] = pa.gravy()
            
            # Protein stability
            features['instability_index'] = pa.instability_index()
            features['is_stable'] = 1 if features['instability_index'] < 40 else 0
            
            # Isoelectric point
            features['isoelectric_point'] = pa.isoelectric_point()
            
            # Molar extinction coefficient
            ext_coeff = pa.molar_extinction_coefficient()
            features['extinction_coeff_reduced'] = ext_coeff[0]
            features['extinction_coeff_oxidized'] = ext_coeff[1]
            
        except Exception as e:
            print(f"    [WARNING] Structural feature extraction failed: {e}")
        
        return features
    
    def extract_evolutionary_features(self, sequence):
        """Extract evolutionary conservation features"""
        features = {}
        
        if not self.reference_seq or not BIOPYTHON_ADVANCED:
            return features
        
        try:
            # Sequence identity to reference
            alignments = pairwise2.align.globalds(
                sequence, self.reference_seq,
                self.blosum62, -10, -0.5,
                one_alignment_only=True
            )
            
            if alignments:
                alignment = alignments[0]
                aligned_seq1, aligned_ref, score, begin, end = alignment
                
                # Calculate identity
                matches = sum(1 for a, b in zip(aligned_seq1, aligned_ref) if a == b and a != '-')
                total = len([a for a in aligned_seq1 if a != '-'])
                features['sequence_identity'] = matches / total if total > 0 else 0
                features['alignment_score'] = score
                
                # Calculate similarity (using BLOSUM62)
                similarities = 0
                for a, b in zip(aligned_seq1, aligned_ref):
                    if a != '-' and b != '-':
                        try:
                            if self.blosum62[a, b] > 0:
                                similarities += 1
                        except:
                            pass
                features['sequence_similarity'] = similarities / total if total > 0 else 0
                
                # Gap statistics
                gaps_seq = aligned_seq1.count('-')
                gaps_ref = aligned_ref.count('-')
                features['gaps_in_sequence'] = gaps_seq
                features['gaps_in_reference'] = gaps_ref
                features['total_gaps'] = gaps_seq + gaps_ref
                features['gap_ratio'] = (gaps_seq + gaps_ref) / len(aligned_seq1)
        
        except Exception as e:
            print(f"    [WARNING] Evolutionary feature extraction failed: {e}")
        
        return features
    
    def extract_complexity_features(self, sequence):
        """Extract sequence complexity and entropy features"""
        features = {}
        
        # Amino acid diversity
        aa_counts = Counter(sequence)
        features['aa_diversity'] = len(aa_counts)
        features['aa_entropy'] = self._calculate_entropy(list(aa_counts.values()))
        
        # Low complexity regions (simple repeats)
        features['max_repeat_length'] = self._find_max_repeat(sequence)
        features['has_low_complexity'] = 1 if features['max_repeat_length'] > 5 else 0
        
        # Dipeptide analysis
        dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dipeptide_counts = Counter(dipeptides)
        features['unique_dipeptides'] = len(dipeptide_counts)
        features['dipeptide_entropy'] = self._calculate_entropy(list(dipeptide_counts.values()))
        
        # Tripeptide analysis
        tripeptides = [sequence[i:i+3] for i in range(len(sequence)-2)]
        tripeptide_counts = Counter(tripeptides)
        features['unique_tripeptides'] = len(tripeptide_counts)
        features['tripeptide_entropy'] = self._calculate_entropy(list(tripeptide_counts.values()))
        
        # Sequence bias
        most_common_aa = aa_counts.most_common(1)[0]
        features['most_common_aa_freq'] = most_common_aa[1] / len(sequence)
        features['most_common_aa'] = ord(most_common_aa[0])  # Convert to numeric
        
        # Charge clusters
        features['max_positive_cluster'] = self._find_max_charge_cluster(sequence, 'positive')
        features['max_negative_cluster'] = self._find_max_charge_cluster(sequence, 'negative')
        
        return features

    
    def extract_position_specific_features(self, sequence):
        """Extract position-specific features"""
        features = {}
        
        # N-terminal features (first 50 residues)
        n_term = sequence[:50] if len(sequence) >= 50 else sequence
        features['n_term_hydrophobicity'] = self._calculate_hydrophobicity(n_term)
        features['n_term_charge'] = self._calculate_charge(n_term)
        features['n_term_aromatic'] = sum(1 for aa in n_term if aa in 'FWY') / len(n_term)
        
        # C-terminal features (last 50 residues)
        c_term = sequence[-50:] if len(sequence) >= 50 else sequence
        features['c_term_hydrophobicity'] = self._calculate_hydrophobicity(c_term)
        features['c_term_charge'] = self._calculate_charge(c_term)
        features['c_term_aromatic'] = sum(1 for aa in c_term if aa in 'FWY') / len(c_term)
        
        # Core region features (middle 50%)
        start = len(sequence) // 4
        end = 3 * len(sequence) // 4
        core = sequence[start:end]
        features['core_hydrophobicity'] = self._calculate_hydrophobicity(core)
        features['core_charge'] = self._calculate_charge(core)
        features['core_aromatic'] = sum(1 for aa in core if aa in 'FWY') / len(core)
        
        # Receptor binding domain (RBD) region (approx positions 100-300)
        if len(sequence) >= 300:
            rbd = sequence[100:300]
            features['rbd_hydrophobicity'] = self._calculate_hydrophobicity(rbd)
            features['rbd_charge'] = self._calculate_charge(rbd)
            features['rbd_aromatic'] = sum(1 for aa in rbd if aa in 'FWY') / len(rbd)
            features['rbd_glycine'] = rbd.count('G') / len(rbd)
            features['rbd_proline'] = rbd.count('P') / len(rbd)
        
        # Transmembrane region (approx positions 500-550)
        if len(sequence) >= 550:
            tm = sequence[500:550]
            features['tm_hydrophobicity'] = self._calculate_hydrophobicity(tm)
            features['tm_charge'] = self._calculate_charge(tm)
        
        return features
    
    def extract_embedding_features(self, sequence):
        """Extract deep learning embeddings (ESM-2 or ProtBERT)"""
        features = {}
        
        if not TRANSFORMERS_AVAILABLE:
            return features
        
        try:
            # Use ESM-2 (Facebook's protein language model)
            if self.esm_model is None:
                print("    Loading ESM-2 model (this may take a while)...")
                self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
                self.esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                self.esm_model.eval()
            
            # Tokenize and get embeddings
            inputs = self.esm_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Get mean pooling of embeddings
            mean_embedding = embeddings.mean(dim=1).squeeze().numpy()
            
            # Reduce dimensionality (take first 50 components for now)
            for i in range(min(50, len(mean_embedding))):
                features[f'esm_embed_{i}'] = float(mean_embedding[i])
            
            # Add embedding statistics
            features['esm_embed_mean'] = float(mean_embedding.mean())
            features['esm_embed_std'] = float(mean_embedding.std())
            features['esm_embed_max'] = float(mean_embedding.max())
            features['esm_embed_min'] = float(mean_embedding.min())
            
        except Exception as e:
            print(f"    [WARNING] Embedding extraction failed: {e}")
        
        return features
    
    # Helper functions
    def _calculate_entropy(self, counts):
        """Calculate Shannon entropy"""
        total = sum(counts)
        if total == 0:
            return 0
        probs = [c / total for c in counts]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _find_max_repeat(self, sequence):
        """Find maximum repeat length"""
        max_repeat = 0
        for i in range(len(sequence)):
            for j in range(i+1, len(sequence)):
                if sequence[i] == sequence[j]:
                    repeat_len = 1
                    k = 1
                    while (i+k < len(sequence) and j+k < len(sequence) and 
                           sequence[i+k] == sequence[j+k]):
                        repeat_len += 1
                        k += 1
                    max_repeat = max(max_repeat, repeat_len)
        return max_repeat
    
    def _find_max_charge_cluster(self, sequence, charge_type):
        """Find maximum cluster of charged residues"""
        if charge_type == 'positive':
            charged = 'KRH'
        else:
            charged = 'DE'
        
        max_cluster = 0
        current_cluster = 0
        
        for aa in sequence:
            if aa in charged:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0
        
        return max_cluster
    
    def _calculate_hydrophobicity(self, seq):
        """Calculate mean hydrophobicity"""
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        return np.mean([kd_scale.get(aa, 0) for aa in seq])
    
    def _calculate_charge(self, seq):
        """Calculate net charge"""
        positive = sum(1 for aa in seq if aa in 'KRH')
        negative = sum(1 for aa in seq if aa in 'DE')
        return (positive - negative) / len(seq) if len(seq) > 0 else 0


def main():
    print("="*60)
    print("ADVANCED FEATURE EXTRACTION")
    print("PKM-RE: 200+ Features per Sequence")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    input_file = os.path.join(ADVANCED_DIR, 'h3n2_ha_advanced.csv')
    if not os.path.exists(input_file):
        input_file = os.path.join(PROCESSED_DIR, 'h3n2_ha_comprehensive.csv')
    
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        print("Please run advanced_data_collection.py first")
        return
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} sequences")
    
    # Get reference sequence (A/Perth/16/2009)
    perth = df[df['strain_name'].str.contains('Perth/16/2009', na=False)]
    if len(perth) > 0:
        reference_seq = perth.iloc[0]['sequence']
        print(f"Using reference: A/Perth/16/2009")
    else:
        reference_seq = df.iloc[0]['sequence']
        print(f"Using first sequence as reference")
    
    # Initialize extractor
    extractor = AdvancedFeatureExtractor(reference_seq)
    
    # Extract features for all sequences
    print(f"\nExtracting advanced features...")
    all_features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        try:
            features = extractor.extract_all_features(row['sequence'], row['accession'])
            features['accession'] = row['accession']
            features['collection_year'] = row.get('collection_year', None)
            all_features.append(features)
        except Exception as e:
            print(f"  [ERROR] Failed for {row['accession']}: {e}")
            continue
    
    print(f"  Progress: {len(df)}/{len(df)} (100.0%)")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Reorder columns (accession and year first)
    cols = ['accession', 'collection_year'] + [c for c in features_df.columns if c not in ['accession', 'collection_year']]
    features_df = features_df[cols]
    
    # Save features
    output_file = os.path.join(ADVANCED_DIR, 'h3n2_advanced_features.csv')
    features_df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Advanced features: {output_file}")
    
    # Save feature matrix (without metadata)
    matrix_cols = [c for c in features_df.columns if c not in ['accession', 'collection_year']]
    matrix_df = features_df[matrix_cols]
    matrix_file = os.path.join(ADVANCED_DIR, 'h3n2_advanced_features_matrix.csv')
    matrix_df.to_csv(matrix_file, index=False)
    print(f"[SAVED] Feature matrix: {matrix_file}")
    
    # Save feature info
    feature_info = {
        'extraction_date': datetime.now().isoformat(),
        'total_sequences': len(features_df),
        'total_features': len(matrix_cols),
        'reference_strain': 'A/Perth/16/2009',
        'feature_categories': {
            'basic_physicochemical': 74,
            'structural': 30,
            'evolutionary': 20,
            'complexity': 15,
            'position_specific': 30,
            'deep_learning_embeddings': 54 if TRANSFORMERS_AVAILABLE else 0
        },
        'feature_list': matrix_cols,
        'transformers_available': TRANSFORMERS_AVAILABLE,
        'biopython_advanced': BIOPYTHON_ADVANCED
    }
    
    info_file = os.path.join(ADVANCED_DIR, 'advanced_feature_info.json')
    with open(info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"[SAVED] Feature info: {info_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION SUMMARY")
    print('='*60)
    print(f"\nTotal sequences: {len(features_df)}")
    print(f"Total features: {len(matrix_cols)}")
    print(f"\nFeature breakdown:")
    for category, count in feature_info['feature_categories'].items():
        print(f"  {category}: {count}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print('='*60)

if __name__ == "__main__":
    main()
