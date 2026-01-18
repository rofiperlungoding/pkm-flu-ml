"""
Comprehensive Feature Extraction Pipeline for H3N2 HA Sequences
================================================================
Extracts physicochemical features from protein sequences for ML model training.

Features extracted:
1. Amino acid composition (20 features)
2. Physicochemical properties (6 properties x multiple metrics)
3. Epitope site mutations
4. Sequence-based features (length, molecular weight, etc.)

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
import os
import json
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physicochemical import (
    HYDROPHOBICITY, VOLUME, POLARITY, CHARGE, 
    MOLECULAR_WEIGHT, ISOELECTRIC_POINT, EPITOPE_SITES
)

# ============== CONFIGURATION ==============
INPUT_CSV = "data/processed/h3n2_ha_comprehensive.csv"
OUTPUT_DIR = "data/processed"
REFERENCE_STRAIN = "A/Perth/16/2009"  # WHO reference strain

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ============== FEATURE EXTRACTION FUNCTIONS ==============

def get_aa_composition(sequence):
    """Calculate amino acid composition (frequency of each AA)"""
    seq_clean = ''.join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
    total = len(seq_clean)
    if total == 0:
        return {f'aa_{aa}': 0.0 for aa in AMINO_ACIDS}
    
    counts = Counter(seq_clean)
    return {f'aa_{aa}': counts.get(aa, 0) / total for aa in AMINO_ACIDS}


def get_physicochemical_features(sequence):
    """Calculate physicochemical property statistics"""
    seq_clean = ''.join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
    if len(seq_clean) == 0:
        return {}
    
    features = {}
    properties = {
        'hydrophobicity': HYDROPHOBICITY,
        'volume': VOLUME,
        'polarity': POLARITY,
        'charge': CHARGE,
        'molecular_weight': MOLECULAR_WEIGHT,
        'isoelectric_point': ISOELECTRIC_POINT
    }
    
    for prop_name, prop_dict in properties.items():
        values = [prop_dict.get(aa, 0) for aa in seq_clean if aa in prop_dict]
        if values:
            features[f'{prop_name}_mean'] = np.mean(values)
            features[f'{prop_name}_std'] = np.std(values)
            features[f'{prop_name}_min'] = np.min(values)
            features[f'{prop_name}_max'] = np.max(values)
            features[f'{prop_name}_sum'] = np.sum(values)
    
    return features


def get_epitope_features(sequence, reference_seq=None):
    """Calculate features related to epitope sites"""
    features = {}
    seq_clean = sequence.upper()
    
    for site_name, positions in EPITOPE_SITES.items():
        # Get amino acids at epitope positions
        site_aas = []
        for pos in positions:
            if pos < len(seq_clean):
                site_aas.append(seq_clean[pos])
        
        site_seq = ''.join(site_aas)
        
        # Composition at this site
        if site_seq:
            # Hydrophobicity at epitope
            hydro_vals = [HYDROPHOBICITY.get(aa, 0) for aa in site_seq if aa in HYDROPHOBICITY]
            if hydro_vals:
                features[f'epitope_{site_name}_hydro_mean'] = np.mean(hydro_vals)
            
            # Charge at epitope
            charge_vals = [CHARGE.get(aa, 0) for aa in site_seq if aa in CHARGE]
            if charge_vals:
                features[f'epitope_{site_name}_charge_sum'] = np.sum(charge_vals)
        
        # Count mutations at epitope if reference provided
        if reference_seq and len(reference_seq) >= max(positions, default=0):
            mutations = 0
            for pos in positions:
                if pos < len(seq_clean) and pos < len(reference_seq):
                    if seq_clean[pos] != reference_seq[pos]:
                        mutations += 1
            features[f'epitope_{site_name}_mutations'] = mutations
    
    return features


def get_sequence_features(sequence):
    """Calculate general sequence-based features"""
    seq_clean = ''.join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
    
    features = {
        'seq_length': len(seq_clean),
        'seq_length_raw': len(sequence),
    }
    
    # Molecular weight
    mw = sum(MOLECULAR_WEIGHT.get(aa, 0) for aa in seq_clean)
    features['total_molecular_weight'] = mw
    
    # Net charge
    net_charge = sum(CHARGE.get(aa, 0) for aa in seq_clean)
    features['net_charge'] = net_charge
    
    # Hydrophobic ratio
    hydrophobic_aas = set('AILMFVWY')
    hydro_count = sum(1 for aa in seq_clean if aa in hydrophobic_aas)
    features['hydrophobic_ratio'] = hydro_count / len(seq_clean) if seq_clean else 0
    
    # Charged ratio
    charged_aas = set('DEKRH')
    charged_count = sum(1 for aa in seq_clean if aa in charged_aas)
    features['charged_ratio'] = charged_count / len(seq_clean) if seq_clean else 0
    
    # Polar ratio
    polar_aas = set('STNQ')
    polar_count = sum(1 for aa in seq_clean if aa in polar_aas)
    features['polar_ratio'] = polar_count / len(seq_clean) if seq_clean else 0
    
    # Aromatic ratio
    aromatic_aas = set('FWY')
    aromatic_count = sum(1 for aa in seq_clean if aa in aromatic_aas)
    features['aromatic_ratio'] = aromatic_count / len(seq_clean) if seq_clean else 0
    
    return features


def count_mutations(seq1, seq2):
    """Count number of mutations between two sequences"""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    
    mutations = sum(1 for a, b in zip(seq1, seq2) if a != b and a in AMINO_ACIDS and b in AMINO_ACIDS)
    return mutations


def extract_all_features(sequence, reference_seq=None):
    """Extract all features from a single sequence"""
    features = {}
    
    # 1. Amino acid composition (20 features)
    features.update(get_aa_composition(sequence))
    
    # 2. Physicochemical properties (30 features: 6 props x 5 stats)
    features.update(get_physicochemical_features(sequence))
    
    # 3. Epitope site features (15+ features)
    features.update(get_epitope_features(sequence, reference_seq))
    
    # 4. General sequence features (10+ features)
    features.update(get_sequence_features(sequence))
    
    # 5. Mutation count vs reference
    if reference_seq:
        features['total_mutations_vs_ref'] = count_mutations(sequence, reference_seq)
    
    return features


class FeatureExtractor:
    def __init__(self, reference_seq=None):
        self.reference_seq = reference_seq
        self.feature_names = None
        
    def fit_transform(self, sequences, metadata_df=None):
        """Extract features from list of sequences"""
        print(f"Extracting features from {len(sequences)} sequences...")
        
        all_features = []
        for i, seq in enumerate(sequences):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(sequences)}...")
            
            feat = extract_all_features(seq, self.reference_seq)
            all_features.append(feat)
        
        df = pd.DataFrame(all_features)
        self.feature_names = df.columns.tolist()
        
        # Add metadata if provided
        if metadata_df is not None:
            # Reset index to ensure alignment
            metadata_df = metadata_df.reset_index(drop=True)
            df = df.reset_index(drop=True)
            
            # Add key metadata columns
            meta_cols = ['accession', 'collection_year', 'location', 'strain_name', 
                        'quality_score', 'source_database', 'is_human', 'ncbi_url']
            for col in meta_cols:
                if col in metadata_df.columns:
                    df[col] = metadata_df[col].values
        
        print(f"  Extracted {len(self.feature_names)} features")
        return df
    
    def get_feature_names(self):
        return self.feature_names


def load_reference_sequence():
    """Load reference sequence (A/Perth/16/2009 or similar)"""
    # Try to find reference in the dataset
    csv_path = INPUT_CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Look for Perth/16/2009 or similar vaccine strain
        ref_patterns = ['Perth/16/2009', 'Darwin/6/2021', 'Hong Kong/4801/2014']
        for pattern in ref_patterns:
            matches = df[df['description'].str.contains(pattern, na=False, case=False)]
            if len(matches) > 0:
                ref_seq = matches.iloc[0]['sequence']
                ref_name = matches.iloc[0].get('strain_name', pattern)
                print(f"Using reference: {ref_name}")
                return ref_seq
    
    # Fallback: use first high-quality sequence
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        hq = df[df['quality_score'] >= 8].head(1)
        if len(hq) > 0:
            print(f"Using first high-quality sequence as reference: {hq.iloc[0]['accession']}")
            return hq.iloc[0]['sequence']
    
    return None


def main():
    print("="*60)
    print("H3N2 HA FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found!")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"    Loaded {len(df)} sequences")
    
    # 2. Filter valid sequences
    print("\n[2] Filtering sequences...")
    # Remove sequences with invalid characters
    valid_chars = set(AMINO_ACIDS + ['X', '-', '*'])
    df['is_valid'] = df['sequence'].apply(
        lambda s: all(c in valid_chars for c in str(s).upper())
    )
    df_valid = df[df['is_valid']].copy()
    print(f"    Valid sequences: {len(df_valid)}")
    
    # Filter by length (typical HA is 550-570 aa)
    df_valid = df_valid[(df_valid['length'] >= 500) & (df_valid['length'] <= 600)]
    print(f"    After length filter (500-600): {len(df_valid)}")
    
    # 3. Load reference sequence
    print("\n[3] Loading reference sequence...")
    reference_seq = load_reference_sequence()
    
    # 4. Extract features
    print("\n[4] Extracting features...")
    extractor = FeatureExtractor(reference_seq=reference_seq)
    
    sequences = df_valid['sequence'].tolist()
    features_df = extractor.fit_transform(sequences, df_valid)
    
    # 5. Save results
    print("\n[5] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save full feature matrix
    output_file = os.path.join(OUTPUT_DIR, 'h3n2_features.csv')
    features_df.to_csv(output_file, index=False)
    print(f"    Saved: {output_file}")
    
    # Save feature-only matrix (for ML)
    feature_cols = [c for c in features_df.columns if c not in 
                   ['accession', 'collection_year', 'location', 'strain_name', 
                    'quality_score', 'source_database', 'is_human', 'ncbi_url']]
    
    features_only = features_df[feature_cols]
    features_only_file = os.path.join(OUTPUT_DIR, 'h3n2_features_matrix.csv')
    features_only.to_csv(features_only_file, index=False)
    print(f"    Saved: {features_only_file}")
    
    # Save feature metadata
    feature_info = {
        'extraction_date': datetime.now().isoformat(),
        'total_sequences': len(features_df),
        'total_features': len(feature_cols),
        'feature_groups': {
            'amino_acid_composition': len([c for c in feature_cols if c.startswith('aa_')]),
            'physicochemical': len([c for c in feature_cols if any(p in c for p in ['hydrophobicity', 'volume', 'polarity', 'charge', 'molecular_weight', 'isoelectric'])]),
            'epitope': len([c for c in feature_cols if c.startswith('epitope_')]),
            'sequence': len([c for c in feature_cols if c.startswith('seq_') or c in ['total_molecular_weight', 'net_charge', 'hydrophobic_ratio', 'charged_ratio', 'polar_ratio', 'aromatic_ratio', 'total_mutations_vs_ref']])
        },
        'feature_names': feature_cols,
        'reference_used': REFERENCE_STRAIN
    }
    
    info_file = os.path.join(OUTPUT_DIR, 'feature_extraction_info.json')
    with open(info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"    Saved: {info_file}")
    
    # 6. Print summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total sequences processed: {len(features_df)}")
    print(f"Total features extracted: {len(feature_cols)}")
    print(f"\nFeature groups:")
    for group, count in feature_info['feature_groups'].items():
        print(f"  - {group}: {count} features")
    
    print(f"\nFeature statistics:")
    numeric_cols = features_only.select_dtypes(include=[np.number]).columns
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Missing values: {features_only.isnull().sum().sum()}")
    
    # Year distribution in features
    if 'collection_year' in features_df.columns:
        year_dist = features_df['collection_year'].value_counts().sort_index()
        print(f"\nYear distribution (top 10):")
        for year, count in year_dist.tail(10).items():
            print(f"  {int(year)}: {count}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    
    return features_df


if __name__ == "__main__":
    main()
