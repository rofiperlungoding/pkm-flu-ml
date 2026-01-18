"""
Modul ekstraksi fitur dari sekuens protein HA
"""
import numpy as np
from .physicochemical import ALL_PROPERTIES, EPITOPE_SITES


def find_mutations(seq, reference):
    """Identifikasi mutasi antara sekuens dan referensi"""
    mutations = []
    for i, (ref_aa, seq_aa) in enumerate(zip(reference, seq)):
        if ref_aa != seq_aa and ref_aa not in '-X' and seq_aa not in '-X':
            mutations.append({
                'position': i,
                'from': ref_aa,
                'to': seq_aa
            })
    return mutations


def extract_features(seq, reference):
    """
    Ekstrak fitur fisikokimia dari sekuens
    
    Returns:
        dict: Dictionary berisi semua fitur
    """
    mutations = find_mutations(seq, reference)
    features = {}
    
    # Total mutasi
    features['total_mutations'] = len(mutations)
    
    # Mutasi per epitope site
    for site, positions in EPITOPE_SITES.items():
        site_muts = [m for m in mutations if m['position'] in positions]
        features[f'mutations_site_{site}'] = len(site_muts)

    
    # Delta fisikokimia
    for prop_name, prop_dict in ALL_PROPERTIES.items():
        delta = 0
        for m in mutations:
            if m['from'] in prop_dict and m['to'] in prop_dict:
                delta += abs(prop_dict[m['to']] - prop_dict[m['from']])
        features[f'delta_{prop_name}'] = delta
    
    return features


def extract_features_batch(sequences, reference):
    """Ekstrak fitur untuk banyak sekuens sekaligus"""
    import pandas as pd
    
    all_features = []
    for seq in sequences:
        feat = extract_features(seq, reference)
        all_features.append(feat)
    
    return pd.DataFrame(all_features)


class FeatureExtractor:
    """
    Class wrapper for feature extraction functions
    Provides object-oriented interface for compatibility with advanced scripts
    """
    
    def __init__(self, reference_seq):
        """
        Initialize feature extractor with reference sequence
        
        Args:
            reference_seq: Reference protein sequence (e.g., A/Perth/16/2009)
        """
        self.reference_seq = reference_seq
    
    def extract_all_features(self, sequence):
        """
        Extract all features from a sequence
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            dict: Dictionary of features
        """
        return extract_features(sequence, self.reference_seq)
    
    def extract_batch(self, sequences):
        """
        Extract features for multiple sequences
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            pd.DataFrame: DataFrame with features for all sequences
        """
        return extract_features_batch(sequences, self.reference_seq)