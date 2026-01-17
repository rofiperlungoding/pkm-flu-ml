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