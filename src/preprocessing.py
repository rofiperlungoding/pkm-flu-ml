"""
Modul preprocessing data sekuens FASTA
"""
from Bio import SeqIO
import pandas as pd
import re


def load_fasta(filepath):
    """Load file FASTA dan konversi ke DataFrame"""
    records = []
    for record in SeqIO.parse(filepath, "fasta"):
        records.append({
            'id': record.id,
            'description': record.description,
            'sequence': str(record.seq),
            'length': len(record.seq)
        })
    return pd.DataFrame(records)


def extract_metadata(df):
    """Ekstrak tahun dan lokasi dari description"""
    # Pattern untuk tahun (4 digit)
    df['year'] = df['description'].str.extract(r'(\d{4})')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df


def filter_sequences(df, min_len=550, max_len=600):
    """Filter sekuens berdasarkan panjang dan kualitas"""
    # Filter panjang
    df = df[(df['length'] >= min_len) & (df['length'] <= max_len)]
    
    # Hapus sekuens dengan karakter invalid
    df = df[~df['sequence'].str.contains(r'[X\-\*]', regex=True)]
    
    return df.reset_index(drop=True)


def preprocess_pipeline(filepath, min_len=550, max_len=600):
    """Pipeline lengkap preprocessing"""
    df = load_fasta(filepath)
    df = extract_metadata(df)
    df = filter_sequences(df, min_len, max_len)
    
    print(f"Loaded {len(df)} sequences after filtering")
    return df