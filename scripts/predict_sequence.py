"""
H3N2 Sequence Prediction Interface
===================================
Predict temporal period and antigenic properties for new H3N2 HA sequences

Usage:
  python scripts/predict_sequence.py --sequence "MKTII..." 
  python scripts/predict_sequence.py --fasta input.fasta
  python scripts/predict_sequence.py --accession ABC12345

Author: PKM-RE Team
Date: 2026-01-18
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import FeatureExtractor
from src.physicochemical import PhysicochemicalCalculator

# Load environment variables
load_dotenv()
Entrez.email = os.getenv('NCBI_EMAIL', 'your_email@example.com')
Entrez.api_key = os.getenv('NCBI_API_KEY')

# Directories
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

class H3N2Predictor:
    def __init__(self):
        """Initialize predictor with trained models"""
        print("Loading models...")
        
        self.binary_model = joblib.load(
            os.path.join(MODELS_DIR, 'h3n2_binary_model.pkl')
        )
        self.multiclass_model = joblib.load(
            os.path.join(MODELS_DIR, 'h3n2_multiclass_model.pkl')
        )
        
        # Load reference strain
        self.reference_seq = self._load_reference()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.reference_seq)
        
        # Period labels
        self.period_labels = {
            0: '<2010 (Historical)',
            1: '2010-2014 (Mid-Historical)',
            2: '2015-2019 (Mid-Recent)',
            3: '≥2020 (Recent)'
        }
        
        print("✅ Models loaded successfully")
    
    def _load_reference(self):
        """Load reference strain sequence"""
        # A/Perth/16/2009 reference
        ref_file = os.path.join(PROCESSED_DIR, 'h3n2_ha_comprehensive.csv')
        if os.path.exists(ref_file):
            df = pd.read_csv(ref_file)
            perth = df[df['strain_name'].str.contains('Perth/16/2009', na=False)]
            if len(perth) > 0:
                return perth.iloc[0]['sequence']
        
        # Fallback: use first sequence from dataset
        df = pd.read_csv(ref_file)
        return df.iloc[0]['sequence']
    
    def predict_from_sequence(self, sequence, seq_id="Unknown"):
        """Predict temporal period from sequence string"""
        print(f"\n{'='*60}")
        print(f"Predicting: {seq_id}")
        print('='*60)
        
        # Validate sequence
        if not self._validate_sequence(sequence):
            print("❌ Invalid sequence!")
            return None
        
        print(f"Sequence length: {len(sequence)} aa")
        
        # Extract features
        print("Extracting features...")
        features = self.feature_extractor.extract_all_features(sequence)
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Predictions
        print("\n📊 PREDICTIONS:")
        
        # Binary prediction
        binary_pred = self.binary_model.predict(X)[0]
        binary_proba = self.binary_model.predict_proba(X)[0]
        
        binary_label = "Recent (≥2020)" if binary_pred == 1 else "Historical (<2020)"
        binary_conf = binary_proba[binary_pred] * 100
        
        print(f"\n  Binary Classification:")
        print(f"    Prediction: {binary_label}")
        print(f"    Confidence: {binary_conf:.2f}%")
        print(f"    Probabilities:")
        print(f"      Historical (<2020): {binary_proba[0]*100:.2f}%")
        print(f"      Recent (≥2020):     {binary_proba[1]*100:.2f}%")
        
        # Multi-class prediction
        multi_pred = self.multiclass_model.predict(X)[0]
        multi_proba = self.multiclass_model.predict_proba(X)[0]
        
        multi_label = self.period_labels[multi_pred]
        multi_conf = multi_proba[multi_pred] * 100
        
        print(f"\n  Temporal Period Classification:")
        print(f"    Prediction: {multi_label}")
        print(f"    Confidence: {multi_conf:.2f}%")
        print(f"    Probabilities:")
        for i, prob in enumerate(multi_proba):
            print(f"      {self.period_labels[i]:30s}: {prob*100:.2f}%")
        
        # Feature highlights
        print(f"\n  Key Features:")
        print(f"    Hydrophobicity: {features.get('mean_hydrophobicity', 0):.4f}")
        print(f"    Charge: {features.get('mean_charge', 0):.4f}")
        print(f"    Aromaticity: {features.get('aromaticity', 0):.4f}")
        print(f"    Epitope mutations: {features.get('epitope_site_mutations', 0)}")
        
        result = {
            'sequence_id': seq_id,
            'sequence_length': len(sequence),
            'binary_prediction': binary_label,
            'binary_confidence': float(binary_conf),
            'binary_probabilities': {
                'historical': float(binary_proba[0]),
                'recent': float(binary_proba[1])
            },
            'multiclass_prediction': multi_label,
            'multiclass_confidence': float(multi_conf),
            'multiclass_probabilities': {
                self.period_labels[i]: float(prob) 
                for i, prob in enumerate(multi_proba)
            },
            'features': features
        }
        
        return result
    
    def predict_from_fasta(self, fasta_file):
        """Predict for all sequences in FASTA file"""
        print(f"\n{'='*60}")
        print(f"Processing FASTA: {fasta_file}")
        print('='*60)
        
        if not os.path.exists(fasta_file):
            print(f"❌ File not found: {fasta_file}")
            return []
        
        results = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            result = self.predict_from_sequence(
                str(record.seq), 
                seq_id=record.id
            )
            if result:
                results.append(result)
        
        return results
    
    def predict_from_accession(self, accession):
        """Download sequence from NCBI and predict"""
        print(f"\n{'='*60}")
        print(f"Fetching from NCBI: {accession}")
        print('='*60)
        
        try:
            handle = Entrez.efetch(
                db="protein",
                id=accession,
                rettype="fasta",
                retmode="text"
            )
            record = SeqIO.read(handle, "fasta")
            handle.close()
            
            print(f"✅ Downloaded: {record.description}")
            
            return self.predict_from_sequence(
                str(record.seq),
                seq_id=f"{accession} - {record.description}"
            )
            
        except Exception as e:
            print(f"❌ Error fetching sequence: {e}")
            return None
    
    def _validate_sequence(self, sequence):
        """Validate protein sequence"""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        seq_upper = sequence.upper()
        
        # Check if all characters are valid amino acids
        if not all(aa in valid_aa for aa in seq_upper):
            invalid = set(seq_upper) - valid_aa
            print(f"Invalid amino acids found: {invalid}")
            return False
        
        # Check length (HA should be ~550 aa)
        if len(sequence) < 400 or len(sequence) > 700:
            print(f"Warning: Unusual sequence length ({len(sequence)} aa)")
            print("Expected: 400-700 aa for H3N2 HA")
        
        return True
    
    def save_results(self, results, output_file):
        """Save prediction results to JSON"""
        import json
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Predict temporal period for H3N2 HA sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from sequence string
  python scripts/predict_sequence.py --sequence "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAETQTAEKQTRMVTLLRNHCRQEQGAIYSLIRPNENPAHKSQLVWMACHSAAFEDLRLLSFIRGTKV"
  
  # Predict from FASTA file
  python scripts/predict_sequence.py --fasta data/raw/new_sequences.fasta
  
  # Predict from NCBI accession
  python scripts/predict_sequence.py --accession ABO21709.1
  
  # Save results to file
  python scripts/predict_sequence.py --fasta input.fasta --output predictions.json
        """
    )
    
    parser.add_argument('--sequence', type=str, help='Protein sequence string')
    parser.add_argument('--fasta', type=str, help='FASTA file path')
    parser.add_argument('--accession', type=str, help='NCBI accession number')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.sequence, args.fasta, args.accession]):
        parser.print_help()
        sys.exit(1)
    
    # Initialize predictor
    predictor = H3N2Predictor()
    
    # Perform prediction
    results = []
    
    if args.sequence:
        result = predictor.predict_from_sequence(args.sequence)
        if result:
            results.append(result)
    
    if args.fasta:
        results.extend(predictor.predict_from_fasta(args.fasta))
    
    if args.accession:
        result = predictor.predict_from_accession(args.accession)
        if result:
            results.append(result)
    
    # Save results if output specified
    if args.output and results:
        predictor.save_results(results, args.output)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
