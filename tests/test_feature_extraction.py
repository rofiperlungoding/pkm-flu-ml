"""
Unit Tests for Feature Extraction
==================================
Tests for basic and advanced feature extraction

Author: PKM-RE Team
Date: 2026-01-18
"""
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import FeatureExtractor
from src.physicochemical import PhysicochemicalCalculator

class TestPhysicochemicalCalculator(unittest.TestCase):
    """Test physicochemical calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calc = PhysicochemicalCalculator()
        self.test_sequence = "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAETQTAEKQTRMVTLLRNHCRQEQGAIYSLIRPNENPAHKSQLVWMACHSAAFEDLRLLSFIRGTKV"
    
    def test_amino_acid_composition(self):
        """Test amino acid composition calculation"""
        comp = self.calc.amino_acid_composition(self.test_sequence)
        
        # Check all 20 amino acids are present
        self.assertEqual(len(comp), 20)
        
        # Check values sum to 1.0
        self.assertAlmostEqual(sum(comp.values()), 1.0, places=5)
        
        # Check all values are between 0 and 1
        for aa, freq in comp.items():
            self.assertGreaterEqual(freq, 0)
            self.assertLessEqual(freq, 1)
    
    def test_hydrophobicity(self):
        """Test hydrophobicity calculation"""
        hydro = self.calc.calculate_hydrophobicity(self.test_sequence)
        
        self.assertIn('mean_hydrophobicity', hydro)
        self.assertIn('hydrophobicity_variance', hydro)
        self.assertIn('hydrophobic_fraction', hydro)
        self.assertIn('hydrophilic_fraction', hydro)
        
        # Check fractions sum to 1.0
        self.assertAlmostEqual(
            hydro['hydrophobic_fraction'] + hydro['hydrophilic_fraction'],
            1.0,
            places=5
        )
    
    def test_charge_properties(self):
        """Test charge property calculation"""
        charge = self.calc.calculate_charge_properties(self.test_sequence)
        
        self.assertIn('mean_charge', charge)
        self.assertIn('positive_charge_fraction', charge)
        self.assertIn('negative_charge_fraction', charge)
        self.assertIn('net_charge', charge)
        
        # Check fractions are valid
        self.assertGreaterEqual(charge['positive_charge_fraction'], 0)
        self.assertLessEqual(charge['positive_charge_fraction'], 1)
    
    def test_polarity(self):
        """Test polarity calculation"""
        polarity = self.calc.calculate_polarity(self.test_sequence)
        
        self.assertIn('polar_fraction', polarity)
        self.assertIn('nonpolar_fraction', polarity)
        
        # Check fractions sum to 1.0
        self.assertAlmostEqual(
            polarity['polar_fraction'] + polarity['nonpolar_fraction'],
            1.0,
            places=5
        )
    
    def test_aromaticity(self):
        """Test aromaticity calculation"""
        arom = self.calc.calculate_aromaticity(self.test_sequence)
        
        self.assertIn('aromaticity', arom)
        self.assertGreaterEqual(arom['aromaticity'], 0)
        self.assertLessEqual(arom['aromaticity'], 1)

class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reference_seq = "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAETQTAEKQTRMVTLLRNHCRQEQGAIYSLIRPNENPAHKSQLVWMACHSAAFEDLRLLSFIRGTKV"
        self.test_sequence = self.reference_seq  # Same for testing
        self.extractor = FeatureExtractor(self.reference_seq)
    
    def test_extract_all_features(self):
        """Test complete feature extraction"""
        features = self.extractor.extract_all_features(self.test_sequence)
        
        # Check we have all expected features (74)
        self.assertGreaterEqual(len(features), 70)
        
        # Check all values are numeric
        for key, value in features.items():
            self.assertIsInstance(value, (int, float, np.integer, np.floating))
    
    def test_epitope_mutations(self):
        """Test epitope mutation calculation"""
        # Test with identical sequence (should have 0 mutations)
        features = self.extractor.extract_all_features(self.reference_seq)
        self.assertEqual(features['epitope_site_mutations'], 0)
        
        # Test with modified sequence
        modified_seq = list(self.reference_seq)
        modified_seq[122] = 'A'  # Mutate position in epitope site A
        modified_seq = ''.join(modified_seq)
        
        features = self.extractor.extract_all_features(modified_seq)
        self.assertGreater(features['epitope_site_mutations'], 0)
    
    def test_sequence_length_validation(self):
        """Test handling of different sequence lengths"""
        # Short sequence
        short_seq = "MKTIIALSYILCLVFAQKLP"
        features = self.extractor.extract_all_features(short_seq)
        self.assertIsInstance(features, dict)
        
        # Very long sequence
        long_seq = self.test_sequence * 2
        features = self.extractor.extract_all_features(long_seq)
        self.assertIsInstance(features, dict)
    
    def test_invalid_amino_acids(self):
        """Test handling of invalid amino acids"""
        # Sequence with invalid characters should raise error or handle gracefully
        invalid_seq = "MKTIIALSYILCLVFAQKLPXXX"
        
        try:
            features = self.extractor.extract_all_features(invalid_seq)
            # If it doesn't raise error, check it returns valid features
            self.assertIsInstance(features, dict)
        except ValueError:
            # Expected behavior - invalid amino acids should raise error
            pass

class TestFeatureConsistency(unittest.TestCase):
    """Test feature consistency and reproducibility"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reference_seq = "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAETQTAEKQTRMVTLLRNHCRQEQGAIYSLIRPNENPAHKSQLVWMACHSAAFEDLRLLSFIRGTKV"
        self.extractor = FeatureExtractor(self.reference_seq)
    
    def test_reproducibility(self):
        """Test that same sequence produces same features"""
        features1 = self.extractor.extract_all_features(self.reference_seq)
        features2 = self.extractor.extract_all_features(self.reference_seq)
        
        # Check all features are identical
        for key in features1.keys():
            self.assertAlmostEqual(features1[key], features2[key], places=10)
    
    def test_feature_ranges(self):
        """Test that features are within expected ranges"""
        features = self.extractor.extract_all_features(self.reference_seq)
        
        # Amino acid compositions should be between 0 and 1
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            key = f'aa_{aa}'
            if key in features:
                self.assertGreaterEqual(features[key], 0)
                self.assertLessEqual(features[key], 1)
        
        # Fractions should be between 0 and 1
        fraction_keys = [
            'hydrophobic_fraction', 'hydrophilic_fraction',
            'positive_charge_fraction', 'negative_charge_fraction',
            'polar_fraction', 'nonpolar_fraction', 'aromaticity'
        ]
        for key in fraction_keys:
            if key in features:
                self.assertGreaterEqual(features[key], 0)
                self.assertLessEqual(features[key], 1)

if __name__ == '__main__':
    unittest.main()
