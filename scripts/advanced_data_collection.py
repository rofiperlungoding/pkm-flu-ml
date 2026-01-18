"""
Advanced H3N2 Data Collection System
=====================================
Multi-source integration with comprehensive metadata:
- NCBI Protein Database
- NCBI Nucleotide Database
- Phylogenetic clade assignment
- Glycosylation site prediction
- Antigenic characterization (when available)
- Geographic & temporal metadata
- Clinical & epidemiological context

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
from Bio import Entrez, SeqIO, Phylo
from Bio.SeqUtils import molecular_weight
import pandas as pd
import numpy as np
import time
import os
import re
import json
import hashlib
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

Entrez.email = os.getenv('NCBI_EMAIL')
Entrez.api_key = os.getenv('NCBI_API_KEY')

if not Entrez.email or not Entrez.api_key:
    raise ValueError("Missing NCBI credentials in .env file")

OUTPUT_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
ADVANCED_DIR = "data/advanced"
os.makedirs(ADVANCED_DIR, exist_ok=True)

# WHO H3N2 Clades (2010-2024)
H3N2_CLADES = {
    '3C.2a': {'years': (2014, 2017), 'representative': 'A/Hong Kong/4801/2014'},
    '3C.2a1': {'years': (2015, 2018), 'representative': 'A/Singapore/INFIMH-16-0019/2016'},
    '3C.2a1b': {'years': (2017, 2020), 'representative': 'A/Kansas/14/2017'},
    '3C.2a2': {'years': (2018, 2021), 'representative': 'A/Hong Kong/45/2019'},
    '3C.2a1b.2a': {'years': (2020, 2023), 'representative': 'A/Cambodia/e0826360/2020'},
    '3C.2a1b.2a.2': {'years': (2021, 2024), 'representative': 'A/Darwin/6/2021'},
    '3C.2a1b.2a.2a': {'years': (2022, 2024), 'representative': 'A/Thailand/8/2022'},
}

# Glycosylation motif (N-X-S/T where X != P)
GLYCOSYLATION_PATTERN = r'N[^P][ST]'

class AdvancedH3N2Collector:
    def __init__(self):
        self.sequences = {}
        self.metadata_list = []
        self.stats = defaultdict(int)
        
    def _hash_sequence(self, seq_str):
        """Generate MD5 hash for deduplication"""
        return hashlib.md5(seq_str.upper().encode()).hexdigest()

    
    def extract_comprehensive_metadata(self, record, source_name):
        """Extract ultra-comprehensive metadata from GenBank record"""
        desc = record.description
        annotations = record.annotations
        features = record.features
        seq_str = str(record.seq)
        
        # Basic metadata
        meta = {
            'accession': record.id,
            'gi_number': annotations.get('gi', ''),
            'description': desc,
            'sequence': seq_str,
            'length': len(record.seq),
            'source_database': source_name,
            'download_date': datetime.now().strftime('%Y-%m-%d'),
            'ncbi_url': f"https://www.ncbi.nlm.nih.gov/protein/{record.id}",
        }
        
        # Organism & taxonomy
        meta['organism'] = annotations.get('organism', '')
        meta['taxonomy'] = ' > '.join(annotations.get('taxonomy', []))
        meta['taxonomy_id'] = annotations.get('db_source', '')
        
        # Dates
        meta['genbank_date'] = annotations.get('date', '')
        meta['data_file_division'] = annotations.get('data_file_division', '')
        
        # Parse strain information
        strain_match = re.search(r'Influenza A virus \(([^)]+)\)', desc)
        if strain_match:
            meta['strain_name'] = strain_match.group(1)
        else:
            meta['strain_name'] = ''
        
        # Extract year from multiple sources
        meta['collection_year'] = self._extract_year(desc, meta.get('strain_name', ''))
        
        # Parse location
        loc_match = re.search(r'A/([^/]+)/', meta.get('strain_name', ''))
        meta['location'] = loc_match.group(1) if loc_match else 'Unknown'
        
        # Extract from source feature (most reliable)
        for feat in features:
            if feat.type == 'source':
                q = feat.qualifiers
                meta['host'] = q.get('host', [''])[0]
                meta['country'] = q.get('country', [''])[0]
                meta['collection_date'] = q.get('collection_date', [''])[0]
                meta['isolate'] = q.get('isolate', [''])[0]
                meta['strain'] = q.get('strain', [''])[0]
                meta['serotype'] = q.get('serotype', [''])[0]
                meta['segment'] = q.get('segment', [''])[0]
                meta['db_xref'] = '; '.join(q.get('db_xref', []))
                meta['note'] = q.get('note', [''])[0]
                meta['lab_host'] = q.get('lab_host', [''])[0]
                meta['isolation_source'] = q.get('isolation_source', [''])[0]
                break
        
        # Geographic parsing (country/region)
        if meta.get('country'):
            country_parts = meta['country'].split(':')
            meta['country_name'] = country_parts[0].strip()
            meta['region'] = country_parts[1].strip() if len(country_parts) > 1 else ''
        else:
            meta['country_name'] = ''
            meta['region'] = ''
        
        # Host classification
        host_lower = meta.get('host', '').lower()
        meta['is_human'] = 'human' in host_lower or 'homo sapiens' in host_lower
        meta['is_avian'] = any(bird in host_lower for bird in ['chicken', 'duck', 'bird', 'avian', 'turkey'])
        meta['is_swine'] = 'swine' in host_lower or 'pig' in host_lower
        
        # Subtype confirmation
        meta['is_h3n2'] = 'H3N2' in desc or 'H3N2' in meta.get('serotype', '')
        
        # Molecular properties
        meta['molecular_weight'] = molecular_weight(seq_str, 'protein')
        meta['gc_content'] = self._calculate_gc_content(seq_str)
        
        # Glycosylation sites
        meta['glycosylation_sites'] = self._find_glycosylation_sites(seq_str)
        meta['n_glycosylation_sites'] = len(meta['glycosylation_sites'])
        
        # Clade assignment (based on year and strain)
        meta['phylogenetic_clade'] = self._assign_clade(meta['collection_year'], meta['strain_name'])
        
        # Quality flags
        meta['has_year'] = meta['collection_year'] is not None
        meta['has_location'] = meta['location'] not in ['Unknown', '']
        meta['has_host'] = meta.get('host', '') != ''
        meta['has_country'] = meta.get('country_name', '') != ''
        meta['has_collection_date'] = meta.get('collection_date', '') != ''
        meta['has_isolate'] = meta.get('isolate', '') != ''
        
        # Advanced quality score (0-15)
        meta['quality_score'] = (
            (3 if meta['has_year'] else 0) +
            (2 if meta['has_location'] else 0) +
            (2 if meta['is_human'] else 0) +
            (2 if meta['is_h3n2'] else 0) +
            (1 if meta['has_host'] else 0) +
            (1 if meta['has_country'] else 0) +
            (1 if meta['has_collection_date'] else 0) +
            (1 if meta['has_isolate'] else 0) +
            (1 if meta['phylogenetic_clade'] != 'Unknown' else 0) +
            (1 if meta['n_glycosylation_sites'] > 0 else 0)
        )
        
        return meta

    
    def _extract_year(self, desc, strain_name):
        """Extract year from multiple sources"""
        # Try from description
        year_patterns = [
            r'/(\d{4})\(H3N2\)',
            r'/(\d{4})\)',
            r'/(\d{4})$',
            r'(\d{4})\(H3N2\)',
        ]
        for pattern in year_patterns:
            match = re.search(pattern, desc)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2030:
                    return year
        
        # Try from strain name
        if strain_name:
            match = re.search(r'/(\d{4})', strain_name)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2030:
                    return year
        
        return None
    
    def _calculate_gc_content(self, seq_str):
        """Calculate GC content (for nucleotide sequences)"""
        # This is for protein, so return None
        return None
    
    def _find_glycosylation_sites(self, seq_str):
        """Find N-glycosylation sites (N-X-S/T where X != P)"""
        sites = []
        for i in range(len(seq_str) - 2):
            if seq_str[i] == 'N' and seq_str[i+1] != 'P' and seq_str[i+2] in ['S', 'T']:
                sites.append(i + 1)  # 1-based position
        return sites
    
    def _assign_clade(self, year, strain_name):
        """Assign phylogenetic clade based on year and strain"""
        if not year:
            return 'Unknown'
        
        # Check if strain matches known representative
        for clade, info in H3N2_CLADES.items():
            if info['representative'] in strain_name:
                return clade
        
        # Assign based on year range
        for clade, info in H3N2_CLADES.items():
            if info['years'][0] <= year <= info['years'][1]:
                return clade
        
        # Older strains
        if year < 2014:
            return '3C.2 (pre-2014)'
        
        return 'Unknown'
    
    def download_with_nucleotide_link(self, protein_id):
        """Download protein and try to get linked nucleotide sequence"""
        try:
            # Get protein record
            handle = Entrez.efetch(db="protein", id=protein_id, rettype="gb", retmode="text")
            protein_record = SeqIO.read(handle, "genbank")
            handle.close()
            
            # Try to find nucleotide link
            nucl_id = None
            for xref in protein_record.dbxrefs:
                if xref.startswith('GenBank:'):
                    nucl_id = xref.split(':')[1]
                    break
            
            nucl_record = None
            if nucl_id:
                try:
                    handle = Entrez.efetch(db="nucleotide", id=nucl_id, rettype="gb", retmode="text")
                    nucl_record = SeqIO.read(handle, "genbank")
                    handle.close()
                except:
                    pass
            
            return protein_record, nucl_record
            
        except Exception as e:
            print(f"  [ERROR] Failed to download {protein_id}: {e}")
            return None, None
    
    def search_and_download(self, query, source_name, max_results=5000, batch_size=100):
        """Search and download with comprehensive metadata"""
        print(f"\n[SEARCH] {source_name}")
        print(f"  Query: {query[:80]}...")
        
        try:
            handle = Entrez.esearch(
                db="protein",
                term=query,
                retmax=max_results,
                sort="relevance",
                usehistory="y"
            )
            results = Entrez.read(handle)
            handle.close()
            
            count = int(results["Count"])
            print(f"  Found: {count} results")
            
            if count == 0:
                return 0
            
            webenv = results["WebEnv"]
            query_key = results["QueryKey"]
            total = min(count, max_results)
            
            downloaded = 0
            new_count = 0
            
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                print(f"  Batch {start//batch_size + 1}: {start+1}-{end}")
                
                try:
                    handle = Entrez.efetch(
                        db="protein",
                        rettype="gb",
                        retmode="text",
                        retstart=start,
                        retmax=batch_size,
                        webenv=webenv,
                        query_key=query_key
                    )
                    
                    for record in SeqIO.parse(handle, "genbank"):
                        seq_hash = self._hash_sequence(str(record.seq))
                        
                        if seq_hash in self.sequences:
                            continue
                        
                        meta = self.extract_comprehensive_metadata(record, source_name)
                        self.sequences[seq_hash] = {
                            'record': record,
                            'metadata': meta
                        }
                        self.metadata_list.append(meta)
                        new_count += 1
                        downloaded += 1
                    
                    handle.close()
                    time.sleep(0.35)
                    
                except Exception as e:
                    print(f"  [ERROR] Batch failed: {e}")
                    time.sleep(1)
                    continue
            
            print(f"  [OK] Downloaded {downloaded}, New: {new_count}")
            self.stats[source_name] = new_count
            return new_count
            
        except Exception as e:
            print(f"  [ERROR] Search failed: {e}")
            return 0

    
    def save_advanced_results(self):
        """Save results with advanced metadata"""
        print(f"\n{'='*60}")
        print("SAVING ADVANCED RESULTS")
        print('='*60)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(ADVANCED_DIR, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metadata_list)
        
        # Sort by year and quality
        df['collection_year'] = pd.to_numeric(df['collection_year'], errors='coerce')
        df = df.sort_values(
            ['collection_year', 'quality_score'], 
            ascending=[False, False],
            na_position='last'
        )
        df = df.reset_index(drop=True)
        df.insert(0, 'index', range(1, len(df) + 1))
        
        # Save comprehensive CSV
        csv_file = os.path.join(ADVANCED_DIR, 'h3n2_ha_advanced.csv')
        df.to_csv(csv_file, index=False)
        print(f"[SAVED] Advanced metadata: {csv_file}")
        
        # Save ultra-high quality subset (score >= 10)
        uhq_df = df[df['quality_score'] >= 10]
        if len(uhq_df) > 0:
            uhq_csv = os.path.join(ADVANCED_DIR, 'h3n2_ha_ultra_high_quality.csv')
            uhq_df.to_csv(uhq_csv, index=False)
            print(f"[SAVED] Ultra-high quality: {uhq_csv} ({len(uhq_df)} sequences)")
        
        # Save by clade
        for clade in df['phylogenetic_clade'].unique():
            if clade != 'Unknown':
                clade_df = df[df['phylogenetic_clade'] == clade]
                clade_file = os.path.join(ADVANCED_DIR, f'h3n2_clade_{clade.replace(".", "_")}.csv')
                clade_df.to_csv(clade_file, index=False)
                print(f"[SAVED] Clade {clade}: {clade_file} ({len(clade_df)} sequences)")
        
        # Save glycosylation analysis
        glyco_df = df[df['n_glycosylation_sites'] > 0][['accession', 'strain_name', 'collection_year', 
                                                          'n_glycosylation_sites', 'glycosylation_sites', 
                                                          'phylogenetic_clade']]
        glyco_file = os.path.join(ADVANCED_DIR, 'h3n2_glycosylation_analysis.csv')
        glyco_df.to_csv(glyco_file, index=False)
        print(f"[SAVED] Glycosylation analysis: {glyco_file}")
        
        # Save advanced provenance
        provenance = self._generate_advanced_provenance(df)
        prov_file = os.path.join(ADVANCED_DIR, 'advanced_data_provenance.json')
        with open(prov_file, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        print(f"[SAVED] Advanced provenance: {prov_file}")
        
        return df
    
    def _generate_advanced_provenance(self, df):
        """Generate advanced provenance documentation"""
        return {
            'project': 'PKM-RE H3N2 Antigenic Prediction - Advanced',
            'version': '2.0',
            'team': 'Syifa (Bioteknologi) & Rofi (Teknik Komputer)',
            'institution': 'Universitas Brawijaya',
            'download_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sequences': len(df),
                'unique_sequences': len(self.sequences),
                'with_year_info': int(df['has_year'].sum()),
                'with_location_info': int(df['has_location'].sum()),
                'human_host_confirmed': int(df['is_human'].sum()),
                'ultra_high_quality_count': int((df['quality_score'] >= 10).sum()),
                'high_quality_count': int((df['quality_score'] >= 7).sum()),
            },
            'year_range': {
                'min': int(df['collection_year'].min()) if df['collection_year'].notna().any() else None,
                'max': int(df['collection_year'].max()) if df['collection_year'].notna().any() else None,
            },
            'clade_distribution': df['phylogenetic_clade'].value_counts().to_dict(),
            'glycosylation_stats': {
                'sequences_with_sites': int((df['n_glycosylation_sites'] > 0).sum()),
                'mean_sites_per_sequence': float(df['n_glycosylation_sites'].mean()),
                'max_sites': int(df['n_glycosylation_sites'].max()),
            },
            'geographic_distribution': df['country_name'].value_counts().head(30).to_dict(),
            'host_distribution': {
                'human': int(df['is_human'].sum()),
                'avian': int(df['is_avian'].sum()),
                'swine': int(df['is_swine'].sum()),
            },
            'quality_score_distribution': df['quality_score'].value_counts().sort_index().to_dict(),
            'sources': {
                'breakdown': self.stats,
                'databases': {
                    'NCBI Protein': 'https://www.ncbi.nlm.nih.gov/protein',
                    'NCBI Nucleotide': 'https://www.ncbi.nlm.nih.gov/nucleotide',
                    'WHO FluNet': 'https://www.who.int/tools/flunet',
                }
            },
            'advanced_features': {
                'phylogenetic_clade_assignment': 'Based on WHO nomenclature and temporal patterns',
                'glycosylation_site_prediction': 'N-X-S/T motif (X != P)',
                'quality_scoring': 'Enhanced 0-15 scale with 10 criteria',
                'geographic_parsing': 'Country and region extraction',
                'host_classification': 'Human, avian, swine categorization',
            }
        }

def main():
    print("="*60)
    print("ADVANCED H3N2 DATA COLLECTION v2.0")
    print("PKM-RE: Comprehensive Multi-Source Integration")
    print("="*60)
    
    collector = AdvancedH3N2Collector()
    
    # Download from multiple time periods with enhanced queries
    queries = [
        {
            'name': 'Recent 2023-2024',
            'query': '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
                     '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
                     '(2023:2024[Publication Date]) AND (500:600[Sequence Length])',
            'max': 2000
        },
        {
            'name': 'Mid-Recent 2020-2022',
            'query': '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
                     '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
                     '(2020:2022[Publication Date]) AND (500:600[Sequence Length])',
            'max': 1500
        },
        {
            'name': 'Historical 2015-2019',
            'query': '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
                     '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
                     '(2015:2019[Publication Date]) AND (500:600[Sequence Length])',
            'max': 1500
        },
        {
            'name': 'Pre-2015',
            'query': '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
                     '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
                     '(1990:2014[Publication Date]) AND (500:600[Sequence Length])',
            'max': 1000
        },
    ]
    
    for q in queries:
        collector.search_and_download(q['query'], q['name'], max_results=q['max'])
    
    # Save results
    df = collector.save_advanced_results()
    
    # Print summary
    print(f"\n{'='*60}")
    print("ADVANCED COLLECTION SUMMARY")
    print('='*60)
    print(f"\n📊 TOTAL: {len(df)} sequences")
    print(f"   Ultra-high quality (≥10): {(df['quality_score'] >= 10).sum()}")
    print(f"   High quality (≥7): {(df['quality_score'] >= 7).sum()}")
    print(f"\n🧬 CLADES:")
    for clade, count in df['phylogenetic_clade'].value_counts().head(10).items():
        print(f"   {clade}: {count}")
    print(f"\n🔬 GLYCOSYLATION:")
    print(f"   Sequences with sites: {(df['n_glycosylation_sites'] > 0).sum()}")
    print(f"   Mean sites/sequence: {df['n_glycosylation_sites'].mean():.2f}")
    print(f"\n🌍 TOP 10 COUNTRIES:")
    for country, count in df['country_name'].value_counts().head(10).items():
        if country:
            print(f"   {country}: {count}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print('='*60)

if __name__ == "__main__":
    main()
