"""
Enhanced H3N2 HA Data Downloader v2.0
=====================================
- Downloads from NCBI with comprehensive metadata
- Prioritizes recent sequences (2020-2026)
- Includes WHO vaccine reference strains
- Removes duplicates with existing data
- Complete labelling with provenance tracking

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd
import time
import os
import re
import json
import hashlib
from datetime import datetime
from collections import defaultdict

# ============== CONFIGURATION ==============
Entrez.email = "mrofid@student.ub.ac.id"
Entrez.api_key = "153134c1b56a48d7acb46f82a8b6aba1f509"

OUTPUT_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
EXISTING_FASTA = "data/raw/h3n2_ha_sequences.fasta"

# WHO Vaccine Reference Strains (2015-2026)
VACCINE_STRAINS = [
    # 2024-2026 Season
    ("A/Thailand/8/2022", "2022", "Thailand", "WHO 2024-25 NH"),
    ("A/Massachusetts/18/2022", "2022", "USA", "WHO 2024-25 SH"),
    # 2023-2024 Season
    ("A/Darwin/6/2021", "2021", "Australia", "WHO 2023-24"),
    ("A/Darwin/9/2021", "2021", "Australia", "WHO 2023-24 egg"),
    # 2022-2023 Season
    ("A/Darwin/11/2021", "2021", "Australia", "WHO 2022-23"),
    # 2021-2022 Season
    ("A/Cambodia/e0826360/2020", "2020", "Cambodia", "WHO 2021-22"),
    ("A/Hong Kong/2671/2019", "2019", "Hong Kong", "WHO 2021-22 egg"),
    # 2020-2021 Season
    ("A/Hong Kong/45/2019", "2019", "Hong Kong", "WHO 2020-21"),
    # 2019-2020 Season
    ("A/Kansas/14/2017", "2017", "USA", "WHO 2019-20"),
    # 2018-2019 Season
    ("A/Singapore/INFIMH-16-0019/2016", "2016", "Singapore", "WHO 2018-19"),
    # 2017-2018 Season
    ("A/Hong Kong/4801/2014", "2014", "Hong Kong", "WHO 2017-18"),
    # Historical references
    ("A/Switzerland/9715293/2013", "2013", "Switzerland", "WHO 2015-16"),
    ("A/Texas/50/2012", "2012", "USA", "WHO 2014-15"),
    ("A/Victoria/361/2011", "2011", "Australia", "WHO 2013-14"),
    ("A/Perth/16/2009", "2009", "Australia", "WHO 2010-12"),
]

class EnhancedH3N2Collector:
    def __init__(self):
        self.sequences = {}  # hash -> {record, metadata}
        self.metadata_list = []
        self.duplicate_count = 0
        self.stats = defaultdict(int)
        
    def _hash_sequence(self, seq_str):
        """Generate MD5 hash for sequence deduplication"""
        return hashlib.md5(seq_str.upper().encode()).hexdigest()
    
    def load_existing_sequences(self):
        """Load existing FASTA to avoid duplicates"""
        if not os.path.exists(EXISTING_FASTA):
            print(f"[INFO] No existing file: {EXISTING_FASTA}")
            return 0
        
        print(f"\n{'='*60}")
        print("LOADING EXISTING SEQUENCES")
        print('='*60)
        
        count = 0
        for record in SeqIO.parse(EXISTING_FASTA, "fasta"):
            seq_hash = self._hash_sequence(str(record.seq))
            
            if seq_hash not in self.sequences:
                meta = self._parse_fasta_header(record)
                self.sequences[seq_hash] = {
                    'record': record,
                    'metadata': meta
                }
                self.metadata_list.append(meta)
                count += 1
        
        print(f"[EXISTING] Loaded {count} unique sequences")
        self.stats['existing'] = count
        return count

    def _parse_fasta_header(self, record):
        """Parse metadata from FASTA header"""
        desc = record.description
        acc = record.id
        
        meta = {
            'accession': acc,
            'description': desc,
            'sequence': str(record.seq),
            'length': len(record.seq),
            'source_database': 'Existing FASTA',
            'download_date': 'pre-existing',
            'ncbi_url': f"https://www.ncbi.nlm.nih.gov/protein/{acc}",
            'data_quality': 'unknown'
        }
        
        # Try to extract strain info: A/Location/ID/Year
        strain_patterns = [
            r'\(A/([^/]+)/([^/]+)/(\d{4})\)',  # (A/Location/ID/Year)
            r'A/([^/]+)/([^/]+)/(\d{4})',       # A/Location/ID/Year
            r'\[A/([^/]+)/([^/]+)/(\d{4})\]',   # [A/Location/ID/Year]
        ]
        
        for pattern in strain_patterns:
            match = re.search(pattern, desc)
            if match:
                meta['location'] = match.group(1)
                meta['isolate_id'] = match.group(2)
                meta['collection_year'] = int(match.group(3))
                meta['strain_name'] = f"A/{match.group(1)}/{match.group(2)}/{match.group(3)}"
                break

        return meta
    
    def search_ncbi(self, query, db="protein", max_results=5000):
        """Search NCBI database"""
        print(f"\n[SEARCH] {query[:80]}...")
        
        try:
            handle = Entrez.esearch(
                db=db,
                term=query,
                retmax=max_results,
                sort="relevance",
                usehistory="y"
            )
            results = Entrez.read(handle)
            handle.close()
            
            count = int(results["Count"])
            print(f"[FOUND] {count} results")
            return results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return None
    
    def download_batch(self, search_results, source_name, batch_size=100):
        """Download sequences in batches with full GenBank metadata"""
        if not search_results or int(search_results["Count"]) == 0:
            return 0
        
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        total = min(int(search_results["Count"]), 5000)
        
        print(f"[DOWNLOAD] {source_name}: {total} sequences")
        
        downloaded = 0
        new_count = 0
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            print(f"  Batch {start//batch_size + 1}/{(total-1)//batch_size + 1}: {start+1}-{end}")
            
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
                        self.duplicate_count += 1
                        continue
                    
                    meta = self._extract_genbank_metadata(record, source_name)
                    self.sequences[seq_hash] = {
                        'record': record,
                        'metadata': meta
                    }
                    self.metadata_list.append(meta)
                    new_count += 1
                    downloaded += 1
                
                handle.close()
                time.sleep(0.35)  # Rate limiting
                
            except Exception as e:
                print(f"  [ERROR] Batch failed: {e}")
                time.sleep(1)
                continue
        
        print(f"  [OK] Downloaded {downloaded}, New: {new_count}, Duplicates skipped: {self.duplicate_count}")
        self.stats[source_name] = new_count
        return new_count

    
    def _extract_genbank_metadata(self, record, source_name):
        """Extract comprehensive metadata from GenBank record"""
        desc = record.description
        annotations = record.annotations
        features = record.features
        
        meta = {
            'accession': record.id,
            'gi_number': annotations.get('gi', ''),
            'description': desc,
            'sequence': str(record.seq),
            'length': len(record.seq),
            'source_database': source_name,
            'download_date': datetime.now().strftime('%Y-%m-%d'),
            'ncbi_url': f"https://www.ncbi.nlm.nih.gov/protein/{record.id}",
            'organism': annotations.get('organism', ''),
            'taxonomy': ' > '.join(annotations.get('taxonomy', [])),
            'genbank_date': annotations.get('date', ''),
            'data_file_division': annotations.get('data_file_division', ''),
        }
        
        # Parse strain from description
        strain_match = re.search(r'Influenza A virus \(([^)]+)\)', desc)
        if strain_match:
            meta['strain_name'] = strain_match.group(1)
        else:
            meta['strain_name'] = ''
        
        # Parse year from strain name or collection_date
        year_patterns = [
            r'/(\d{4})\(H3N2\)',
            r'/(\d{4})\)',
            r'/(\d{4})$',
            r'(\d{4})\(H3N2\)',
        ]
        for pattern in year_patterns:
            match = re.search(pattern, desc)
            if match:
                meta['collection_year'] = int(match.group(1))
                break
        else:
            meta['collection_year'] = None
        
        # Parse location from strain
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
                break
        
        # Data quality flags
        meta['has_year'] = meta['collection_year'] is not None
        meta['has_location'] = meta['location'] not in ['Unknown', '']
        meta['has_host'] = meta.get('host', '') != ''
        meta['is_human'] = 'human' in meta.get('host', '').lower() or 'homo sapiens' in meta.get('host', '').lower()
        meta['is_h3n2'] = 'H3N2' in desc or 'H3N2' in meta.get('serotype', '')
        
        # Try to extract year from collection_date if not found
        if not meta['has_year'] and meta.get('collection_date'):
            date_str = meta['collection_date']
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                meta['collection_year'] = int(year_match.group(1))
                meta['has_year'] = True
        
        # Quality score (0-10)
        meta['quality_score'] = (
            (3 if meta['has_year'] else 0) +
            (2 if meta['has_location'] else 0) +
            (2 if meta['is_human'] else 0) +
            (2 if meta['is_h3n2'] else 0) +
            (1 if meta['has_host'] else 0)
        )
        
        return meta

    
    def download_vaccine_strains(self):
        """Download WHO vaccine reference strains"""
        print(f"\n{'='*60}")
        print("DOWNLOADING WHO VACCINE REFERENCE STRAINS")
        print('='*60)
        
        found = 0
        for strain, year, location, note in VACCINE_STRAINS:
            query = f'"{strain}"[All Fields] AND hemagglutinin[Protein Name]'
            
            try:
                handle = Entrez.esearch(db="protein", term=query, retmax=5)
                results = Entrez.read(handle)
                handle.close()
                
                if results["IdList"]:
                    handle = Entrez.efetch(
                        db="protein",
                        id=results["IdList"],
                        rettype="gb",
                        retmode="text"
                    )
                    
                    for record in SeqIO.parse(handle, "genbank"):
                        seq_hash = self._hash_sequence(str(record.seq))
                        
                        if seq_hash not in self.sequences:
                            meta = self._extract_genbank_metadata(record, f"WHO Vaccine: {note}")
                            meta['is_vaccine_strain'] = True
                            meta['vaccine_note'] = note
                            meta['collection_year'] = int(year)
                            meta['location'] = location
                            
                            self.sequences[seq_hash] = {
                                'record': record,
                                'metadata': meta
                            }
                            self.metadata_list.append(meta)
                            found += 1
                            print(f"  [OK] {strain} ({note})")
                    
                    handle.close()
                else:
                    print(f"  [--] {strain} not found")
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"  [ERROR] {strain}: {e}")
        
        print(f"\n[VACCINE] Found {found} vaccine reference sequences")
        self.stats['vaccine_strains'] = found
        return found

    
    def save_all_results(self):
        """Save all results with comprehensive labelling"""
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print('='*60)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metadata_list)
        
        # Sort by year (latest first), then by quality score
        df['collection_year'] = pd.to_numeric(df['collection_year'], errors='coerce')
        df = df.sort_values(
            ['collection_year', 'quality_score'], 
            ascending=[False, False],
            na_position='last'
        )
        df = df.reset_index(drop=True)
        
        # Add row index
        df.insert(0, 'index', range(1, len(df) + 1))
        
        # 1. Save comprehensive CSV
        csv_file = os.path.join(PROCESSED_DIR, 'h3n2_ha_comprehensive.csv')
        df.to_csv(csv_file, index=False)
        print(f"[SAVED] Metadata CSV: {csv_file}")
        
        # 2. Save merged FASTA (sorted by year)
        fasta_file = os.path.join(OUTPUT_DIR, 'h3n2_ha_all.fasta')
        records = []
        for _, row in df.iterrows():
            seq_hash = self._hash_sequence(row['sequence'])
            if seq_hash in self.sequences:
                rec = self.sequences[seq_hash]['record']
                records.append(rec)
        
        SeqIO.write(records, fasta_file, "fasta")
        print(f"[SAVED] FASTA: {fasta_file} ({len(records)} sequences)")
        
        # 3. Save high-quality subset (quality_score >= 7)
        hq_df = df[df['quality_score'] >= 7]
        if len(hq_df) > 0:
            hq_csv = os.path.join(PROCESSED_DIR, 'h3n2_ha_high_quality.csv')
            hq_df.to_csv(hq_csv, index=False)
            print(f"[SAVED] High-quality CSV: {hq_csv} ({len(hq_df)} sequences)")
        
        # 4. Save data provenance JSON
        provenance = self._generate_provenance(df)
        prov_file = os.path.join(PROCESSED_DIR, 'data_provenance.json')
        with open(prov_file, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        print(f"[SAVED] Provenance: {prov_file}")
        
        # 5. Update dashboard data
        self._update_dashboard(df, provenance)
        
        return df

    
    def _generate_provenance(self, df):
        """Generate data provenance documentation"""
        year_dist = df['collection_year'].dropna().astype(int).value_counts().sort_index()
        loc_dist = df['location'].value_counts()
        
        return {
            'project': 'PKM-RE H3N2 Antigenic Prediction',
            'team': 'Syifa (Bioteknologi) & Rofi (Teknik Komputer)',
            'institution': 'Universitas Brawijaya',
            'download_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sequences': len(df),
                'unique_sequences': len(self.sequences),
                'duplicates_removed': self.duplicate_count,
                'with_year_info': int(df['has_year'].sum()),
                'with_location_info': int(df['has_location'].sum()),
                'human_host_confirmed': int(df['is_human'].sum()),
                'high_quality_count': int((df['quality_score'] >= 7).sum()),
            },
            'year_range': {
                'min': int(df['collection_year'].min()) if df['collection_year'].notna().any() else None,
                'max': int(df['collection_year'].max()) if df['collection_year'].notna().any() else None,
            },
            'sources': {
                'breakdown': self.stats,
                'databases': {
                    'NCBI Protein': {
                        'url': 'https://www.ncbi.nlm.nih.gov/protein',
                        'description': 'NCBI Protein Database - primary source for protein sequences',
                        'access_method': 'Entrez API (Biopython)',
                    },
                    'NCBI Virus': {
                        'url': 'https://www.ncbi.nlm.nih.gov/labs/virus/vssi/',
                        'description': 'NCBI Virus portal - curated viral sequences',
                    },
                    'NCBI Influenza': {
                        'url': 'https://www.ncbi.nlm.nih.gov/genomes/FLU/',
                        'description': 'NCBI Influenza Virus Resource',
                    },
                    'GISAID': {
                        'url': 'https://www.gisaid.org/',
                        'description': 'Global Initiative on Sharing All Influenza Data (requires registration)',
                        'note': 'Not used in this download - requires separate access',
                    }
                }
            },
            'year_distribution': year_dist.to_dict(),
            'top_20_locations': loc_dist.head(20).to_dict(),
            'quality_distribution': df['quality_score'].value_counts().sort_index().to_dict(),
            'data_quality_criteria': {
                'has_year': '+3 points',
                'has_location': '+2 points',
                'is_human_host': '+2 points',
                'is_h3n2_confirmed': '+2 points',
                'has_host_info': '+1 point',
                'max_score': 10,
                'high_quality_threshold': 7,
            },
            'files_generated': [
                'data/raw/h3n2_ha_all.fasta',
                'data/processed/h3n2_ha_comprehensive.csv',
                'data/processed/h3n2_ha_high_quality.csv',
                'data/processed/data_provenance.json',
                'dashboard/data.json',
            ]
        }

    
    def _update_dashboard(self, df, provenance):
        """Update dashboard JSON with latest data"""
        os.makedirs('dashboard', exist_ok=True)
        
        # Prepare year distribution for chart
        year_dist = df['collection_year'].dropna().astype(int).value_counts().sort_index()
        
        # Get sample data (top 10 by year)
        sample_df = df.head(10)
        samples = []
        for _, row in sample_df.iterrows():
            samples.append({
                'acc': row['accession'],
                'strain': row.get('strain_name', 'Unknown'),
                'year': int(row['collection_year']) if pd.notna(row['collection_year']) else 0,
                'loc': row.get('location', 'Unknown'),
                'len': row['length'],
                'quality': row.get('quality_score', 0),
                'source': row.get('source_database', 'Unknown'),
            })
        
        # Build data sources list
        data_sources = []
        for source, count in self.stats.items():
            if count > 0:
                data_sources.append({
                    'database': source,
                    'url': 'https://www.ncbi.nlm.nih.gov/protein',
                    'count': count,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                })
        
        dashboard_data = {
            'lastUpdate': datetime.now().isoformat(),
            'projectInfo': {
                'title': 'Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2',
                'team': 'Syifa (Ketua/Bioteknologi) & Rofi (Anggota/Teknik Komputer)',
                'institution': 'Universitas Brawijaya',
                'scheme': 'PKM-RE 2026',
            },
            'dataSources': data_sources,
            'stats': {
                'totalSeq': len(df),
                'uniqueSeq': len(self.sequences),
                'withYear': int(df['has_year'].sum()),
                'withLocation': int(df['has_location'].sum()),
                'humanHost': int(df['is_human'].sum()),
                'highQuality': int((df['quality_score'] >= 7).sum()),
                'yearRange': f"{int(year_dist.index.min())}-{int(year_dist.index.max())}" if len(year_dist) > 0 else "N/A",
            },
            'pipeline': [
                {'step': 'Data Collection', 'status': 'done', 'desc': f'Downloaded {len(df)} sequences from NCBI'},
                {'step': 'Data Preprocessing', 'status': 'done', 'desc': f'Deduplicated, labeled, quality scored'},
                {'step': 'Feature Extraction', 'status': 'pending', 'desc': 'Physicochemical feature extraction'},
                {'step': 'Model Training', 'status': 'pending', 'desc': 'XGBoost classifier training'},
                {'step': 'Evaluation', 'status': 'pending', 'desc': 'Cross-validation & metrics'},
                {'step': 'Interpretation', 'status': 'pending', 'desc': 'Feature importance analysis'},
            ],
            'sampleData': samples,
            'yearDistribution': {
                'labels': [str(y) for y in year_dist.index.tolist()],
                'data': year_dist.values.tolist(),
            },
            'qualityDistribution': df['quality_score'].value_counts().sort_index().to_dict(),
            'locationDistribution': df['location'].value_counts().head(15).to_dict(),
        }
        
        with open('dashboard/data.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"[SAVED] Dashboard: dashboard/data.json")

    
    def print_summary(self, df):
        """Print comprehensive summary"""
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print('='*60)
        
        print(f"\n📊 TOTAL STATISTICS:")
        print(f"   Total unique sequences: {len(df)}")
        print(f"   Duplicates removed: {self.duplicate_count}")
        print(f"   With year info: {df['has_year'].sum()} ({df['has_year'].mean()*100:.1f}%)")
        print(f"   With location info: {df['has_location'].sum()} ({df['has_location'].mean()*100:.1f}%)")
        print(f"   Human host confirmed: {df['is_human'].sum()} ({df['is_human'].mean()*100:.1f}%)")
        print(f"   High quality (score>=7): {(df['quality_score']>=7).sum()}")
        
        if df['collection_year'].notna().any():
            print(f"\n📅 YEAR RANGE: {int(df['collection_year'].min())} - {int(df['collection_year'].max())}")
            print("\n   Top 10 years:")
            year_counts = df['collection_year'].value_counts().sort_index(ascending=False).head(10)
            for year, count in year_counts.items():
                print(f"   {int(year)}: {count} sequences")
        
        print(f"\n🌍 TOP 10 LOCATIONS:")
        loc_counts = df['location'].value_counts().head(10)
        for loc, count in loc_counts.items():
            print(f"   {loc}: {count}")
        
        print(f"\n📁 SOURCES:")
        for source, count in self.stats.items():
            print(f"   {source}: {count}")
        
        print(f"\n✅ FILES SAVED:")
        print(f"   - data/raw/h3n2_ha_all.fasta")
        print(f"   - data/processed/h3n2_ha_comprehensive.csv")
        print(f"   - data/processed/h3n2_ha_high_quality.csv")
        print(f"   - data/processed/data_provenance.json")
        print(f"   - dashboard/data.json")
        
        print(f"\n{'='*60}")
        print("DONE!")
        print('='*60)


def main():
    print("="*60)
    print("H3N2 HA COMPREHENSIVE DATA COLLECTION v2.0")
    print("PKM-RE: Prediksi Antigenic Drift dengan Machine Learning")
    print("="*60)
    
    collector = EnhancedH3N2Collector()
    
    # 1. Load existing sequences first (to avoid duplicates)
    collector.load_existing_sequences()
    
    # 2. Download recent sequences (2023-2026) - highest priority
    print(f"\n{'='*60}")
    print("DOWNLOADING RECENT SEQUENCES (2023-2026)")
    print('='*60)
    
    query_2023_2026 = (
        '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
        '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
        '(2023:2026[Publication Date]) AND (500:600[Sequence Length])'
    )
    results = collector.search_ncbi(query_2023_2026, max_results=3000)
    if results:
        collector.download_batch(results, "NCBI 2023-2026")
    
    # 3. Download 2020-2022 sequences
    print(f"\n{'='*60}")
    print("DOWNLOADING 2020-2022 SEQUENCES")
    print('='*60)
    
    query_2020_2022 = (
        '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
        '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
        '(2020:2022[Publication Date]) AND (500:600[Sequence Length])'
    )
    results = collector.search_ncbi(query_2020_2022, max_results=2000)
    if results:
        collector.download_batch(results, "NCBI 2020-2022")
    
    # 4. Download historical sequences (2015-2019)
    print(f"\n{'='*60}")
    print("DOWNLOADING HISTORICAL SEQUENCES (2015-2019)")
    print('='*60)
    
    query_2015_2019 = (
        '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
        '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
        '(2015:2019[Publication Date]) AND (500:600[Sequence Length])'
    )
    results = collector.search_ncbi(query_2015_2019, max_results=2000)
    if results:
        collector.download_batch(results, "NCBI 2015-2019")
    
    # 5. Download WHO vaccine reference strains
    collector.download_vaccine_strains()
    
    # 6. Save all results
    df = collector.save_all_results()
    
    # 7. Print summary
    collector.print_summary(df)


if __name__ == "__main__":
    main()
