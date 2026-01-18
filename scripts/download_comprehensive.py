"""
Comprehensive H3N2 HA Data Downloader
- Download dari multiple sources
- Merge dengan data existing
- Remove duplicates
- Complete metadata labelling

Sources:
1. NCBI Protein Database (Entrez API)
2. NCBI Virus (newer, more curated)
"""
from Bio import Entrez, SeqIO
import pandas as pd
import time
import os
import re
import json
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
Entrez.email = os.getenv('NCBI_EMAIL')
Entrez.api_key = os.getenv('NCBI_API_KEY')

# Validate environment variables
if not Entrez.email or not Entrez.api_key:
    raise ValueError(
        "Missing NCBI credentials! Please set NCBI_EMAIL and NCBI_API_KEY in .env file.\n"
        "Copy .env.example to .env and fill in your credentials."
    )

OUTPUT_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

class H3N2DataCollector:
    def __init__(self):
        self.all_sequences = {}  # key: sequence hash, value: record
        self.metadata = []
        
    def search_ncbi_protein(self, query, max_results=5000):
        """Search NCBI Protein database"""
        print(f"\n[NCBI Protein] Searching: {query[:50]}...")
        
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
        print(f"[NCBI Protein] Found {count} sequences")
        
        return results
    
    def download_sequences(self, search_results, source_name, batch_size=200):
        """Download sequences in batches with full metadata"""
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        count = min(int(search_results["Count"]), 5000)
        
        print(f"[{source_name}] Downloading {count} sequences...")
        
        downloaded = 0
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            print(f"  Batch {start//batch_size + 1}: {start+1}-{end}...")
            
            try:
                # Download GenBank format for full metadata
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
                    self._process_record(record, source_name)
                    downloaded += 1
                
                handle.close()
                time.sleep(0.4)  # Rate limiting
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"[{source_name}] Downloaded {downloaded} sequences")
        return downloaded

    
    def _process_record(self, record, source_name):
        """Process a single GenBank record and extract metadata"""
        seq_str = str(record.seq)
        seq_hash = hashlib.md5(seq_str.encode()).hexdigest()
        
        # Skip if duplicate sequence
        if seq_hash in self.all_sequences:
            return
        
        # Extract metadata from GenBank record
        meta = self._extract_metadata(record, source_name)
        
        # Store
        self.all_sequences[seq_hash] = {
            'record': record,
            'metadata': meta
        }
        self.metadata.append(meta)
    
    def _extract_metadata(self, record, source_name):
        """Extract comprehensive metadata from GenBank record"""
        desc = record.description
        annotations = record.annotations
        features = record.features
        
        # Basic info
        meta = {
            'accession': record.id,
            'gi': record.annotations.get('gi', ''),
            'description': desc,
            'sequence': str(record.seq),
            'length': len(record.seq),
            'source_database': source_name,
            'download_date': datetime.now().strftime('%Y-%m-%d'),
        }
        
        # Parse strain name: A/Location/ID/Year(H3N2)
        strain_match = re.search(r'\(A/([^)]+)\)', desc)
        if strain_match:
            meta['strain_full'] = f"A/{strain_match.group(1)}"
        else:
            strain_match2 = re.search(r'Influenza A virus \(([^)]+)\)', desc)
            meta['strain_full'] = strain_match2.group(1) if strain_match2 else ''
        
        # Parse year
        year_match = re.search(r'/(\d{4})\(H3N2\)', desc)
        if year_match:
            meta['collection_year'] = int(year_match.group(1))
        else:
            year_match2 = re.search(r'/(\d{4})\)', desc)
            if year_match2:
                meta['collection_year'] = int(year_match2.group(1))
            else:
                meta['collection_year'] = None
        
        # Parse location
        loc_match = re.search(r'A/([^/]+)/', meta.get('strain_full', ''))
        meta['location'] = loc_match.group(1) if loc_match else 'Unknown'
        
        # From annotations
        meta['organism'] = annotations.get('organism', '')
        meta['taxonomy'] = ' > '.join(annotations.get('taxonomy', []))
        meta['date'] = annotations.get('date', '')
        meta['data_file_division'] = annotations.get('data_file_division', '')
        
        # From source feature
        for feat in features:
            if feat.type == 'source':
                qualifiers = feat.qualifiers
                meta['host'] = qualifiers.get('host', [''])[0]
                meta['country'] = qualifiers.get('country', [''])[0]
                meta['collection_date'] = qualifiers.get('collection_date', [''])[0]
                meta['isolate'] = qualifiers.get('isolate', [''])[0]
                meta['strain'] = qualifiers.get('strain', [''])[0]
                meta['serotype'] = qualifiers.get('serotype', [''])[0]
                meta['segment'] = qualifiers.get('segment', [''])[0]
                meta['db_xref'] = '; '.join(qualifiers.get('db_xref', []))
                break
        
        # Generate NCBI link
        meta['ncbi_url'] = f"https://www.ncbi.nlm.nih.gov/protein/{record.id}"
        
        # Determine data quality
        meta['has_year'] = meta['collection_year'] is not None
        meta['has_location'] = meta['location'] != 'Unknown' and meta['location'] != ''
        meta['has_host'] = meta.get('host', '') != ''
        meta['is_human'] = 'human' in meta.get('host', '').lower() or 'homo' in meta.get('host', '').lower()
        
        return meta

    
    def load_existing_fasta(self, fasta_file):
        """Load existing FASTA file and add to collection"""
        if not os.path.exists(fasta_file):
            print(f"[Existing] File not found: {fasta_file}")
            return 0
        
        print(f"\n[Existing] Loading {fasta_file}...")
        count = 0
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq)
            seq_hash = hashlib.md5(seq_str.encode()).hexdigest()
            
            if seq_hash not in self.all_sequences:
                # Create basic metadata from FASTA header
                meta = {
                    'accession': record.id,
                    'description': record.description,
                    'sequence': seq_str,
                    'length': len(record.seq),
                    'source_database': 'Existing FASTA',
                    'download_date': 'pre-existing',
                    'ncbi_url': f"https://www.ncbi.nlm.nih.gov/protein/{record.id}"
                }
                
                # Try to parse year and location from description
                year_match = re.search(r'/(\d{4})\)', record.description)
                meta['collection_year'] = int(year_match.group(1)) if year_match else None
                
                loc_match = re.search(r'A/([^/]+)/', record.description)
                meta['location'] = loc_match.group(1) if loc_match else 'Unknown'
                
                self.all_sequences[seq_hash] = {
                    'record': record,
                    'metadata': meta
                }
                self.metadata.append(meta)
                count += 1
        
        print(f"[Existing] Loaded {count} unique sequences")
        return count
    
    def save_results(self):
        """Save all results with comprehensive labelling"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metadata)
        
        # Sort by year (latest first)
        df['collection_year'] = pd.to_numeric(df['collection_year'], errors='coerce')
        df = df.sort_values('collection_year', ascending=False, na_position='last')
        df = df.reset_index(drop=True)
        
        # Add quality score
        df['quality_score'] = (
            df['has_year'].astype(int) * 3 +
            df['has_location'].astype(int) * 2 +
            df['has_host'].astype(int) * 1 +
            df['is_human'].astype(int) * 2
        )
        
        # Save comprehensive CSV
        csv_file = os.path.join(PROCESSED_DIR, 'h3n2_ha_comprehensive.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nSaved metadata to {csv_file}")
        
        # Save FASTA (sorted by year)
        fasta_file = os.path.join(OUTPUT_DIR, 'h3n2_ha_all.fasta')
        records = []
        for _, row in df.iterrows():
            # Find the record
            seq_hash = hashlib.md5(row['sequence'].encode()).hexdigest()
            if seq_hash in self.all_sequences:
                rec = self.all_sequences[seq_hash]['record']
                records.append(rec)
        
        SeqIO.write(records, fasta_file, "fasta")
        print(f"Saved {len(records)} sequences to {fasta_file}")
        
        # Save data provenance JSON
        provenance = {
            'download_date': datetime.now().isoformat(),
            'total_sequences': len(df),
            'unique_sequences': len(self.all_sequences),
            'sources': df['source_database'].value_counts().to_dict(),
            'year_range': {
                'min': int(df['collection_year'].min()) if df['collection_year'].notna().any() else None,
                'max': int(df['collection_year'].max()) if df['collection_year'].notna().any() else None
            },
            'with_year': int(df['has_year'].sum()),
            'with_location': int(df['has_location'].sum()),
            'human_host': int(df['is_human'].sum()),
            'year_distribution': df['collection_year'].value_counts().sort_index().to_dict(),
            'top_locations': df['location'].value_counts().head(20).to_dict(),
            'database_urls': {
                'NCBI Protein': 'https://www.ncbi.nlm.nih.gov/protein',
                'NCBI Virus': 'https://www.ncbi.nlm.nih.gov/labs/virus/vssi/',
                'NCBI Influenza': 'https://www.ncbi.nlm.nih.gov/genomes/FLU/'
            }
        }
        
        prov_file = os.path.join(PROCESSED_DIR, 'data_provenance.json')
        with open(prov_file, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        print(f"Saved provenance to {prov_file}")
        
        return df



def main():
    print("="*60)
    print("H3N2 HA Comprehensive Data Collection")
    print("="*60)
    
    collector = H3N2DataCollector()
    
    # 1. Load existing data first
    existing_file = "data/raw/h3n2_ha_sequences.fasta"
    collector.load_existing_fasta(existing_file)
    
    # 2. Download from NCBI Protein - Recent years (2020-2026)
    query_recent = (
        '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
        '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
        '(2020:2026[Publication Date]) AND (500:600[Sequence Length])'
    )
    results_recent = collector.search_ncbi_protein(query_recent, max_results=3000)
    collector.download_sequences(results_recent, "NCBI Protein (2020-2026)")
    
    # 3. Download from NCBI Protein - Historical (2015-2019)
    query_historical = (
        '(Influenza A virus[Organism]) AND (H3N2[All Fields]) AND '
        '(hemagglutinin[Protein Name]) AND (human[Host]) AND '
        '(2015:2019[Publication Date]) AND (500:600[Sequence Length])'
    )
    results_hist = collector.search_ncbi_protein(query_historical, max_results=2000)
    collector.download_sequences(results_hist, "NCBI Protein (2015-2019)")
    
    # 4. Download vaccine reference strains
    vaccine_strains = [
        "A/Darwin/6/2021", "A/Darwin/9/2021",
        "A/Hong Kong/45/2019", "A/Kansas/14/2017",
        "A/Singapore/INFIMH-16-0019/2016", "A/Hong Kong/4801/2014",
        "A/Switzerland/9715293/2013", "A/Texas/50/2012",
        "A/Victoria/361/2011", "A/Perth/16/2009"
    ]
    
    print("\n[Vaccine Strains] Searching for reference strains...")
    for strain in vaccine_strains:
        query = f'"{strain}"[All Fields] AND hemagglutinin[Protein Name]'
        try:
            handle = Entrez.esearch(db="protein", term=query, retmax=10)
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
                    collector._process_record(record, f"Vaccine Reference: {strain}")
                handle.close()
                print(f"  Found: {strain}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error for {strain}: {e}")
    
    # 5. Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    df = collector.save_results()
    
    # 6. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total unique sequences: {len(df)}")
    print(f"With collection year: {df['has_year'].sum()}")
    print(f"With location: {df['has_location'].sum()}")
    print(f"Human host confirmed: {df['is_human'].sum()}")
    
    if df['collection_year'].notna().any():
        print(f"\nYear range: {int(df['collection_year'].min())} - {int(df['collection_year'].max())}")
        print("\nTop 10 years:")
        print(df['collection_year'].value_counts().sort_index(ascending=False).head(10))
    
    print("\nTop 10 locations:")
    print(df['location'].value_counts().head(10))
    
    print("\nSources:")
    print(df['source_database'].value_counts())
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()