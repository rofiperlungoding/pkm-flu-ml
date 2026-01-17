"""
Script untuk update dashboard dengan data terbaru
Jalankan setelah setiap tahap pipeline selesai
"""
import json
import os
from datetime import datetime
from Bio import SeqIO
import re

def get_sequence_stats(fasta_file):
    """Get statistics from FASTA file"""
    records = list(SeqIO.parse(fasta_file, 'fasta'))
    
    years = []
    samples = []
    
    for rec in records[:100]:  # Sample first 100
        desc = rec.description
        # Extract year
        match = re.search(r'/([12][09]\d{2})\)', desc)
        if match:
            years.append(int(match.group(1)))
        
        # Extract location
        loc_match = re.search(r'A/([^/]+)/', desc)
        loc = loc_match.group(1) if loc_match else 'Unknown'
        
        samples.append({
            'acc': rec.id,
            'strain': f"A/{loc}/{match.group(1) if match else '?'}",
            'year': int(match.group(1)) if match else 0,
            'loc': loc,
            'len': len(rec.seq)
        })
    
    # Sort by year descending
    samples.sort(key=lambda x: x['year'], reverse=True)
    
    # Year distribution
    year_counts = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    
    return {
        'total': len(records),
        'samples': samples[:10],
        'year_dist': year_counts,
        'year_range': f"{min(years)}-{max(years)}" if years else "N/A"
    }

def generate_dashboard_data(stats, pipeline_status):
    """Generate dashboard JSON data"""
    # Sort years for chart
    sorted_years = sorted(stats['year_dist'].keys())
    
    return {
        'lastUpdate': datetime.now().isoformat(),
        'dataSources': [{
            'database': 'NCBI Protein Database',
            'url': 'https://www.ncbi.nlm.nih.gov/protein',
            'query': 'Influenza A H3N2 HA human 500:600[len]',
            'count': stats['total'],
            'date': '2026-01-18'
        }],
        'stats': {
            'totalSeq': stats['total'],
            'yearRange': stats['year_range']
        },
        'pipeline': pipeline_status,
        'sampleData': stats['samples'],
        'yearDistribution': {
            'labels': [str(y) for y in sorted_years],
            'data': [stats['year_dist'][y] for y in sorted_years]
        }
    }

def main():
    fasta_file = 'data/raw/h3n2_ha_sequences.fasta'
    
    if not os.path.exists(fasta_file):
        print(f"Error: {fasta_file} not found")
        return
    
    print("Analyzing sequences...")
    stats = get_sequence_stats(fasta_file)
    
    # Current pipeline status
    pipeline = [
        {'step': 'Data Collection', 'status': 'done', 'desc': f'Downloaded {stats["total"]} sequences from NCBI'},
        {'step': 'Data Preprocessing', 'status': 'progress', 'desc': 'Parsing, filtering, quality control'},
        {'step': 'Feature Extraction', 'status': 'pending', 'desc': 'Physicochemical feature extraction'},
        {'step': 'Model Training', 'status': 'pending', 'desc': 'XGBoost classifier training'},
        {'step': 'Evaluation', 'status': 'pending', 'desc': 'Cross-validation & metrics'},
        {'step': 'Interpretation', 'status': 'pending', 'desc': 'Feature importance analysis'}
    ]
    
    data = generate_dashboard_data(stats, pipeline)
    
    # Save to JSON
    os.makedirs('dashboard', exist_ok=True)
    with open('dashboard/data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Dashboard data updated!")
    print(f"  Total sequences: {stats['total']}")
    print(f"  Year range: {stats['year_range']}")
    print(f"  Data saved to: dashboard/data.json")

if __name__ == '__main__':
    main()