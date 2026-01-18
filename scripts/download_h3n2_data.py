"""
Script untuk download sekuens protein HA H3N2 dari NCBI
Jalankan: python scripts/download_h3n2_data.py
"""
from Bio import Entrez, SeqIO
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

Entrez.email = os.getenv('NCBI_EMAIL')
Entrez.api_key = os.getenv('NCBI_API_KEY')

# Validate environment variables
if not Entrez.email:
    raise ValueError(
        "Missing NCBI email! Please set NCBI_EMAIL in .env file.\n"
        "Copy .env.example to .env and fill in your credentials."
    )

def search_h3n2_ha(max_results=2000):
    print("Searching NCBI for H3N2 HA sequences...")
    query = ('Influenza A virus[Organism] AND H3N2[All Fields] AND '
             'hemagglutinin[Protein Name] AND human[Host] AND '
             '500:600[Sequence Length]')
    handle = Entrez.esearch(db="protein", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"Found {len(ids)} sequences")
    return ids

def download_sequences(ids, output_file, batch_size=100):
    print(f"Downloading {len(ids)} sequences...")
    all_records = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} seqs...")
        try:
            handle = Entrez.efetch(db="protein", id=batch, rettype="fasta", retmode="text")
            records = list(SeqIO.parse(handle, "fasta"))
            all_records.extend(records)
            handle.close()
            time.sleep(0.4)
        except Exception as e:
            print(f"  Error: {e}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    SeqIO.write(all_records, output_file, "fasta")
    print(f"Saved {len(all_records)} sequences")
    return len(all_records)

if __name__ == "__main__":
    ids = search_h3n2_ha(max_results=1500)
    if ids:
        download_sequences(ids, "data/raw/h3n2_ha_sequences.fasta")
        print("Done!")
    else:
        print("No sequences found!")