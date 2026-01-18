import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

df = pd.read_csv('data/processed/h3n2_ha_comprehensive.csv')
test_df = df.sample(20, random_state=42)
records = [SeqRecord(Seq(row['sequence']), id=row['accession'], description=f"{row['strain_name']} {row['collection_year']}") for _, row in test_df.iterrows()]
SeqIO.write(records, 'test_batch.fasta', 'fasta')
print(f'Created test_batch.fasta with {len(records)} sequences')
