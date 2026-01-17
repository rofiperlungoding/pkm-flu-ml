# PKM-RE: Prediksi Antigenic Drift H3N2 dengan Machine Learning

Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2 Melalui Integrasi Machine Learning Berbasis Sifat Fisikokimia Protein Hemaglutinin dan Dinamika Evolusi Epistatik untuk Optimalis
asi Strain Vaksin

## Tim
- **Ketua**: Syifa (Bioteknologi)
- **Anggota**: Rofi (Teknik Komputer)

## Struktur Project
```
├── data/
│   ├── raw/           # Data FASTA mentah
│   └── processed/     # Data setelah preprocessing
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── utils.py
├── notebooks/         # Jupyter notebooks untuk eksplorasi
├── models/            # Trained models
├── results/           # Output visualisasi & hasil
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Workflow
1. Download data dari NCBI/GISAID
2. Preprocessing & alignment
3. Ekstraksi fitur fisikokimia
4. Training model ML
5. Evaluasi & interpretasi