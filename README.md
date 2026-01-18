# PKM-RE: Prediksi Antigenic Drift H3N2 dengan Machine Learning

🧬 **Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2 Melalui Integrasi Machine Learning Berbasis Sifat Fisikokimia Protein Hemaglutinin**

## 📋 Informasi Project

| Item | Detail |
|------|--------|
| **Skema** | PKM-RE (Riset Eksakta) 2026 |
| **Institusi** | Universitas Brawijaya |
| **Tim** | Syifa (Ketua/Bioteknologi) & Rofi (Anggota/Teknik Komputer) |
| **Email** | syifazavira@student.ub.ac.id |

## 🎯 Tujuan Penelitian

Mengembangkan model machine learning berbasis XGBoost untuk memprediksi perubahan antigenik virus Influenza A H3N2 menggunakan fitur fisikokimia protein Hemaglutinin (HA).

## 📊 Dataset

- **Total Sequences**: 2,818 unique H3N2 HA sequences
- **Year Range**: 1996-2024
- **High Quality**: 2,204 sequences (quality score ≥7)
- **Human Host**: 2,184 sequences (98.2%)
- **Sources**: NCBI Protein Database, WHO Vaccine Reference Strains

## 🔬 Metodologi

### 1. Data Collection
- Download dari NCBI Protein Database via Entrez API
- Filtering: length 500-600 aa, human host, H3N2 subtype
- Deduplication menggunakan MD5 hash

### 2. Feature Extraction (74 features)
- **Amino Acid Composition** (20 features): Frekuensi setiap asam amino
- **Physicochemical Properties** (38 features): Hydrophobicity, volume, polarity, charge, molecular weight, isoelectric point
- **Epitope Site Features** (15 features): Mutasi dan properti di 5 epitope sites (A-E)
- **Sequence Features** (9 features): Length, net charge, hydrophobic ratio, dll

### 3. Model Training
- **Algorithm**: XGBoost Classifier
- **Cross-validation**: 5-fold stratified CV
- **Train/Test Split**: 80/20

## 📈 Hasil Model

### Multi-class Model (4 Temporal Periods)
| Metric | Score |
|--------|-------|
| CV Accuracy | 87.64% ± 8.34% |
| Test Accuracy | **93.48%** |
| F1-Score | 93.48% |
| Precision | 93.49% |

### Binary Model (Recent vs Historical)
| Metric | Score |
|--------|-------|
| CV Accuracy | 99.42% ± 0.84% |
| Test Accuracy | **99.55%** |
| F1-Score | 99.55% |
| ROC-AUC | **1.00** |

### Top 5 Important Features
1. `aa_Y` - Tyrosine composition (26.1%)
2. `polar_ratio` - Polar amino acid ratio (9.8%)
3. `aa_Q` - Glutamine composition (7.6%)
4. `epitope_A_charge_sum` - Epitope A charge (4.9%)
5. `epitope_E_hydro_mean` - Epitope E hydrophobicity (4.8%)


## 📁 Struktur Project

```
pkm-flu-ml/
├── dashboard/
│   ├── index.html          # Interactive dashboard
│   └── data.json           # Dashboard data
├── data/
│   ├── raw/
│   │   ├── h3n2_ha_sequences.fasta
│   │   └── h3n2_ha_all.fasta
│   └── processed/
│       ├── h3n2_ha_comprehensive.csv
│       ├── h3n2_ha_high_quality.csv
│       ├── h3n2_features.csv
│       ├── h3n2_features_matrix.csv
│       ├── data_provenance.json
│       └── feature_extraction_info.json
├── models/
│   ├── h3n2_multiclass_model.pkl
│   └── h3n2_binary_model.pkl
├── results/
│   ├── training_results.json
│   ├── year_distribution.png
│   ├── multiclass_confusion_matrix.png
│   ├── multiclass_feature_importance.png
│   ├── binary_confusion_matrix.png
│   └── binary_feature_importance.png
├── scripts/
│   ├── download_comprehensive_h3n2.py
│   ├── extract_features.py
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── physicochemical.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   └── model.py
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/rofiperlungoding/pkm-flu-ml.git
cd pkm-flu-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (optional - data sudah tersedia)
python scripts/download_comprehensive_h3n2.py

# 4. Extract features
python scripts/extract_features.py

# 5. Train model
python scripts/train_model.py

# 6. View dashboard
# Buka dashboard/index.html di browser
```

## 📚 Referensi

1. Li X, et al. (2024). A sequence-based machine learning model for predicting antigenic distance for H3N2 influenza virus. *Front Microbiol.* [DOI](https://doi.org/10.3389/fmicb.2024.1345794)

2. Allen JD, Ross TM. (2024). mRNA vaccines encoding computationally optimized hemagglutinin. *Front. Immunol.* [DOI](https://doi.org/10.3389/fimmu.2024.1334670)

3. NCBI Influenza Virus Resource: https://www.ncbi.nlm.nih.gov/genomes/FLU/

## 📄 License

This project is for academic purposes (PKM-RE 2026).

---
*PKM-RE 2026 | Universitas Brawijaya*
