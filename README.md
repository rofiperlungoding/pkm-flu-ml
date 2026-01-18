# PKM-RE: Prediksi Antigenic Drift H3N2 dengan Machine Learning

🧬 **Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2 Melalui Integrasi Machine Learning Berbasis Sifat Fisikokimia Protein Hemaglutinin**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📋 Informasi Project

| Item | Detail |
|------|--------|
| **Skema** | PKM-RE (Riset Eksakta) 2026 |
| **Institusi** | Universitas Brawijaya |
| **Tim** | Syifa Zavira Ramadhani (Ketua/Bioteknologi) & Rofi Perdana (Anggota/Teknik Komputer) |
| **Email** | syifazavira@student.ub.ac.id, mrofid@student.ub.ac.id |
| **Repository** | https://github.com/rofiperlungoding/pkm-flu-ml |

---

## 🎯 Tujuan Penelitian

Mengembangkan **pipeline machine learning end-to-end** untuk memprediksi perubahan antigenik virus Influenza A H3N2 menggunakan fitur fisikokimia protein Hemaglutinin (HA), dengan tujuan:

✅ Prediksi periode temporal strain H3N2 dengan akurasi tinggi  
✅ Identifikasi fitur fisikokimia yang berkorelasi dengan antigenic drift  
✅ Analisis epitope site mutations sebagai prediktor perubahan antigenik  
✅ Menyediakan tools prediksi untuk surveillance epidemiologi

---

## 📊 Dataset

| Statistik | Nilai |
|-----------|-------|
| **Total Sequences** | 2,818 unique H3N2 HA sequences |
| **Year Range** | 1996-2024 (29 years) |
| **High Quality** | 2,204 sequences (quality score ≥7) |
| **Human Host** | 2,184 sequences (98.2%) |
| **Recent Sequences** | 1,455 sequences (≥2020) |
| **Sources** | NCBI Protein Database, WHO Vaccine Reference Strains |

### Data Distribution by Period
- **<2010 (Historical):** 234 sequences
- **2010-2014 (Mid-Historical):** 459 sequences
- **2015-2019 (Mid-Recent):** 670 sequences
- **≥2020 (Recent):** 1,455 sequences

---

## 🔬 Metodologi

### 1. Data Collection
- **Source:** NCBI Protein Database via Entrez API
- **Filtering:** Length 500-600 aa, human host, H3N2 subtype
- **Deduplication:** MD5 hash-based
- **Quality Scoring:** 0-10 scale based on metadata completeness
- **WHO Reference Strains:** 15 vaccine strains (2010-2025)

### 2. Feature Extraction (74 features)

#### Amino Acid Composition (20 features)
Frekuensi relatif setiap asam amino (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)

#### Physicochemical Properties (30+ features)
- **Hydrophobicity:** Mean, variance, hydrophobic/hydrophilic fractions
- **Charge:** Mean charge, positive/negative fractions, net charge, pI
- **Polarity:** Polar/non-polar fractions, polarity ratio
- **Aromaticity:** Aromatic residue fraction
- **Molecular:** Weight, instability index, aliphatic index

#### Epitope Site Analysis (24 features)
Berdasarkan 5 epitope sites H3N2 (Koel et al., 2013):
- **Site A, B, C, D, E:** Mutations, mutation rates, hydrophobicity/charge changes
- **Total epitope mutations:** Sum across all sites
- **Reference strain:** A/Perth/16/2009

### 3. Model Training
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Models:** Binary (Recent vs Historical) & Multi-class (4 periods)
- **Cross-validation:** 5-fold stratified CV
- **Train/Test Split:** 80:20 with stratification
- **Hyperparameters:** n_estimators=200, max_depth=6, learning_rate=0.1

### 4. Model Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Validation:** Cross-validation, ROC curves, PR curves, learning curves
- **Interpretation:** Feature importance, correlation analysis

---

## 📈 Hasil Model

### Binary Model (Recent ≥2020 vs Historical <2020)

| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| **Accuracy** | 99.47% ± 0.31% | **99.55%** |
| **Precision** | 99.48% ± 0.33% | 99.58% |
| **Recall** | 99.58% ± 0.29% | 99.65% |
| **F1-Score** | 99.53% ± 0.30% | 99.62% |
| **ROC-AUC** | - | **0.9998** |

**Confusion Matrix (Test Set):**
```
                Predicted
              Historical  Recent
Actual
Historical       272        1
Recent             1       290
```

### Multi-class Model (4 Temporal Periods)

| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| **Accuracy** | 93.12% ± 1.24% | **93.48%** |
| **Precision** | 93.15% ± 1.28% | 93.52% |
| **Recall** | 93.12% ± 1.24% | 93.48% |
| **F1-Score** | 93.11% ± 1.26% | 93.48% |

**Per-class Performance (Test Set):**
- **<2010:** Precision 95.65%, Recall 93.62%
- **2010-2014:** Precision 91.30%, Recall 91.30%
- **2015-2019:** Precision 92.54%, Recall 92.54%
- **≥2020:** Precision 94.59%, Recall 96.53%

### Top 10 Most Important Features (Binary Model)

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | `epitope_site_mutations` | 0.0847 | Total mutations across epitope sites |
| 2 | `site_B_mutations` | 0.0623 | Mutations in epitope site B |
| 3 | `site_D_mutations` | 0.0521 | Mutations in epitope site D |
| 4 | `mean_hydrophobicity` | 0.0489 | Average hydrophobicity |
| 5 | `site_C_mutations` | 0.0445 | Mutations in epitope site C |
| 6 | `aa_K` | 0.0398 | Lysine composition |
| 7 | `site_E_mutations` | 0.0387 | Mutations in epitope site E |
| 8 | `aa_N` | 0.0356 | Asparagine composition |
| 9 | `positive_charge_fraction` | 0.0334 | Fraction of positively charged residues |
| 10 | `site_A_mutations` | 0.0312 | Mutations in epitope site A |

**Key Finding:** Epitope site mutations adalah prediktor terkuat, mengkonfirmasi peran antigenic drift dalam evolusi H3N2.

---

## 📁 Struktur Project

```
pkm-flu-ml/
├── dashboard/
│   ├── index.html                      # Interactive dashboard
│   └── data.json                       # Dashboard data
├── data/
│   ├── raw/
│   │   ├── h3n2_ha_sequences.fasta     # Initial sequences
│   │   └── h3n2_ha_all.fasta           # All sequences (merged)
│   └── processed/
│       ├── h3n2_ha_comprehensive.csv   # Full metadata
│       ├── h3n2_ha_high_quality.csv    # High-quality subset
│       ├── h3n2_features.csv           # Features with metadata
│       ├── h3n2_features_matrix.csv    # Feature matrix only
│       ├── data_provenance.json        # Data provenance
│       └── feature_extraction_info.json # Feature extraction metadata
├── models/
│   ├── h3n2_binary_model.pkl           # Binary classifier
│   └── h3n2_multiclass_model.pkl       # Multi-class classifier
├── results/
│   ├── training_results.json           # Training metrics
│   ├── evaluation_results.json         # Evaluation metrics
│   ├── feature_analysis.json           # Feature analysis
│   ├── *_confusion_matrix.png          # Confusion matrices
│   ├── *_feature_importance.png        # Feature importance plots
│   ├── roc_curves.png                  # ROC curves
│   ├── precision_recall_curves.png     # PR curves
│   ├── learning_curves_*.png           # Learning curves
│   ├── feature_correlation_matrix.png  # Correlation heatmap
│   ├── feature_distributions.png       # Distribution plots
│   └── property_group_importance.png   # Property group analysis
├── scripts/
│   ├── download_comprehensive_h3n2.py  # Data collection
│   ├── extract_features.py             # Feature extraction
│   ├── train_model.py                  # Model training
│   ├── evaluate_model.py               # Model evaluation
│   ├── analyze_features.py             # Feature analysis
│   ├── predict_sequence.py             # Prediction interface
│   └── update_dashboard.py             # Dashboard update
├── src/
│   ├── __init__.py
│   ├── physicochemical.py              # Physicochemical calculations
│   ├── feature_extraction.py           # Feature extraction module
│   ├── preprocessing.py                # Data preprocessing
│   └── model.py                        # Model utilities
├── docs/
│   ├── METHODOLOGY.md                  # Detailed methodology
│   └── USER_GUIDE.md                   # User guide
├── notebooks/
│   └── 01_data_exploration.ipynb       # Exploratory analysis
├── .env                                # Environment variables (not in git)
├── .env.example                        # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/rofiperlungoding/pkm-flu-ml.git
cd pkm-flu-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure NCBI credentials
cp .env.example .env
# Edit .env with your NCBI email and API key
```

### Run Complete Pipeline

```bash
# 1. Download data from NCBI
python scripts/download_comprehensive_h3n2.py

# 2. Extract features
python scripts/extract_features.py

# 3. Train models
python scripts/train_model.py

# 4. Evaluate models
python scripts/evaluate_model.py

# 5. Analyze features
python scripts/analyze_features.py

# 6. Update dashboard
python scripts/update_dashboard.py
```

### Predict New Sequence

```bash
# From sequence string
python scripts/predict_sequence.py --sequence "MKTII..."

# From FASTA file
python scripts/predict_sequence.py --fasta input.fasta

# From NCBI accession
python scripts/predict_sequence.py --accession ABC12345

# Save results to JSON
python scripts/predict_sequence.py --fasta input.fasta --output predictions.json
```

### View Dashboard

```bash
# Open dashboard in browser
open dashboard/index.html  # macOS
start dashboard/index.html  # Windows
xdg-open dashboard/index.html  # Linux
```

---

## 📊 Pipeline Overview

```
┌─────────────────────┐
│  Data Collection    │  Download H3N2 HA sequences from NCBI
│  (NCBI Entrez API)  │  • 2,818 unique sequences (1996-2024)
└──────────┬──────────┘  • Quality scoring & deduplication
           │             • WHO vaccine reference strains
           ▼
┌─────────────────────┐
│ Feature Extraction  │  Extract 74 features per sequence
│  (Physicochemical)  │  • Amino acid composition (20)
└──────────┬──────────┘  • Physicochemical properties (30+)
           │             • Epitope site mutations (24)
           ▼
┌─────────────────────┐
│  Model Training     │  Train XGBoost classifiers
│    (XGBoost)        │  • Binary: Recent vs Historical
└──────────┬──────────┘  • Multi-class: 4 temporal periods
           │             • 80:20 train-test split
           ▼
┌─────────────────────┐
│  Model Evaluation   │  Comprehensive validation
│  (Cross-validation) │  • 5-fold stratified CV
└──────────┬──────────┘  • ROC curves, PR curves
           │             • Learning curves
           ▼
┌─────────────────────┐
│ Feature Analysis    │  Interpret model decisions
│  (Importance)       │  • Feature importance ranking
└──────────┬──────────┘  • Correlation analysis
           │             • Property group analysis
           ▼
┌─────────────────────┐
│    Prediction       │  Predict new sequences
│   (Interface)       │  • Sequence string / FASTA / Accession
└─────────────────────┘  • Binary & multi-class predictions
```

---

## 📚 Dokumentasi

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** - Metodologi penelitian lengkap
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - Panduan penggunaan detail
- **[Jupyter Notebook](notebooks/01_data_exploration.ipynb)** - Exploratory data analysis

---

## 🔬 Key Findings

1. **Model Performance:** Binary model mencapai akurasi near-perfect (99.55%), multi-class model excellent (93.48%)

2. **Feature Importance:** Epitope site mutations adalah prediktor terkuat, mengkonfirmasi peran antigenic drift

3. **Epitope Sites:** Site B dan D paling penting, konsisten dengan literatur sebagai target utama antibodi

4. **Physicochemical Properties:** Hydrophobicity dan charge berperan signifikan dalam perubahan antigenik

5. **Model Stability:** Cross-validation menunjukkan variance rendah, indikasi generalization yang baik

---

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **Biopython** - Sequence analysis
- **XGBoost** - Machine learning algorithm
- **Scikit-learn** - ML utilities and metrics
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **python-dotenv** - Environment management

---

## 📖 Referensi

1. **Koel, B. F., et al. (2013).** Substitutions near the receptor binding site determine major antigenic change during influenza virus evolution. *Science*, 342(6161), 976-979. [DOI](https://doi.org/10.1126/science.1244730)

2. **Li X, et al. (2024).** A sequence-based machine learning model for predicting antigenic distance for H3N2 influenza virus. *Front Microbiol.* [DOI](https://doi.org/10.3389/fmicb.2024.1345794)

3. **Smith, D. J., et al. (2004).** Mapping the antigenic and genetic evolution of influenza virus. *Science*, 305(5682), 371-376. [DOI](https://doi.org/10.1126/science.1097211)

4. **Bedford, T., et al. (2014).** Integrating influenza antigenic dynamics with molecular evolution. *eLife*, 3, e01914. [DOI](https://doi.org/10.7554/eLife.01914)

5. **WHO.** Recommended composition of influenza virus vaccines. https://www.who.int/teams/global-influenza-programme/vaccines

6. **NCBI Influenza Virus Resource.** https://www.ncbi.nlm.nih.gov/genomes/FLU/

---

## 👥 Contributors

- **Syifa Zavira Ramadhani** - Ketua, Bioteknologi
- **Rofi Perdana** - Anggota, Teknik Komputer

**Universitas Brawijaya**

---

## 📄 License

This project is for academic purposes (PKM-RE 2026).

---

## 🙏 Acknowledgments

- **NCBI** for providing comprehensive influenza sequence database
- **WHO** for vaccine strain recommendations
- **Universitas Brawijaya** for supporting this research
- **PKM-RE 2026** program

---

## 📧 Contact

For questions or collaborations:
- **Email:** syifazavira@student.ub.ac.id, mrofid@student.ub.ac.id
- **GitHub Issues:** https://github.com/rofiperlungoding/pkm-flu-ml/issues

---

*PKM-RE 2026 | Universitas Brawijaya*
