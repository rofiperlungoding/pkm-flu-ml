# User Guide
## H3N2 Antigenic Prediction Pipeline

Panduan lengkap untuk menggunakan pipeline machine learning prediksi perubahan antigenik H3N2.

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Collection](#data-collection)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Feature Analysis](#feature-analysis)
8. [Prediction](#prediction)
9. [Dashboard](#dashboard)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup

1. **Clone repository:**
```bash
git clone https://github.com/rofiperlungoding/pkm-flu-ml.git
cd pkm-flu-ml
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure NCBI credentials:**
```bash
# Copy template
cp .env.example .env

# Edit .env file
# Add your NCBI email and API key
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_api_key_here
```

**Get NCBI API Key:**
- Register at https://www.ncbi.nlm.nih.gov/account/
- Go to Settings → API Key Management
- Generate new API key

---

## Quick Start

### Complete Pipeline (All Steps)

```bash
# 1. Download data
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
```

---

## Data Collection

### Script: `download_comprehensive_h3n2.py`

**Purpose:** Download H3N2 HA sequences from NCBI with comprehensive metadata

**Features:**
- Downloads from multiple time periods (2023-2026, 2020-2022, 2015-2019)
- Includes WHO vaccine reference strains
- Automatic deduplication
- Quality scoring (0-10)
- Complete metadata extraction

**Usage:**
```bash
python scripts/download_comprehensive_h3n2.py
```

**Output Files:**
- `data/raw/h3n2_ha_all.fasta` - All sequences (FASTA format)
- `data/processed/h3n2_ha_comprehensive.csv` - Full metadata
- `data/processed/h3n2_ha_high_quality.csv` - High-quality subset (score ≥7)
- `data/processed/data_provenance.json` - Data provenance documentation

**Expected Runtime:** 15-30 minutes (depending on network speed)

**Quality Scoring:**
- Has year info: +3 points
- Has location: +2 points
- Human host: +2 points
- H3N2 confirmed: +2 points
- Has host info: +1 point
- **High quality:** ≥7 points

---

## Feature Extraction

### Script: `extract_features.py`

**Purpose:** Extract 74 physicochemical and epitope features from sequences

**Features Extracted:**
1. **Amino Acid Composition (20):** Frequency of each amino acid
2. **Physicochemical Properties (30+):**
   - Hydrophobicity (mean, variance, fractions)
   - Charge (mean, positive/negative fractions, pI)
   - Polarity (polar/non-polar fractions)
   - Aromaticity
   - Molecular weight, instability index, aliphatic index
3. **Epitope Site Analysis (24):**
   - Mutations per site (A, B, C, D, E)
   - Mutation rates
   - Hydrophobicity/charge changes
   - Total epitope mutations

**Usage:**
```bash
python scripts/extract_features.py
```

**Input:**
- `data/processed/h3n2_ha_comprehensive.csv`

**Output:**
- `data/processed/h3n2_features.csv` - Features with metadata
- `data/processed/h3n2_features_matrix.csv` - Feature matrix only (for ML)
- `data/processed/feature_extraction_info.json` - Extraction metadata

**Expected Runtime:** 5-10 minutes

**Reference Strain:** A/Perth/16/2009 (for mutation calculation)

---

## Model Training

### Script: `train_model.py`

**Purpose:** Train XGBoost models for temporal period classification

**Models:**
1. **Binary Model:** Recent (≥2020) vs Historical (<2020)
2. **Multi-class Model:** 4 temporal periods (<2010, 2010-2014, 2015-2019, ≥2020)

**Usage:**
```bash
python scripts/train_model.py
```

**Input:**
- `data/processed/h3n2_features_matrix.csv`
- `data/processed/h3n2_features.csv`

**Output:**
- `models/h3n2_binary_model.pkl` - Binary classifier
- `models/h3n2_multiclass_model.pkl` - Multi-class classifier
- `results/training_results.json` - Training metrics
- `results/binary_confusion_matrix.png` - Binary confusion matrix
- `results/multiclass_confusion_matrix.png` - Multi-class confusion matrix
- `results/binary_feature_importance.png` - Binary feature importance
- `results/multiclass_feature_importance.png` - Multi-class feature importance
- `results/year_distribution.png` - Data distribution by year

**Expected Runtime:** 2-5 minutes

**Hyperparameters:**
```python
n_estimators=200
max_depth=6
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
```

---

## Model Evaluation

### Script: `evaluate_model.py`

**Purpose:** Comprehensive model evaluation with multiple metrics

**Analyses:**
1. **Cross-validation:** 5-fold stratified CV
2. **ROC Curves:** With confidence intervals
3. **Precision-Recall Curves:** For imbalanced classes
4. **Learning Curves:** Detect overfitting/underfitting
5. **Detailed Classification Reports:** Per-class metrics

**Usage:**
```bash
python scripts/evaluate_model.py
```

**Output:**
- `results/evaluation_results.json` - All metrics in JSON
- `results/roc_curves.png` - ROC curves with 5-fold CV
- `results/precision_recall_curves.png` - PR curves
- `results/learning_curves_binary_model.png` - Binary learning curves
- `results/learning_curves_multi-class_model.png` - Multi-class learning curves
- `results/confusion_matrix_detailed_binary_model.png` - Detailed binary CM
- `results/confusion_matrix_detailed_multi-class_model.png` - Detailed multi-class CM

**Expected Runtime:** 5-10 minutes

**Metrics Reported:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC with standard deviation
- Per-class performance
- Cross-validation statistics

---

## Feature Analysis

### Script: `analyze_features.py`

**Purpose:** Deep dive into feature importance and correlations

**Analyses:**
1. **Feature Importance Ranking:** Top 20 features per model
2. **Feature Correlations:** Correlation matrix and highly correlated pairs
3. **Feature Distributions:** By temporal class
4. **Physicochemical Property Groups:** Grouped importance analysis

**Usage:**
```bash
python scripts/analyze_features.py
```

**Output:**
- `results/feature_analysis.json` - Analysis results
- `results/feature_importance_detailed_binary_model.png` - Binary feature importance
- `results/feature_importance_detailed_multi-class_model.png` - Multi-class feature importance
- `results/feature_correlation_matrix.png` - Correlation heatmap
- `results/feature_distributions.png` - Distribution plots
- `results/property_group_importance.png` - Property group analysis

**Expected Runtime:** 3-5 minutes

**Property Groups:**
- Amino Acid Composition
- Hydrophobicity
- Charge
- Polarity
- Aromaticity
- Epitope Sites
- Sequence Properties

---

## Prediction

### Script: `predict_sequence.py`

**Purpose:** Predict temporal period for new H3N2 HA sequences

**Input Methods:**
1. **Sequence string:** Direct amino acid sequence
2. **FASTA file:** Multiple sequences
3. **NCBI accession:** Download and predict

**Usage:**

```bash
# From sequence string
python scripts/predict_sequence.py --sequence "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAETQTAEKQTRMVTLLRNHCRQEQGAIYSLIRPNENPAHKSQLVWMACHSAAFEDLRLLSFIRGTKV"

# From FASTA file
python scripts/predict_sequence.py --fasta data/raw/new_sequences.fasta

# From NCBI accession
python scripts/predict_sequence.py --accession ABO21709.1

# Save results to JSON
python scripts/predict_sequence.py --fasta input.fasta --output predictions.json
```

**Output:**
- Console output with predictions
- Optional JSON file with detailed results

**Prediction Output:**
```
Binary Classification:
  Prediction: Recent (≥2020)
  Confidence: 98.45%
  Probabilities:
    Historical (<2020): 1.55%
    Recent (≥2020):     98.45%

Temporal Period Classification:
  Prediction: ≥2020 (Recent)
  Confidence: 96.23%
  Probabilities:
    <2010 (Historical):        0.12%
    2010-2014 (Mid-Historical): 0.89%
    2015-2019 (Mid-Recent):     2.76%
    ≥2020 (Recent):            96.23%

Key Features:
  Hydrophobicity: 0.0234
  Charge: -0.0156
  Aromaticity: 0.0789
  Epitope mutations: 45
```

**Sequence Requirements:**
- Length: 400-700 amino acids (typical HA length ~550)
- Valid amino acids only (ACDEFGHIKLMNPQRSTVWY)
- Protein sequence (not DNA)

---

## Dashboard

### Script: `update_dashboard.py`

**Purpose:** Update interactive HTML dashboard with latest results

**Usage:**
```bash
python scripts/update_dashboard.py
```

**Output:**
- `dashboard/data.json` - Dashboard data
- `dashboard/index.html` - Interactive dashboard (already exists)

**Dashboard Features:**
- Project information
- Data sources and statistics
- Pipeline progress tracker
- Model performance metrics
- Sample data preview
- Year distribution chart
- Quality distribution
- Location distribution

**View Dashboard:**
```bash
# Open in browser
open dashboard/index.html  # macOS
start dashboard/index.html  # Windows
xdg-open dashboard/index.html  # Linux
```

---

## Troubleshooting

### Common Issues

#### 1. NCBI API Rate Limiting
**Error:** `HTTP Error 429: Too Many Requests`

**Solution:**
- Add API key to `.env` file (increases rate limit from 3 to 10 requests/second)
- Add delays between requests (already implemented)
- Run during off-peak hours

#### 2. Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### 3. Environment Variables Not Loading
**Error:** `Missing NCBI credentials!`

**Solution:**
- Ensure `.env` file exists in project root
- Check file format (no quotes around values)
- Verify file is not named `.env.txt`

#### 4. Memory Issues
**Error:** `MemoryError` during training

**Solution:**
- Reduce batch size in download scripts
- Use high-quality subset only
- Increase system RAM or use cloud instance

#### 5. Feature Extraction Fails
**Error:** `KeyError` or `ValueError` during feature extraction

**Solution:**
- Check sequence format (must be protein, not DNA)
- Verify reference strain is available
- Check for invalid characters in sequences

#### 6. Model Loading Fails
**Error:** `FileNotFoundError: models/xxx.pkl`

**Solution:**
- Run training script first: `python scripts/train_model.py`
- Check models directory exists
- Verify file permissions

### Getting Help

**GitHub Issues:** https://github.com/rofiperlungoding/pkm-flu-ml/issues

**Contact:**
- Syifa Zavira Ramadhani (Ketua)
- Rofi Perdana (Anggota)
- Universitas Brawijaya

---

## Best Practices

### 1. Data Collection
- Run during off-peak hours for faster downloads
- Use API key for better rate limits
- Verify data quality scores before training
- Keep provenance documentation

### 2. Feature Extraction
- Use consistent reference strain (A/Perth/16/2009)
- Validate sequences before extraction
- Check for missing values
- Document feature engineering decisions

### 3. Model Training
- Use stratified splits to maintain class balance
- Set random seed for reproducibility
- Save models with version numbers
- Document hyperparameters

### 4. Model Evaluation
- Always use cross-validation
- Check for overfitting with learning curves
- Analyze feature importance for interpretability
- Compare multiple metrics

### 5. Prediction
- Validate input sequences
- Check prediction confidence
- Use ensemble predictions for critical decisions
- Document prediction provenance

---

## File Structure

```
pkm-flu-ml/
├── data/
│   ├── raw/                    # Raw FASTA files
│   └── processed/              # Processed CSV and JSON
├── models/                     # Trained models (.pkl)
├── results/                    # Plots and metrics
├── scripts/                    # Python scripts
│   ├── download_comprehensive_h3n2.py
│   ├── extract_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── analyze_features.py
│   ├── predict_sequence.py
│   └── update_dashboard.py
├── src/                        # Source modules
│   ├── feature_extraction.py
│   ├── physicochemical.py
│   ├── preprocessing.py
│   └── model.py
├── dashboard/                  # Interactive dashboard
│   ├── index.html
│   └── data.json
├── docs/                       # Documentation
│   ├── METHODOLOGY.md
│   └── USER_GUIDE.md
├── notebooks/                  # Jupyter notebooks
├── .env                        # Environment variables (not in git)
├── .env.example                # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Citation

If you use this pipeline in your research, please cite:

```
Ramadhani, S. Z., & Perdana, R. (2026). 
Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2 
Menggunakan Machine Learning. 
PKM-RE, Universitas Brawijaya.
```

---

**Last Updated:** January 18, 2026  
**Version:** 1.0.0  
**Authors:** Syifa Zavira Ramadhani & Rofi Perdana
