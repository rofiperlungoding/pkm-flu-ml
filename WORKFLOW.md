# 🚀 WORKFLOW LENGKAP: H3N2 ML Pipeline
## Step-by-Step Guide - Tinggal Copy-Paste!

**PKM-RE Team: Syifa & Rofi**  
**Panduan ini dibuat super jelas - tinggal ikutin aja sat-set! ✨**

---

## 📋 Table of Contents
1. [Setup Awal (Sekali Aja)](#1-setup-awal-sekali-aja)
2. [Workflow Basic (Cepat & Simple)](#2-workflow-basic-cepat--simple)
3. [Workflow Advanced (Super Lengkap)](#3-workflow-advanced-super-lengkap)
4. [Prediksi Sequence Baru](#4-prediksi-sequence-baru)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Setup Awal (Sekali Aja)

### Step 1.1: Install Dependencies
```bash
# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

**Catatan:** Kalau ada error, install satu-satu:
```bash
pip install pandas numpy biopython scikit-learn xgboost matplotlib seaborn python-dotenv joblib requests
```

### Step 1.2: Setup API Key NCBI
```bash
# Copy template .env
cp .env.example .env
```

**Edit file `.env`** (pakai notepad/text editor):
```
NCBI_EMAIL=opikopi32@gmail.com
NCBI_API_KEY=your_api_key_here
```

**Cara dapetin API Key:**
1. Buka https://www.ncbi.nlm.nih.gov/account/
2. Login/Register
3. Settings → API Key Management
4. Create new API key
5. Copy paste ke `.env`

### Step 1.3: Cek Struktur Folder
```bash
# Pastikan folder-folder ini ada
mkdir -p data/raw data/processed data/advanced
mkdir -p models models/advanced
mkdir -p results results/advanced results/batch
mkdir -p dashboard
mkdir -p tests
```

**✅ Setup selesai! Sekarang tinggal pilih workflow mana yang mau dijalanin.**

---

## 2. Workflow Basic (Cepat & Simple)

**Cocok untuk:** Testing cepat, demo, atau kalau mau hasil cepat  
**Waktu:** ~30-60 menit  
**Akurasi:** 99.5% (binary), 93.5% (multiclass)

### Step 2.1: Download Data
```bash
python scripts/download_comprehensive_h3n2.py
```

**Output:**
- `data/processed/h3n2_ha_comprehensive.csv` (2,818 sequences)
- `data/processed/h3n2_ha_high_quality.csv` (2,204 sequences)

**Cek hasilnya:**
```bash
# Lihat jumlah data
python -c "import pandas as pd; df = pd.read_csv('data/processed/h3n2_ha_comprehensive.csv'); print(f'Total sequences: {len(df)}'); print(f'Year range: {df.collection_year.min():.0f}-{df.collection_year.max():.0f}')"
```

### Step 2.2: Extract Features
```bash
python scripts/extract_features.py
```

**Output:**
- `data/processed/h3n2_features.csv` (74 features)
- `data/processed/h3n2_features_matrix.csv` (feature matrix)
- `data/processed/feature_extraction_info.json` (metadata)

**Cek hasilnya:**
```bash
# Lihat jumlah features
python -c "import pandas as pd; df = pd.read_csv('data/processed/h3n2_features_matrix.csv'); print(f'Total features: {df.shape[1]}'); print(f'Total samples: {df.shape[0]}')"
```

### Step 2.3: Train Model
```bash
python scripts/train_model.py
```

**Output:**
- `models/h3n2_binary_model.pkl` (Recent vs Historical)
- `models/h3n2_multiclass_model.pkl` (4 periods)
- `results/training_results.json` (metrics)
- `results/binary_confusion_matrix.png`
- `results/multiclass_confusion_matrix.png`
- `results/binary_feature_importance.png`
- `results/multiclass_feature_importance.png`

**Cek hasilnya:**
```bash
# Lihat akurasi model
python -c "import json; data = json.load(open('results/training_results.json')); print(f\"Binary Accuracy: {data['binary']['test_accuracy']:.2%}\"); print(f\"Multiclass Accuracy: {data['multiclass']['test_accuracy']:.2%}\")"
```

### Step 2.4: Evaluate Model (Optional)
```bash
python scripts/evaluate_model.py
```

**Output:**
- Cross-validation results
- ROC curves
- PR curves
- Learning curves

### Step 2.5: Update Dashboard
```bash
python scripts/update_dashboard.py
```

**Buka dashboard:**
```bash
# Windows
start dashboard/index.html

# macOS
open dashboard/index.html

# Linux
xdg-open dashboard/index.html
```

**✅ SELESAI! Model basic sudah siap dipakai!**

---

## 3. Workflow Advanced (Super Lengkap)

**Cocok untuk:** Penelitian, paper, hasil terbaik  
**Waktu:** ~2-4 jam (tergantung hardware)  
**Akurasi:** 99.8% (binary), 95.5% (multiclass)  
**Features:** 200+ features, ensemble models, deep learning

### Step 3.1: Advanced Data Collection
```bash
python scripts/advanced_data_collection.py
```

**Output:**
- `data/advanced/h3n2_ha_advanced.csv` (semua data + metadata lengkap)
- `data/advanced/h3n2_ha_ultra_high_quality.csv` (quality ≥12)
- `data/advanced/h3n2_ha_clade_*.csv` (per clade)
- `data/advanced/data_collection_report.json`

**Fitur tambahan:**
- ✅ Phylogenetic clade assignment (7 clades)
- ✅ Glycosylation site prediction
- ✅ Enhanced quality scoring (0-15)
- ✅ 30+ metadata fields

**Cek hasilnya:**
```bash
# Lihat statistik data
python -c "import pandas as pd; df = pd.read_csv('data/advanced/h3n2_ha_advanced.csv'); print(f'Total: {len(df)}'); print(f'Ultra HQ: {len(df[df.quality_score>=12])}'); print(f'Clades: {df.phylogenetic_clade.nunique()}')"
```

### Step 3.2: Advanced Feature Extraction
```bash
python scripts/advanced_feature_extraction.py
```

**Output:**
- `data/advanced/h3n2_advanced_features.csv` (200+ features)
- `data/advanced/h3n2_advanced_features_matrix.csv`
- `data/advanced/advanced_feature_info.json`

**Feature categories:**
- ✅ Basic physicochemical (74)
- ✅ Structural features (30+)
- ✅ Evolutionary features (20+)
- ✅ Sequence complexity (15+)
- ✅ Position-specific (30+)
- ✅ Deep learning embeddings (54, optional)

**Catatan:** Kalau mau pakai deep learning embeddings (ESM-2), install dulu:
```bash
pip install transformers torch
```

**Cek hasilnya:**
```bash
# Lihat jumlah features
python -c "import pandas as pd; df = pd.read_csv('data/advanced/h3n2_advanced_features_matrix.csv'); print(f'Total features: {df.shape[1]}'); print(f'Total samples: {df.shape[0]}')"
```

### Step 3.3: Run Tests (Optional tapi Recommended)
```bash
# Install pytest kalau belum
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run dengan coverage report
pytest tests/ --cov=src --cov-report=html
```

**Cek coverage:**
```bash
# Windows
start htmlcov/index.html

# macOS/Linux
open htmlcov/index.html
```

### Step 3.4: Advanced Model Training
```bash
python scripts/advanced_model_training.py
```

**⚠️ PENTING:** Ini akan training banyak model, butuh waktu lama!

**Models yang di-train:**
1. ✅ Stacking Ensemble (6 base models + meta-learner)
2. ✅ Voting Ensemble (soft & hard)
3. ✅ Multi-Layer Perceptron (MLP)
4. ✅ 1D CNN (kalau TensorFlow installed)
5. ✅ CatBoost (kalau installed)
6. ✅ LightGBM (kalau installed)

**Output:**
- `models/advanced/*_model.pkl` (semua model)
- `models/advanced/*_scaler.pkl` (scalers)
- `models/advanced/*_calibrated.pkl` (calibrated models)
- `results/advanced/advanced_training_results.json`
- `results/advanced/model_comparison.csv`
- `results/advanced/model_comparison.png`
- `results/advanced/shap_*.pkl` (SHAP values)
- `results/advanced/shap_summary_*.png`
- `results/advanced/calibration_*.png`

**Cek hasilnya:**
```bash
# Lihat perbandingan model
python -c "import pandas as pd; df = pd.read_csv('results/advanced/model_comparison.csv'); print(df.sort_values('accuracy', ascending=False).head(10).to_string(index=False))"
```

### Step 3.5: Analyze Features
```bash
python scripts/analyze_features.py
```

**Output:**
- Feature importance rankings
- Correlation analysis
- Feature distributions
- Visualization plots

### Step 3.6: Update Dashboard
```bash
python scripts/update_dashboard.py
```

**✅ SELESAI! Advanced system sudah siap!**

---

## 4. Prediksi Sequence Baru

### Option A: Single Sequence

**Dari sequence string:**
```bash
python scripts/predict_sequence.py --sequence "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLNFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAEQTDQTRMQAIVTDTGSADTVSLPTQSTDVQICDPKFSGDSSSQVKSELSA" --output results/prediction_result.json
```

**Dari FASTA file:**
```bash
python scripts/predict_sequence.py --fasta my_sequence.fasta --output results/prediction_result.json
```

**Dari NCBI accession:**
```bash
python scripts/predict_sequence.py --accession ABC12345 --output results/prediction_result.json
```

### Option B: Batch Prediction (Banyak Sequences)

**Basic batch prediction:**
```bash
python scripts/batch_prediction.py \
    --fasta sequences.fasta \
    --output batch_results.csv
```

**Advanced batch dengan ensemble:**
```bash
python scripts/batch_prediction.py \
    --fasta sequences.fasta \
    --output batch_results.json \
    --model-type advanced \
    --ensemble \
    --n-jobs 8 \
    --analyze \
    --analysis-dir results/batch_analysis
```

**Parameters:**
- `--fasta`: File FASTA input
- `--output`: File output (.csv atau .json)
- `--model-type`: basic / advanced / all
- `--ensemble`: Pakai ensemble prediction
- `--n-jobs`: Jumlah CPU cores (8 = pakai 8 cores, -1 = pakai semua)
- `--batch-size`: Jumlah sequence per batch (default: 100)
- `--analyze`: Generate analysis plots
- `--analysis-dir`: Folder untuk hasil analysis

**Output:**
- Prediction results (CSV/JSON)
- Statistical analysis (JSON)
- Visualization plots (PNG)

---

## 5. Troubleshooting

### Problem 1: Import Error
```
ImportError: cannot import name 'FeatureExtractor'
```

**Solution:**
```bash
# Pastikan di folder root project
cd /path/to/pkm-flu-ml

# Cek struktur folder
ls -la src/
```

### Problem 2: NCBI API Error
```
HTTPError: 429 Too Many Requests
```

**Solution:**
- Tunggu 1-2 menit
- Pastikan API key sudah di-set di `.env`
- Kalau masih error, kurangi jumlah request dengan edit script

### Problem 3: Memory Error
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Kurangi batch size
python scripts/batch_prediction.py --batch-size 50 ...

# Atau pakai sequential processing
python scripts/batch_prediction.py --n-jobs 1 ...
```

### Problem 4: Model Not Found
```
FileNotFoundError: models/h3n2_binary_model.pkl
```

**Solution:**
```bash
# Train model dulu
python scripts/train_model.py
```

### Problem 5: Feature Mismatch
```
ValueError: Feature names should match
```

**Solution:**
- Pastikan pakai feature extraction yang sama dengan training
- Kalau pakai basic model, pakai `extract_features.py`
- Kalau pakai advanced model, pakai `advanced_feature_extraction.py`

---

## 📊 Quick Reference: Command Cheatsheet

### Basic Workflow (Copy-Paste Semua)
```bash
# 1. Download data
python scripts/download_comprehensive_h3n2.py

# 2. Extract features
python scripts/extract_features.py

# 3. Train model
python scripts/train_model.py

# 4. Evaluate
python scripts/evaluate_model.py

# 5. Update dashboard
python scripts/update_dashboard.py
```

### Advanced Workflow (Copy-Paste Semua)
```bash
# 1. Advanced data collection
python scripts/advanced_data_collection.py

# 2. Advanced feature extraction
python scripts/advanced_feature_extraction.py

# 3. Run tests
pytest tests/ -v

# 4. Train advanced models
python scripts/advanced_model_training.py

# 5. Analyze features
python scripts/analyze_features.py

# 6. Update dashboard
python scripts/update_dashboard.py
```

### Prediction Commands
```bash
# Single sequence
python scripts/predict_sequence.py --sequence "MKTII..." --output result.json

# Batch prediction
python scripts/batch_prediction.py --fasta input.fasta --output results.csv --analyze
```

---

## 🎯 Recommended Workflow untuk Paper/Penelitian

**Untuk hasil terbaik dan paling lengkap:**

```bash
# Step 1: Setup (sekali aja)
pip install -r requirements.txt
cp .env.example .env
# Edit .env dengan API key

# Step 2: Data Collection
python scripts/advanced_data_collection.py

# Step 3: Feature Engineering
python scripts/advanced_feature_extraction.py

# Step 4: Testing
pytest tests/ --cov=src --cov-report=html

# Step 5: Model Training
python scripts/advanced_model_training.py

# Step 6: Analysis
python scripts/analyze_features.py
python scripts/evaluate_model.py

# Step 7: Dashboard
python scripts/update_dashboard.py

# Step 8: Prediction
python scripts/batch_prediction.py \
    --fasta new_sequences.fasta \
    --output predictions.csv \
    --model-type advanced \
    --ensemble \
    --analyze \
    --analysis-dir results/final_analysis
```

**✅ Selesai! Semua hasil ada di folder `results/` dan `models/`**

---

## 📁 Output Files Reference

### Basic Workflow
```
data/processed/
├── h3n2_ha_comprehensive.csv          # Raw data
├── h3n2_features.csv                  # Features + metadata
└── h3n2_features_matrix.csv           # Feature matrix only

models/
├── h3n2_binary_model.pkl              # Binary classifier
└── h3n2_multiclass_model.pkl          # Multiclass classifier

results/
├── training_results.json              # Metrics
├── binary_confusion_matrix.png        # Confusion matrix
├── multiclass_confusion_matrix.png
├── binary_feature_importance.png      # Feature importance
└── multiclass_feature_importance.png
```

### Advanced Workflow
```
data/advanced/
├── h3n2_ha_advanced.csv               # Advanced data
├── h3n2_ha_ultra_high_quality.csv     # Ultra HQ data
├── h3n2_advanced_features.csv         # 200+ features
└── h3n2_advanced_features_matrix.csv

models/advanced/
├── stacking_binary_model.pkl          # Stacking ensemble
├── voting_soft_binary_model.pkl       # Voting ensemble
├── mlp_binary_model.pkl               # Neural network
├── catboost_binary_model.pkl          # CatBoost
└── *_scaler.pkl                       # Feature scalers

results/advanced/
├── advanced_training_results.json     # All metrics
├── model_comparison.csv               # Model comparison
├── model_comparison.png               # Visualization
├── shap_*.pkl                         # SHAP values
├── shap_summary_*.png                 # SHAP plots
└── calibration_*.png                  # Calibration curves
```

---

## 💡 Tips & Best Practices

### 1. Untuk Development/Testing
- Pakai **basic workflow** dulu
- Cepat dan cukup akurat (99.5%)
- Cocok untuk iterasi cepat

### 2. Untuk Production/Paper
- Pakai **advanced workflow**
- Akurasi maksimal (99.8%)
- Lengkap dengan interpretability

### 3. Untuk Batch Prediction
- Pakai `--n-jobs -1` untuk speed maksimal
- Pakai `--checkpoint-interval 500` untuk safety
- Pakai `--analyze` untuk insights

### 4. Untuk Reproducibility
- Simpan semua output files
- Catat versi library (`pip freeze > requirements_exact.txt`)
- Backup model files

### 5. Untuk Debugging
- Cek log files di console
- Pakai `--n-jobs 1` untuk sequential processing
- Run tests dengan `pytest -v` untuk detail

---

## 🚀 Next Steps

Setelah workflow selesai, kamu bisa:

1. **Analisis Hasil:**
   - Buka dashboard (`dashboard/index.html`)
   - Lihat feature importance
   - Analisis SHAP values

2. **Prediksi Baru:**
   - Pakai `predict_sequence.py` untuk single
   - Pakai `batch_prediction.py` untuk batch

3. **Improve Model:**
   - Tambah data baru
   - Tune hyperparameters
   - Coba feature engineering baru

4. **Deploy:**
   - Buat REST API (Flask/FastAPI)
   - Containerize dengan Docker
   - Deploy ke cloud

---

## 📞 Contact

**Ada masalah? Hubungi:**
- Email: opikopi32@gmail.com
- GitHub: https://github.com/rofiperlungoding/pkm-flu-ml

---

**Last Updated:** January 18, 2026  
**Version:** 2.0 (Advanced System)

**Happy Coding! 🎉**


---

## 6. Workflow Advanced Pipeline (Super Lengkap & Akademis)

### 🎯 Kapan Pakai Advanced Pipeline?
- Untuk penelitian yang lebih mendalam
- Butuh akurasi maksimal (>99%)
- Perlu interpretability (SHAP analysis)
- Dataset besar (>5000 sequences)
- Publikasi jurnal internasional

### Step 6.1: Run Automated Advanced Pipeline

**Cara Termudah - Otomatis Semua:**
```bash
python run_advanced_pipeline.py
```

**Atau skip data collection (pakai data existing):**
```bash
python run_advanced_pipeline.py --skip-data-collection
```

**Atau skip data collection & feature extraction:**
```bash
python run_advanced_pipeline.py --skip-data-collection --skip-feature-extraction
```

### Step 6.2: Manual Advanced Pipeline (Step-by-Step)

#### 6.2.1 Advanced Data Collection
```bash
python scripts/advanced_data_collection.py
```

**Output:**
- `data/advanced/h3n2_ha_advanced.csv` - All sequences dengan metadata lengkap
- `data/advanced/h3n2_ha_ultra_high_quality.csv` - Quality score ≥12
- `data/advanced/h3n2_ha_clade_*.csv` - Per-clade datasets

**Fitur:**
- ✅ Phylogenetic clade assignment (7 H3N2 clades)
- ✅ Glycosylation site prediction
- ✅ Enhanced quality scoring (0-15 scale)
- ✅ 30+ metadata fields

**Durasi:** ~30-60 menit

#### 6.2.2 Advanced Feature Extraction
```bash
python scripts/advanced_feature_extraction.py
```

**Output:**
- `data/advanced/h3n2_advanced_features.csv` - Features + metadata
- `data/advanced/h3n2_advanced_features_matrix.csv` - Feature matrix
- `data/advanced/advanced_feature_info.json` - Feature documentation

**Fitur:**
- ✅ 200+ features total:
  - Basic physicochemical (74)
  - Structural features (30+)
  - Evolutionary features (20+)
  - Sequence complexity (15+)
  - Position-specific (30+)
  - Deep learning embeddings (54, optional)

**Durasi:** ~10-30 menit (lebih lama kalau pakai deep learning embeddings)

#### 6.2.3 Advanced Model Training
```bash
python scripts/advanced_model_training.py
```

**Output:**
- `models/advanced/*.pkl` - Trained models
- `results/advanced/advanced_training_results.json` - Metrics
- `results/advanced/model_comparison.csv` - Performance comparison
- `results/advanced/shap_*.pkl` - SHAP values
- Various plots (SHAP, calibration, comparison)

**Models Trained:**
- ✅ Stacking Ensemble (6 base models + meta-learner)
- ✅ Voting Ensemble (hard & soft)
- ✅ Multi-Layer Perceptron (MLP)
- ✅ 1D CNN (Convolutional Neural Network)
- ✅ CatBoost
- ✅ LightGBM

**Interpretability:**
- ✅ SHAP analysis
- ✅ Model calibration
- ✅ Feature importance

**Durasi:** ~30-120 menit (tergantung hardware)

#### 6.2.4 Update Dashboard
```bash
python scripts/update_dashboard.py
```

### Step 6.3: Batch Prediction (High Performance)

**Basic Usage:**
```bash
python scripts/batch_prediction.py \
    --fasta input_sequences.fasta \
    --output results.csv
```

**Advanced Usage dengan Ensemble:**
```bash
python scripts/batch_prediction.py \
    --fasta input_sequences.fasta \
    --output results.json \
    --model-type advanced \
    --ensemble \
    --n-jobs 8 \
    --batch-size 200 \
    --analyze \
    --analysis-dir results/batch_analysis
```

**Parameters:**
- `--model-type`: basic, advanced, atau all
- `--ensemble`: Gunakan ensemble predictions
- `--n-jobs`: Jumlah parallel workers (-1 untuk semua CPU)
- `--batch-size`: Sequences per batch (100-500)
- `--analyze`: Lakukan statistical analysis
- `--analysis-dir`: Directory untuk output analysis

**Fitur:**
- ✅ Parallel processing (multiprocessing)
- ✅ Progress tracking
- ✅ Checkpoint system (resumable)
- ✅ Memory efficient
- ✅ Ensemble aggregation
- ✅ Statistical analysis & visualization

**Performance:**
- 100-1000 sequences per minute (tergantung hardware)
- Automatic load balancing
- Checkpoint every 500 sequences

---

## 7. Perbandingan Basic vs Advanced

| Aspek | Basic Pipeline | Advanced Pipeline |
|-------|---------------|-------------------|
| **Features** | 74 features | 200+ features |
| **Models** | XGBoost | Ensemble + Deep Learning |
| **Accuracy** | 99.5% | 99.8%+ |
| **Training Time** | ~5-10 menit | ~30-120 menit |
| **Interpretability** | Feature importance | SHAP + Calibration |
| **Batch Processing** | Sequential | Parallel + Checkpointing |
| **Use Case** | Quick analysis | Research & Production |

---

## 8. Tips & Best Practices

### Data Collection
✅ Gunakan NCBI API key untuk download lebih cepat  
✅ Cek data quality scores  
✅ Verifikasi temporal distribution  
✅ Include diverse geographic locations  

### Feature Extraction
✅ Pastikan reference sequence benar (A/Perth/16/2009)  
✅ Cek missing values  
✅ Validate feature ranges  
✅ Gunakan advanced features untuk performa terbaik  

### Model Training
✅ Monitor cross-validation scores  
✅ Cek overfitting (train vs test performance)  
✅ Save model artifacts dan metadata  
✅ Gunakan ensemble methods untuk aplikasi critical  

### Batch Prediction
✅ Gunakan parallel processing untuk dataset besar  
✅ Enable checkpointing untuk long-running jobs  
✅ Analyze prediction confidence dan uncertainty  
✅ Gunakan ensemble predictions untuk reliability tinggi  

### Performance Optimization
✅ Gunakan SSD untuk I/O lebih cepat  
✅ Increase batch size untuk throughput lebih baik  
✅ Gunakan GPU untuk deep learning models (kalau ada)  
✅ Monitor memory usage untuk large datasets  

---

## 9. Workflow Diagram Lengkap

### Basic Pipeline
```
Data Collection → Feature Extraction → Model Training → Evaluation → Dashboard
     (5 min)           (2 min)            (5 min)        (3 min)     (1 min)
```

### Advanced Pipeline
```
Advanced Data → Advanced Features → Advanced Training → Batch Prediction
   (30-60 min)      (10-30 min)       (30-120 min)       (variable)
                                            ↓
                                    SHAP Analysis
                                    Calibration
                                    Ensemble Methods
```

---

## 10. Next Steps Setelah Pipeline Selesai

### 1. Explore Results
- 📊 View interactive dashboard (`dashboard/index.html`)
- 📈 Analyze feature importance
- 🎯 Check model performance metrics
- 🔍 Review SHAP interpretability plots

### 2. Make Predictions
- 🧬 Test dengan new sequences
- 📦 Batch process large datasets
- 🔌 Integrate into applications

### 3. Iterate & Improve
- 📥 Collect more recent data
- ⚙️ Engineer new features
- 🎛️ Tune hyperparameters
- 🏗️ Try different model architectures

### 4. Deploy
- 🌐 Create REST API for predictions
- 💻 Build web interface
- 🔗 Integrate dengan surveillance systems

---

## 11. Dokumentasi Lengkap

Untuk informasi lebih detail, lihat:

- 📖 **USER_GUIDE.md** - Panduan lengkap setiap script
- 🔬 **METHODOLOGY.md** - Metodologi ilmiah
- 🏗️ **ADVANCED_SYSTEM.md** - Arsitektur sistem advanced
- 🔧 **TROUBLESHOOTING.md** - Solusi masalah umum
- 🚀 **QUICKSTART.md** - Quick start guide

---

## 12. Support & Contact

Butuh bantuan? Hubungi:
- 📧 Email: opikopi32@gmail.com
- 🐙 GitHub: https://github.com/rofiperlungoding/pkm-flu-ml
- 📚 Documentation: `docs/` folder

---

**🎉 Selamat! Sistem ML H3N2 Antigenic Prediction sudah siap digunakan!**

**Last Updated:** January 18, 2026  
**PKM-RE Team:** Syifa Zavira Ramadhani & Rofi Perdana  
**Universitas Brawijaya**
