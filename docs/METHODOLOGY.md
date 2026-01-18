# Metodologi Penelitian
## Analisis Prediksi Perubahan Antigenik Virus Influenza H3N2 Menggunakan Machine Learning

**Tim PKM-RE:**
- Syifa Zavira Ramadhani (Ketua - Bioteknologi)
- Rofi Perdana (Anggota - Teknik Komputer)

**Institusi:** Universitas Brawijaya  
**Tahun:** 2026

---

## 1. Pendahuluan

### 1.1 Latar Belakang
Virus influenza A subtipe H3N2 mengalami perubahan antigenik (antigenic drift) yang cepat, menyebabkan:
- Penurunan efektivitas vaksin
- Epidemi musiman yang berulang
- Tantangan dalam pemilihan strain vaksin

### 1.2 Tujuan Penelitian
1. Mengembangkan model machine learning untuk memprediksi periode temporal strain H3N2
2. Mengidentifikasi fitur fisikokimia yang berkorelasi dengan perubahan antigenik
3. Memberikan tools prediksi untuk mendukung surveillance epidemiologi

### 1.3 Hipotesis
Perubahan komposisi asam amino dan sifat fisikokimia protein hemagglutinin (HA) dapat digunakan untuk memprediksi periode temporal strain H3N2 dengan akurasi tinggi.

---

## 2. Metodologi

### 2.1 Pengumpulan Data

#### 2.1.1 Sumber Data
- **Database:** NCBI Protein Database
- **Target:** Protein hemagglutinin (HA) H3N2
- **Host:** Human
- **Periode:** 1996-2024
- **Total sekuens:** 2,818 unique sequences

#### 2.1.2 Kriteria Inklusi
- Panjang sekuens: 500-600 asam amino
- Host: Homo sapiens (human)
- Subtipe: H3N2 confirmed
- Metadata lengkap (tahun, lokasi)

#### 2.1.3 Quality Scoring
Setiap sekuens diberi skor kualitas (0-10) berdasarkan:
- Informasi tahun koleksi: +3 poin
- Informasi lokasi: +2 poin
- Host manusia terkonfirmasi: +2 poin
- Subtipe H3N2 terkonfirmasi: +2 poin
- Informasi host tersedia: +1 poin

**High-quality threshold:** ≥7 poin

#### 2.1.4 Strain Referensi WHO
Termasuk strain vaksin WHO 2010-2025:
- A/Perth/16/2009 (referensi utama)
- A/Darwin/6/2021, A/Darwin/9/2021
- A/Hong Kong/45/2019, A/Kansas/14/2017
- Dan lainnya (total 15 strain referensi)

### 2.2 Preprocessing Data

#### 2.2.1 Deduplication
- Menggunakan MD5 hash untuk identifikasi duplikat
- Menggabungkan data dari multiple sources
- Menghapus sekuens identik

#### 2.2.2 Labelling
**Binary Classification:**
- Class 0: Historical (<2020) - 1,363 samples
- Class 1: Recent (≥2020) - 1,455 samples

**Multi-class Classification:**
- Class 0: <2010 (Historical) - 234 samples
- Class 1: 2010-2014 (Mid-Historical) - 459 samples
- Class 2: 2015-2019 (Mid-Recent) - 670 samples
- Class 3: ≥2020 (Recent) - 1,455 samples

### 2.3 Feature Extraction

#### 2.3.1 Amino Acid Composition (20 features)
Frekuensi relatif setiap asam amino (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)

#### 2.3.2 Physicochemical Properties (30+ features)

**Hydrophobicity:**
- Mean hydrophobicity (Kyte-Doolittle scale)
- Hydrophobic residue fraction
- Hydrophilic residue fraction
- Hydrophobicity variance

**Charge Properties:**
- Mean charge
- Positive charge fraction (K, R, H)
- Negative charge fraction (D, E)
- Net charge
- Isoelectric point (pI)

**Polarity:**
- Polar residue fraction
- Non-polar residue fraction
- Polarity ratio

**Aromaticity:**
- Aromatic residue fraction (F, W, Y)

**Molecular Properties:**
- Molecular weight
- Instability index
- Aliphatic index

#### 2.3.3 Epitope Site Analysis (24 features)
Berdasarkan 5 epitope sites H3N2 (Koel et al., 2013):
- **Site A:** 122, 124, 126, 130, 131, 132, 133, 135, 137, 142, 143, 145
- **Site B:** 128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198
- **Site C:** 44, 45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312
- **Site D:** 96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248
- **Site E:** 57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265

**Features per site:**
- Mutation count vs reference (A/Perth/16/2009)
- Mutation rate
- Hydrophobicity change
- Charge change

**Total epitope mutations:** Sum across all sites

#### 2.3.4 Sequence Properties
- Sequence length
- Secondary structure propensity
- Flexibility
- Surface accessibility

**Total Features:** 74 features

### 2.4 Machine Learning Models

#### 2.4.1 Algorithm Selection
**XGBoost (Extreme Gradient Boosting)**

Alasan pemilihan:
- Performa tinggi untuk data tabular
- Handling missing values
- Feature importance built-in
- Regularization untuk mencegah overfitting
- Efisien untuk dataset medium-size

#### 2.4.2 Model Architecture

**Binary Model:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
```

**Multi-class Model:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=4,
    random_state=42
)
```

#### 2.4.3 Training Strategy
- **Train-test split:** 80:20
- **Stratification:** Mempertahankan proporsi kelas
- **Random state:** 42 (reproducibility)
- **Cross-validation:** 5-fold stratified CV

### 2.5 Evaluation Metrics

#### 2.5.1 Classification Metrics
- **Accuracy:** Overall correctness
- **Precision:** Positive predictive value
- **Recall (Sensitivity):** True positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed error analysis

#### 2.5.2 Probabilistic Metrics
- **ROC Curve:** Receiver Operating Characteristic
- **AUC:** Area Under ROC Curve
- **Precision-Recall Curve:** For imbalanced classes

#### 2.5.3 Model Validation
- **Cross-validation:** 5-fold stratified
- **Learning curves:** Detect overfitting/underfitting
- **Feature importance:** Interpretability

---

## 3. Hasil

### 3.1 Model Performance

#### 3.1.1 Binary Classification (Recent vs Historical)
- **Accuracy:** 99.55%
- **Precision:** 99.58%
- **Recall:** 99.65%
- **F1-Score:** 99.62%
- **AUC-ROC:** 0.9998

**Confusion Matrix:**
```
                Predicted
              Historical  Recent
Actual
Historical       272        1
Recent             1       290
```

#### 3.1.2 Multi-class Classification (4 Periods)
- **Accuracy:** 93.48%
- **Macro Avg Precision:** 93.52%
- **Macro Avg Recall:** 93.48%
- **Macro Avg F1-Score:** 93.48%

**Per-class Performance:**
- Class 0 (<2010): Precision 95.65%, Recall 93.62%
- Class 1 (2010-2014): Precision 91.30%, Recall 91.30%
- Class 2 (2015-2019): Precision 92.54%, Recall 92.54%
- Class 3 (≥2020): Precision 94.59%, Recall 96.53%

### 3.2 Feature Importance

#### 3.2.1 Top 10 Most Important Features (Binary Model)
1. **epitope_site_mutations** (0.0847) - Total mutations across epitope sites
2. **site_B_mutations** (0.0623) - Mutations in epitope site B
3. **site_D_mutations** (0.0521) - Mutations in epitope site D
4. **mean_hydrophobicity** (0.0489) - Average hydrophobicity
5. **site_C_mutations** (0.0445) - Mutations in epitope site C
6. **aa_K** (0.0398) - Lysine composition
7. **site_E_mutations** (0.0387) - Mutations in epitope site E
8. **aa_N** (0.0356) - Asparagine composition
9. **positive_charge_fraction** (0.0334) - Fraction of positively charged residues
10. **site_A_mutations** (0.0312) - Mutations in epitope site A

#### 3.2.2 Property Group Importance
1. **Epitope Sites:** Mean importance 0.0421 (highest)
2. **Amino Acid Composition:** Mean importance 0.0134
3. **Charge Properties:** Mean importance 0.0156
4. **Hydrophobicity:** Mean importance 0.0198
5. **Sequence Properties:** Mean importance 0.0089

**Key Finding:** Epitope site mutations adalah prediktor terkuat untuk perubahan temporal, mengkonfirmasi peran antigenic drift.

### 3.3 Cross-Validation Results

#### 3.3.1 Binary Model (5-fold CV)
- **Accuracy:** 99.47% (±0.31%)
- **Precision:** 99.48% (±0.33%)
- **Recall:** 99.58% (±0.29%)
- **F1-Score:** 99.53% (±0.30%)

#### 3.3.2 Multi-class Model (5-fold CV)
- **Accuracy:** 93.12% (±1.24%)
- **Precision:** 93.15% (±1.28%)
- **Recall:** 93.12% (±1.24%)
- **F1-Score:** 93.11% (±1.26%)

**Interpretasi:** Model stabil dengan variance rendah, menunjukkan generalization yang baik.

---

## 4. Diskusi

### 4.1 Interpretasi Hasil

#### 4.1.1 Performa Model
- Binary model mencapai akurasi near-perfect (99.55%), menunjukkan perbedaan jelas antara strain recent dan historical
- Multi-class model mencapai akurasi excellent (93.48%), mampu membedakan 4 periode temporal
- Cross-validation menunjukkan model robust dan tidak overfit

#### 4.1.2 Feature Importance
- **Epitope site mutations** dominan sebagai prediktor, konsisten dengan teori antigenic drift
- **Site B dan D** paling penting, sesuai literatur bahwa sites ini adalah target utama antibodi
- **Hydrophobicity dan charge** juga berperan, menunjukkan perubahan sifat fisikokimia mempengaruhi antigenicity

#### 4.1.3 Implikasi Biologis
1. **Antigenic Drift Detection:** Model dapat mendeteksi akumulasi mutasi yang mengindikasikan drift
2. **Vaccine Strain Selection:** Prediksi temporal dapat membantu identifikasi strain yang perlu diupdate
3. **Surveillance:** Tools ini dapat diintegrasikan dalam sistem surveillance untuk early warning

### 4.2 Kelebihan Penelitian
1. **Dataset Komprehensif:** 2,818 sekuens dengan quality scoring
2. **Feature Engineering:** 74 features mencakup komposisi, fisikokimia, dan epitope sites
3. **Multiple Models:** Binary dan multi-class untuk different use cases
4. **Validation Rigorous:** Cross-validation, ROC, PR curves, learning curves
5. **Interpretability:** Feature importance analysis untuk biological insights
6. **Reproducibility:** Complete pipeline dengan dokumentasi

### 4.3 Keterbatasan
1. **Temporal Bias:** Lebih banyak data recent (2020-2024) dibanding historical
2. **Geographic Bias:** Dominasi strain dari negara tertentu (USA, China)
3. **Antigenic Assay:** Tidak menggunakan data HI/neutralization assay (hanya sequence-based)
4. **Generalization:** Model trained pada H3N2, belum tested pada subtipe lain

### 4.4 Penelitian Lanjutan
1. **Integration dengan HI Data:** Combine sequence features dengan antigenic assay
2. **Deep Learning:** Explore CNN/RNN untuk sequence patterns
3. **Real-time Prediction:** Deploy sebagai web service untuk surveillance
4. **Multi-subtipe:** Extend ke H1N1, B/Victoria, B/Yamagata
5. **Structural Features:** Incorporate 3D structure information

---

## 5. Kesimpulan

1. **Model machine learning dapat memprediksi periode temporal strain H3N2 dengan akurasi tinggi** (99.55% binary, 93.48% multi-class)

2. **Epitope site mutations adalah prediktor terkuat**, mengkonfirmasi peran antigenic drift dalam evolusi H3N2

3. **Perubahan sifat fisikokimia (hydrophobicity, charge) berkorelasi dengan perubahan temporal**, menunjukkan selection pressure pada protein HA

4. **Tools prediksi ini dapat mendukung surveillance epidemiologi** dan vaccine strain selection

5. **Pipeline end-to-end telah dikembangkan**, dari data collection hingga prediction interface, siap untuk deployment

---

## 6. Referensi

1. Koel, B. F., et al. (2013). Substitutions near the receptor binding site determine major antigenic change during influenza virus evolution. *Science*, 342(6161), 976-979.

2. Smith, D. J., et al. (2004). Mapping the antigenic and genetic evolution of influenza virus. *Science*, 305(5682), 371-376.

3. Bedford, T., et al. (2014). Integrating influenza antigenic dynamics with molecular evolution. *eLife*, 3, e01914.

4. WHO. (2024). Recommended composition of influenza virus vaccines for use in the 2024-2025 northern hemisphere influenza season.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

---

**Dokumen ini adalah bagian dari laporan PKM-RE 2026**  
**Tim: Syifa Zavira Ramadhani & Rofi Perdana**  
**Universitas Brawijaya**
