# 📊 Dashboard Guide - Super Lengkap!
## PKM-RE: H3N2 Antigenic Prediction Dashboard

**Dibuat untuk:** Orang yang "planga-plongo" dan butuh penjelasan super jelas! 😊

---

## 🎯 Apa itu Dashboard Ini?

Dashboard ini adalah **tampilan visual interaktif** yang menjelaskan seluruh sistem machine learning untuk prediksi virus H3N2. 

**Analogi sederhana:** Bayangkan dashboard ini seperti **buku cerita bergambar** yang menjelaskan bagaimana komputer belajar mengenali virus baru vs lama!

---

## 🚀 Cara Membuka Dashboard

### Windows:
```bash
# Cara 1: Double-click
Buka folder "dashboard" → Double-click "dashboard_v2.html"

# Cara 2: Command line
start dashboard/dashboard_v2.html
```

### Mac:
```bash
open dashboard/dashboard_v2.html
```

### Linux:
```bash
xdg-open dashboard/dashboard_v2.html
```

---

## 📚 Penjelasan Setiap Tab

### Tab 1: 🏠 Pengenalan

**Apa yang ada di sini?**
- Penjelasan sederhana tentang sistem ini
- Kenapa sistem ini penting?
- Siapa tim yang buat?

**Penjelasan untuk yang "planga-plongo":**
- **"Apa itu sistem ini?"** → Program komputer pintar yang bisa bilang virus itu baru atau lama
- **"Kenapa penting?"** → Bantu bikin vaksin yang tepat
- **"Akurasi 99.55%?"** → Dari 100 prediksi, 99-100 nya benar!

**Analogi:**
Bayangkan kamu punya teman yang bisa lihat foto virus dan langsung bilang "ini virus tahun 2020-an!" atau "ini virus tahun 2010-an!" dengan hampir selalu benar. Itulah sistem ini!

---

### Tab 2: 📊 Alur Kerja

**Apa yang ada di sini?**
- Alur lengkap dari awal sampai akhir
- Step-by-step dijelaskan dengan detail
- Berapa lama setiap step?

**4 Step Utama:**

#### Step 1: Data Collection (Pengumpulan Data)
**Apa yang dilakukan?**
- Download 2,818 urutan protein virus dari internet
- Sumber: Database NCBI (database virus terbesar di dunia)
- Filter: Ambil yang bagus aja, buang yang jelek

**Analogi:**
Seperti kamu ngumpulin 2,818 foto virus dari Google, tapi yang HD aja, yang blur dibuang!

**Waktu:** ~30-60 menit

#### Step 2: Feature Extraction (Ekstraksi Ciri)
**Apa yang dilakukan?**
- Hitung 74 ciri khas dari setiap virus
- Ciri-ciri: komposisi asam amino, sifat fisik, dll

**Analogi:**
Seperti kamu ukur tinggi, berat, warna mata, warna rambut dari setiap orang. Tapi untuk virus!

**Waktu:** ~2-5 menit

#### Step 3: Model Training (Melatih Model)
**Apa yang dilakukan?**
- Latih komputer untuk mengenali pola
- Bagi data: 80% untuk belajar, 20% untuk ujian
- Algoritma: XGBoost (salah satu yang terbaik)

**Analogi:**
Seperti kamu ajarin anak kecil bedain kucing vs anjing dengan kasih lihat 2,000+ foto. Setelah belajar, dia bisa langsung bilang "ini kucing!" atau "ini anjing!"

**Waktu:** ~5-10 menit

#### Step 4: Prediction (Prediksi)
**Apa yang dilakukan?**
- Kasih virus baru → Model prediksi → Hasilnya keluar!
- Akurasi: 99.55%

**Analogi:**
Sekarang komputer udah pinter, kasih foto virus baru, dia langsung bisa bilang "ini virus baru!" dengan confidence 99.8%!

**Waktu:** <1 detik per virus

---

### Tab 3: 💾 Data

**Apa yang ada di sini?**
- Statistik dataset lengkap
- Distribusi data per periode
- Sumber data

**Angka-angka Penting:**
- **Total:** 2,818 urutan protein
- **Tahun:** 1996-2024 (29 tahun!)
- **Kualitas tinggi:** 2,204 (78%)
- **Virus terbaru (≥2020):** 1,455 (52%)

**Chart yang ada:**
- **Bar chart:** Distribusi data per periode
  - Period 1 (2009-2013): 303 samples
  - Period 2 (2014-2016): 858 samples
  - Period 3 (2017-2019): 356 samples
  - Period 4 (≥2020): 708 samples

**Penjelasan chart:**
Semakin tinggi bar-nya, semakin banyak data dari periode itu. Period 2 paling banyak karena banyak penelitian di tahun 2014-2016!

---

### Tab 4: 🔬 Features

**Apa yang ada di sini?**
- Penjelasan 74 features (ciri-ciri virus)
- Kategori features
- Feature importance (yang paling penting)

**74 Features dibagi jadi 3 kategori:**

#### 1. Amino Acid Composition (20 features)
**Apa ini?**
Menghitung berapa banyak setiap jenis asam amino.

**Analogi:**
Protein itu seperti kata yang terdiri dari 20 huruf (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). 
Kita hitung: berapa banyak huruf A? berapa banyak huruf C? dst.

**Contoh:**
- Alanine (A): 10%
- Glycine (G): 5%
- Leucine (L): 8%
- dst...

**Kenapa penting?**
Virus baru dan lama punya komposisi yang beda! Seperti bahasa Inggris vs Indonesia punya frekuensi huruf yang beda.

#### 2. Physicochemical Properties (30+ features)
**Apa ini?**
Sifat fisik dan kimia dari protein.

**Yang dihitung:**
- **Hydrophobicity:** Suka air atau minyak?
- **Charge:** Bermuatan + atau -?
- **Polarity:** Polar atau non-polar?
- **Molecular Weight:** Berat molekul

**Analogi:**
Seperti kamu ukur: apakah benda ini berat/ringan? keras/lembek? panas/dingin? basah/kering?

**Kenapa penting?**
Sifat ini menentukan bagaimana virus berinteraksi dengan sel kita!

#### 3. Epitope Site Analysis (24 features)
**Apa ini?**
Analisis bagian protein yang dikenali sistem imun.

**5 Epitope Sites:** A, B, C, D, E

**Analogi:**
Epitope sites itu seperti "wajah" virus yang dilihat oleh antibodi kita. Kalau wajahnya berubah (mutasi), antibodi lama tidak bisa mengenali → butuh vaksin baru!

**Yang dihitung:**
- Berapa banyak mutasi di setiap site?
- Apakah mutasi mengubah sifat?

**Kenapa SUPER penting?**
Ini adalah bagian yang paling menentukan apakah vaksin lama masih efektif atau tidak!

**Chart Feature Importance:**
Chart ini menunjukkan features mana yang paling berpengaruh. Top 3:
1. **Epitope Site A Mutations** (18%) - Paling penting!
2. **Hydrophobicity Mean** (15%)
3. **RBD Hydrophobicity** (12%)

---

### Tab 5: 🤖 Models

**Apa yang ada di sini?**
- Penjelasan tentang machine learning model
- Kenapa pakai XGBoost?
- Proses training
- 2 model yang dilatih

**Apa itu Machine Learning Model?**
Model ML adalah **"otak komputer"** yang sudah dilatih untuk mengenali pola.

**Analogi:**
Seperti kamu punya teman yang sudah lihat 2,000+ foto kucing dan anjing. Sekarang dia bisa langsung bilang "ini kucing!" atau "ini anjing!" hanya dengan sekilas lihat. Itulah model ML!

**Kenapa XGBoost?**
- ✅ Akurasi tinggi (salah satu yang terbaik)
- ✅ Cepat (training dan prediksi)
- ✅ Tidak mudah "ngapal" (robust)
- ✅ Bisa lihat feature importance

**2 Model yang Dilatih:**

#### Model 1: Binary Classification
**Tugas:** Jawab "Baru atau Lama?"
**Akurasi:** 99.55%
**ROC-AUC:** 100% (perfect!)

**Penjelasan:**
Model ini hanya jawab 2 kemungkinan:
- Recent (≥2020) → Virus baru
- Historical (<2020) → Virus lama

Karena hanya 2 pilihan, lebih mudah → akurasi lebih tinggi!

#### Model 2: Multi-class Classification
**Tugas:** Jawab "Periode mana?"
**Akurasi:** 93.48%

**Penjelasan:**
Model ini jawab 4 kemungkinan:
- Period 1: 2009-2013
- Period 2: 2014-2016
- Period 3: 2017-2019
- Period 4: ≥2020

Karena 4 pilihan, lebih sulit → akurasi sedikit lebih rendah (tapi masih sangat bagus!)

**Analogi:**
Lebih mudah bilang "ini buah atau sayur?" (2 pilihan) daripada "ini apel, jeruk, pisang, atau mangga?" (4 pilihan).

**Proses Training (4 Step):**
1. **Split Data:** 80% training, 20% testing
2. **Train Model:** Model belajar dari 80% data
3. **Cross-Validation:** Validasi 5x untuk memastikan tidak "ngapal"
4. **Test:** Ujian akhir dengan 20% data → 99.55%!

---

### Tab 6: 📈 Hasil

**Apa yang ada di sini?**
- Performance metrics lengkap
- Confusion matrix
- Kesimpulan

**Binary Model Performance:**
- **Test Accuracy:** 99.55% (dari 564 test samples)
- **CV Accuracy:** 99.42% ±0.84% (konsisten!)
- **ROC-AUC:** 100% (perfect score!)
- **F1-Score:** 99.55%

**Apa arti metrik-metrik ini?**

**Accuracy:**
Berapa persen prediksi yang benar?
- 99.55% = dari 564 prediksi, 561 benar, 3 salah
- Analogi: Dari 100 soal ujian, kamu bener 99-100 soal!

**ROC-AUC:**
Seberapa baik model membedakan kelas?
- 100% = perfect! Model bisa bedain dengan sempurna
- Analogi: Kamu bisa bedain kucing vs anjing dengan mata tertutup!

**F1-Score:**
Keseimbangan antara precision dan recall
- 99.55% = sangat seimbang, tidak bias ke satu kelas
- Analogi: Kamu tidak cuma jago bedain kucing, tapi juga jago bedain anjing!

**Confusion Matrix:**
Chart yang menunjukkan berapa banyak prediksi benar dan salah.

**Cara baca:**
- **Diagonal (kiri atas ke kanan bawah):** Prediksi BENAR ✅
- **Off-diagonal:** Prediksi SALAH ❌

Semakin banyak di diagonal, semakin bagus!

**Contoh:**
```
                Predicted
              Historical  Recent
Actual
Historical      420        2      ← 420 benar, 2 salah
Recent           1        141     ← 141 benar, 1 salah
```

Total benar: 420 + 141 = 561
Total salah: 2 + 1 = 3
Accuracy: 561/564 = 99.55%!

---

### Tab 7: 🔴 Live Monitor

**Apa yang ada di sini?**
- Live training monitor (demo)
- Progress bar
- Training log real-time
- Metrics yang update terus

**Cara pakai:**
1. Klik tombol **"🚀 Start Training (Demo)"**
2. Lihat progress bar naik
3. Lihat log yang muncul real-time
4. Lihat metrics (epoch, accuracy, time) yang update

**Apa yang ditampilkan:**

**Progress Bar:**
Menunjukkan berapa persen training sudah selesai (0-100%)

**Metrics:**
- **Current Epoch:** Iterasi ke berapa (0-200)
- **Training Accuracy:** Akurasi di training set
- **Validation Accuracy:** Akurasi di validation set
- **Time Elapsed:** Waktu yang sudah lewat

**Training Log:**
Log yang menunjukkan apa yang sedang dilakukan:
```
[19:30:00] 🚀 Starting training process...
[19:30:00] 📊 Loading dataset: 2,818 sequences
[19:30:00] ✂️ Splitting data: 80% train, 20% test
[19:30:02] ✅ Model initialized successfully
[19:30:02] 🎓 Starting training loop...
[19:30:03] 📈 Epoch 10/200 - Train Acc: 85.23% - Val Acc: 84.12%
[19:30:04] 📈 Epoch 20/200 - Train Acc: 91.45% - Val Acc: 90.87%
...
[19:30:20] 🎉 Training completed!
[19:30:20] 💾 Saving model to disk...
[19:30:20] ✅ Model saved: h3n2_binary_model.pkl
[19:30:20] 📊 Final Test Accuracy: 99.55%
```

**Penjelasan Log:**
- **Loading dataset:** Memuat data ke memory
- **Splitting data:** Membagi jadi training dan testing
- **Training epoch X:** Sedang belajar (iterasi ke-X)
- **Validation:** Mengecek akurasi
- **Saving model:** Menyimpan model yang sudah dilatih

**Note:** Ini adalah **demo/simulasi**. Training asli memakan waktu 5-10 menit, tapi di demo ini dipercepat jadi ~20 detik untuk demonstrasi!

---

## 🎨 Fitur-fitur Dashboard

### 1. **Responsive Design**
Dashboard otomatis menyesuaikan dengan ukuran layar (desktop, tablet, mobile)

### 2. **Interactive Charts**
Semua chart bisa di-hover untuk lihat detail

### 3. **Info Boxes**
Kotak biru dengan penjelasan tambahan di setiap section

### 4. **Tooltips**
Hover di teks yang berwarna ungu untuk lihat penjelasan

### 5. **Beautiful Gradient**
Design modern dengan gradient ungu-pink yang eye-catching

### 6. **Tab Navigation**
7 tab yang mudah di-navigate

### 7. **Live Monitor**
Demo training real-time dengan progress bar dan log

---

## 💡 Tips Menggunakan Dashboard

### Untuk Presentasi PKM-RE:

**1. Mulai dari Tab "Pengenalan"**
- Jelaskan apa itu sistem ini dengan analogi sederhana
- Tunjukkan akurasi 99.55%
- Jelaskan kenapa penting

**2. Lanjut ke Tab "Alur Kerja"**
- Tunjukkan 4 step utama
- Jelaskan setiap step dengan detail
- Tunjukkan berapa lama setiap step

**3. Tab "Data"**
- Tunjukkan chart distribusi data
- Jelaskan 2,818 sequences dari 29 tahun
- Tunjukkan sumber data (NCBI + WHO)

**4. Tab "Features"**
- Jelaskan 74 features dengan analogi
- Tunjukkan feature importance chart
- Fokus ke epitope sites (paling penting!)

**5. Tab "Models"**
- Jelaskan XGBoost dengan analogi
- Tunjukkan 2 model (binary 99.55%, multi-class 93.48%)
- Jelaskan proses training

**6. Tab "Hasil"**
- Tunjukkan confusion matrix
- Jelaskan metrics (accuracy, ROC-AUC, F1)
- Kesimpulan: Model sangat baik!

**7. Tab "Live Monitor" (DEMO!)**
- Klik "Start Training"
- Tunjukkan progress bar dan log real-time
- Jelaskan apa yang sedang terjadi
- **WOW FACTOR!** 🎉

### Untuk Orang Awam:

**Fokus ke:**
- Analogi-analogi sederhana
- Info boxes (kotak biru)
- Chart yang visual
- Live monitor (paling menarik!)

**Hindari:**
- Istilah teknis yang terlalu detail
- Angka-angka yang terlalu banyak
- Penjelasan matematis

### Untuk Reviewer/Juri:

**Fokus ke:**
- Metodologi (Tab "Alur Kerja")
- Feature engineering (Tab "Features")
- Model performance (Tab "Hasil")
- Confusion matrix dan metrics

**Highlight:**
- Akurasi 99.55% (sangat tinggi!)
- ROC-AUC 100% (perfect!)
- 2,818 sequences (dataset besar!)
- 74 features (comprehensive!)

---

## 🚀 Keunggulan Dashboard Ini

### 1. **Super Jelas**
Semua dijelaskan dengan bahasa sederhana + analogi

### 2. **Visual Menarik**
Design modern dengan gradient dan chart interaktif

### 3. **Lengkap**
7 tab yang cover semua aspek dari awal sampai akhir

### 4. **Live Monitor**
Demo training real-time (WOW factor!)

### 5. **Responsive**
Bisa dibuka di desktop, tablet, atau mobile

### 6. **No Installation**
Cukup buka file HTML di browser, tidak perlu install apa-apa

### 7. **Offline**
Bisa dibuka tanpa internet (semua sudah di file HTML)

---

## 📝 Catatan Penting

### Untuk Demo/Presentasi:

**1. Persiapan:**
- Buka dashboard sebelum presentasi
- Test semua tab
- Test live monitor
- Pastikan browser support Chart.js

**2. Saat Presentasi:**
- Mulai dari tab "Pengenalan"
- Jelaskan dengan analogi
- Tunjukkan chart
- Demo live monitor di akhir (WOW!)

**3. Q&A:**
- Gunakan dashboard untuk jawab pertanyaan
- Tunjukkan tab yang relevan
- Gunakan chart untuk visualisasi

### Browser Support:

Dashboard ini support:
- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Edge
- ✅ Safari
- ⚠️ Internet Explorer (not recommended)

---

## 🎓 Untuk PKM-RE

Dashboard ini **PERFECT** untuk:
- ✅ Presentasi proposal
- ✅ Presentasi hasil
- ✅ Demo ke reviewer
- ✅ Dokumentasi visual
- ✅ Teaching tool

**Keunggulan untuk PKM-RE:**
- Mudah dipahami (bahkan untuk non-teknis)
- Visual menarik (eye-catching)
- Comprehensive (semua ada)
- Interactive (bisa di-demo)
- Professional (design bagus)

---

## 🎉 Kesimpulan

Dashboard ini dibuat khusus untuk orang yang **"planga-plongo"** dan butuh penjelasan super jelas!

**Fitur utama:**
- 7 tab lengkap
- Penjelasan dengan analogi
- Chart interaktif
- Live training monitor
- Design modern

**Perfect untuk:**
- Presentasi PKM-RE
- Demo ke reviewer
- Teaching tool
- Dokumentasi

**Cara pakai:**
1. Buka `dashboard/dashboard_v2.html`
2. Explore 7 tab
3. Klik "Start Training" di tab Live Monitor
4. Enjoy! 🎉

---

**Dibuat dengan ❤️ oleh PKM-RE Team**  
**Syifa Zavira Ramadhani & Rofi Perdana**  
**Universitas Brawijaya**

**Last Updated:** January 18, 2026
