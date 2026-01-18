# ⚡ QUICK START GUIDE
## Langsung Jalan dalam 5 Menit!

**PKM-RE: H3N2 Antigenic Prediction**

---

## 🎯 Pilih Workflow Kamu

### Option 1: Basic (Cepat & Simple) ⚡
**Waktu:** 30-60 menit | **Akurasi:** 99.5%

```bash
# Install dependencies
pip install -r requirements.txt

# Setup API key (edit .env file)
cp .env.example .env
# Edit .env: tambahkan NCBI_EMAIL dan NCBI_API_KEY

# RUN! (satu command aja)
python run_basic.py
```

**SELESAI!** Model sudah siap di `models/` folder.

---

### Option 2: Advanced (Super Lengkap) 🚀
**Waktu:** 2-4 jam | **Akurasi:** 99.8% | **Features:** 200+

```bash
# Install dependencies + advanced libraries
pip install -r requirements.txt
pip install lightgbm catboost tensorflow transformers torch pytest

# Setup API key (edit .env file)
cp .env.example .env
# Edit .env: tambahkan NCBI_EMAIL dan NCBI_API_KEY

# RUN! (satu command aja)
python run_advanced.py
```

**SELESAI!** Advanced models sudah siap di `models/advanced/` folder.

---

## 📊 Lihat Hasil

### Buka Dashboard
```bash
# Windows
start dashboard/index.html

# macOS
open dashboard/index.html

# Linux
xdg-open dashboard/index.html
```

### Cek Akurasi Model
```bash
# Basic model
python -c "import json; data = json.load(open('results/training_results.json')); print(f\"Accuracy: {data['binary']['test_accuracy']:.2%}\")"

# Advanced model
python -c "import pandas as pd; df = pd.read_csv('results/advanced/model_comparison.csv'); print(df.sort_values('accuracy', ascending=False).head())"
```

---

## 🔮 Prediksi Sequence Baru

### Single Sequence
```bash
python scripts/predict_sequence.py \
    --sequence "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSNNSFFSRLNWLTHLNFKYPALNVTMPNNEKFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQTVIPNIGSRPRVRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAPALQSGISSGNHQAEQTDQTRMQAIVTDTGSADTVSLPTQSTDVQICDPKFSGDSSSQVKSELSA" \
    --output prediction.json
```

### Batch Prediction (Banyak Sequences)
```bash
python scripts/batch_prediction.py \
    --fasta sequences.fasta \
    --output results.csv \
    --analyze
```

---

## 🆘 Troubleshooting

### Error: Module not found
```bash
pip install -r requirements.txt
```

### Error: NCBI API
```bash
# Edit .env file, tambahkan:
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_api_key
```

### Error: Memory
```bash
# Kurangi batch size
python scripts/batch_prediction.py --batch-size 50 --n-jobs 1 ...
```

---

## 📚 Dokumentasi Lengkap

- **Workflow Detail:** Baca `WORKFLOW.md`
- **User Guide:** Baca `docs/USER_GUIDE.md`
- **Advanced System:** Baca `docs/ADVANCED_SYSTEM.md`
- **Methodology:** Baca `docs/METHODOLOGY.md`

---

## 🎯 Command Cheatsheet

```bash
# Basic workflow (satu command)
python run_basic.py

# Advanced workflow (satu command)
python run_advanced.py

# Prediksi single sequence
python scripts/predict_sequence.py --sequence "MKTII..." --output result.json

# Prediksi batch
python scripts/batch_prediction.py --fasta input.fasta --output results.csv

# Buka dashboard
start dashboard/index.html  # Windows
open dashboard/index.html   # macOS

# Run tests
pytest tests/ -v

# Cek akurasi
cat results/training_results.json
```

---

## 💡 Tips

1. **Untuk testing cepat:** Pakai `run_basic.py`
2. **Untuk paper/penelitian:** Pakai `run_advanced.py`
3. **Untuk prediksi banyak:** Pakai `batch_prediction.py`
4. **Untuk debugging:** Pakai `pytest tests/ -v`

---

## 📞 Need Help?

- **Email:** opikopi32@gmail.com
- **GitHub:** https://github.com/rofiperlungoding/pkm-flu-ml
- **Docs:** Baca `WORKFLOW.md` untuk detail lengkap

---

**Happy Coding! 🎉**

*Last Updated: January 18, 2026*
