# 🏦 Studi Kasus Machine Learning — Klasifikasi Kelayakan Pinjaman

> **Loan Approval Classification System**
> Berdasarkan materi webinar **Dr. Taniza Tajudin** — *Universiti Teknologi MARA* bersama *Universitas Bengkulu*

---

## 📋 Deskripsi

Proyek ini mengimplementasikan **klasifikasi biner** untuk memprediksi kelayakan pinjaman bank (*Approved* / *Rejected*) menggunakan **Random Forest Classifier** dari library *scikit-learn*.

### Fitur (Atribut) yang Digunakan

| Fitur               | Tipe       | Keterangan                                   |
|----------------------|------------|----------------------------------------------|
| `Income`            | Numerik    | Pendapatan calon peminjam (2 000 – 15 000)   |
| `Debt_Ratio`        | Numerik    | Rasio utang terhadap pendapatan (0.1 – 0.8)  |
| `Credit_History`    | Kategorikal| Good / Fair / Poor                           |
| `Employment_Status` | Kategorikal| Employed / Self-Employed / Unemployed        |
| `Loan_Amount`       | Numerik    | Jumlah pinjaman yang diminta (5 000 – 50 000)|

### Target

`Loan_Status` → *Approved* atau *Rejected*

---

## 🔄 Alur Kerja (*Machine Learning Workflow*)

```
┌──────────────────┐
│ 1. Data Collection│
│    & Synthesis    │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ 2. Data           │
│    Preprocessing  │
│  (Encoding,       │
│   Scaling,        │
│   Missing Values) │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ 3. Data Splitting │
│  80% Train        │
│  20% Test         │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ 4. Model Training │
│  Random Forest    │
│  (100 trees)      │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ 5. Model          │
│    Evaluation     │
│  Accuracy,        │
│  Precision,       │
│  Recall, F1,      │
│  Confusion Matrix │
└──────────────────┘
```

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.9+

### Langkah

```bash
# 1. Clone repositori
git clone https://github.com/<USERNAME>/ml-loan-approval-project.git
cd ml-loan-approval-project

# 2. Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan program
python main.py
```

Hasil evaluasi akan ditampilkan di terminal, dan tiga file visualisasi akan disimpan di folder `output/`:

| File                      | Deskripsi                              |
|---------------------------|----------------------------------------|
| `confusion_matrix.png`   | Heatmap Confusion Matrix               |
| `feature_importance.png` | Bar chart tingkat kepentingan fitur     |
| `metrics_summary.png`    | Ringkasan metrik (Acc, Prec, Rec, F1)  |
| `loan_dataset.csv`       | Dataset sintetis yang dihasilkan        |

---

## 📂 Struktur Proyek

```
ml-loan-approval-project/
├── main.py              # Skrip utama ML
├── requirements.txt     # Daftar dependensi Python
├── .gitignore           # File yang diabaikan Git
├── LICENSE              # Lisensi MIT
├── README.md            # Dokumentasi (file ini)
└── output/              # Hasil keluaran (dibuat saat runtime)
    ├── loan_dataset.csv
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── metrics_summary.png
```

---

## 📤 Panduan Mengunggah ke GitHub

### A. Melalui Terminal / Command Line

```bash
# 1. Inisialisasi repositori Git
cd ml-loan-approval-project
git init

# 2. Tambahkan semua file
git add .

# 3. Buat commit pertama
git commit -m "feat: initial ML loan approval classification project"

# 4. Buat repositori baru di GitHub (https://github.com/new)
#    — Beri nama: ml-loan-approval-project
#    — Jangan centang "Initialize with README"

# 5. Hubungkan ke remote repository
git remote add origin https://github.com/<USERNAME>/ml-loan-approval-project.git

# 6. Push ke GitHub
git branch -M main
git push -u origin main
```

### B. Melalui GitHub Desktop

1. Buka **GitHub Desktop** → *File* → *Add Local Repository*
2. Pilih folder `ml-loan-approval-project`
3. Klik **Create Repository**
4. Klik **Publish Repository** → pilih visibilitas (Public/Private) → **Publish**

### C. Melalui Web (Upload)

1. Buka https://github.com/new → buat repositori baru
2. Klik **"uploading an existing file"**
3. Seret semua file proyek ke area upload
4. Klik **Commit changes**

---

## 📚 Referensi

1. Materi Webinar Dr. Taniza Tajudin — *Universiti Teknologi MARA* & *Universitas Bengkulu*
2. Scikit-learn Documentation — https://scikit-learn.org/stable/
3. Pandas Documentation — https://pandas.pydata.org/docs/
4. Matplotlib & Seaborn — https://matplotlib.org/ | https://seaborn.pydata.org/

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
