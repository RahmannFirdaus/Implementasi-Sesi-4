# 🏦 Studi Kasus Machine Learning - Klasifikasi Kelayakan Pinjaman

> **Loan Approval Classification System**

---

## 📋 Deskripsi

Proyek ini mengimplementasikan **klasifikasi biner** untuk memprediksi kelayakan pinjaman bank (*Approved* / *Rejected*) menggunakan **Random Forest Classifier** dari library *scikit-learn*.

### Fitur (Atribut) yang Digunakan

| Fitur               | Tipe       | Keterangan                                   |
|----------------------|------------|----------------------------------------------|
| `Income`            | Numerik    | Pendapatan calon peminjam (2 000 - 15 000)   |
| `Debt_Ratio`        | Numerik    | Rasio utang terhadap pendapatan (0.1 - 0.8)  |
| `Credit_History`    | Kategorikal| Good / Fair / Poor                           |
| `Employment_Status` | Kategorikal| Employed / Self-Employed / Unemployed        |
| `Loan_Amount`       | Numerik    | Jumlah pinjaman yang diminta (5 000 - 50 000)|

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
└── output/              
    ├── loan_dataset.csv
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── metrics_summary.png
---

## 📚 Referensi

1. Materi Webinar Dr. Taniza Tajudin - *Universiti Teknologi MARA* & *Universitas Bengkulu*
2. Scikit-learn Documentation - https://scikit-learn.org/stable/
3. Pandas Documentation - https://pandas.pydata.org/docs/
4. Matplotlib & Seaborn - https://matplotlib.org/ | https://seaborn.pydata.org/
