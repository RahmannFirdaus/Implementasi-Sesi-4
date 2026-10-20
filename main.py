"""
========================================================================
STUDI KASUS MACHINE LEARNING: KLASIFIKASI KELAYAKAN PINJAMAN
(Loan Approval Classification System)
========================================================================
Berdasarkan materi webinar Dr. Taniza Tajudin
Universiti Teknologi MARA & Universitas Bengkulu

Workflow:
  1. Data Collection & Synthesis
  2. Data Preprocessing (Missing Values, Encoding, Scaling)
  3. Data Splitting (80% Train / 20% Test)
  4. Model Training (Random Forest Classifier)
  5. Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')          # backend non-interaktif agar aman di server
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Konfigurasi plot ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================================
# 1. DATA COLLECTION & SYNTHESIS
# ====================================================================
def create_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Membuat dataset sintetis calon peminjam."""
    np.random.seed(seed)

    data = pd.DataFrame({
        'Income': np.random.randint(2000, 15000, n_samples),
        'Debt_Ratio': np.round(np.random.uniform(0.1, 0.8, n_samples), 2),
        'Credit_History': np.random.choice(
            ['Good', 'Fair', 'Poor'], n_samples, p=[0.5, 0.3, 0.2]
        ),
        'Employment_Status': np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed'], n_samples, p=[0.7, 0.2, 0.1]
        ),
        'Loan_Amount': np.random.randint(5000, 50000, n_samples),
    })

    def determine_status(row):
        score = 0
        if row['Credit_History'] == 'Good':
            score += 3
        elif row['Credit_History'] == 'Fair':
            score += 1

        if row['Employment_Status'] == 'Employed':
            score += 2
        elif row['Employment_Status'] == 'Self-Employed':
            score += 1

        if row['Debt_Ratio'] < 0.4:
            score += 2
        if row['Income'] > 5000:
            score += 2
        if row['Loan_Amount'] < 25000:
            score += 1

        return 'Approved' if score >= 5 else 'Rejected'

    data['Loan_Status'] = data.apply(determine_status, axis=1)
    return data


# ====================================================================
# 2. DATA PREPROCESSING
# ====================================================================
def preprocess(df: pd.DataFrame):
    """
    Menangani missing values, melakukan encoding kategorikal,
    dan normalisasi skala fitur (StandardScaler).
    Mengembalikan (X_scaled, y, feature_names, label_encoder_target).
    """
    data = df.copy()

    # Isi missing values numerik dengan median
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Label Encoding untuk fitur kategorikal & target
    le_credit = LabelEncoder()
    le_emp = LabelEncoder()
    le_target = LabelEncoder()

    data['Credit_History'] = le_credit.fit_transform(data['Credit_History'])
    data['Employment_Status'] = le_emp.fit_transform(data['Employment_Status'])
    data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])

    feature_cols = [c for c in data.columns if c != 'Loan_Status']
    X = data[feature_cols]
    y = data['Loan_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, le_target


# ====================================================================
# 3. DATA SPLITTING
# ====================================================================
def split_data(X, y, test_size=0.2, seed=42):
    """Membagi data: 80 % training, 20 % testing (stratified)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


# ====================================================================
# 4. MODEL TRAINING
# ====================================================================
def train_model(X_train, y_train, n_estimators=100, seed=42):
    """Melatih Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model


# ====================================================================
# 5. MODEL EVALUATION
# ====================================================================
def evaluate_model(model, X_test, y_test, label_encoder, feature_names):
    """Menghitung metrik evaluasi dan menyimpan visualisasi."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )

    # ── Print ke terminal ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HASIL EVALUASI MODEL (METRIK KINERJA)")
    print("=" * 60)
    print(f"  - Accuracy  : {acc:.4f}")
    print(f"  - Precision : {prec:.4f}")
    print(f"  - Recall    : {rec:.4f}")
    print(f"  - F1-Score  : {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(cm)
    print("\n  Classification Report:")
    print(report)

    # ── Visualisasi 1: Confusion Matrix Heatmap ────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        linewidths=0.8, linecolor='white', ax=ax1
    )
    ax1.set_xlabel('Prediksi', fontweight='bold')
    ax1.set_ylabel('Aktual', fontweight='bold')
    ax1.set_title('Confusion Matrix – Loan Approval', fontweight='bold')
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close(fig1)

    # ── Visualisasi 2: Feature Importance ──────────────────────────
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("viridis", len(feature_names))
    ax2.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=[colors[i] for i in idx],
        edgecolor='white', linewidth=0.6
    )
    ax2.set_xlabel('Importance', fontweight='bold')
    ax2.set_title('Feature Importance – Random Forest', fontweight='bold')
    ax2.invert_yaxis()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close(fig2)

    # ── Visualisasi 3: Ringkasan Metrik (Bar Chart) ────────────────
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bars = ax3.bar(
        metrics.keys(), metrics.values(),
        color=sns.color_palette("coolwarm", 4),
        edgecolor='white', linewidth=0.8
    )
    for bar, val in zip(bars, metrics.values()):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2%}", ha='center', fontweight='bold', fontsize=11
        )
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel('Skor', fontweight='bold')
    ax3.set_title('Ringkasan Metrik Evaluasi Model', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'metrics_summary.png'), dpi=150)
    plt.close(fig3)

    print(f"\n[OK] Visualisasi disimpan di folder: {OUTPUT_DIR}/")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("  STUDI KASUS ML: KLASIFIKASI KELAYAKAN PINJAMAN")
    print("  (LOAN APPROVAL CLASSIFICATION SYSTEM)")
    print("=" * 60)

    # 1. Data Collection
    data = create_dataset()
    print("\n[DATA] 1. Sampel Dataset Awal:")
    print(data.head(10).to_string(index=False))
    print(f"\n   Distribusi Target:\n{data['Loan_Status'].value_counts().to_string()}")

    # Simpan dataset ke CSV
    csv_path = os.path.join(OUTPUT_DIR, 'loan_dataset.csv')
    data.to_csv(csv_path, index=False)
    print(f"   Dataset disimpan -> {csv_path}")

    # 2. Preprocessing
    X_scaled, y, feature_names, le_target = preprocess(data)

    # 3. Splitting
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    print(f"\n[SPLIT] 2. Data Splitting Selesai")
    print(f"   Train : {X_train.shape[0]} sampel")
    print(f"   Test  : {X_test.shape[0]} sampel")

    # 4. Training
    model = train_model(X_train, y_train)
    print("\n[MODEL] 3. Pelatihan Model Random Forest Classifier -- Berhasil!")

    # 5. Evaluation
    evaluate_model(model, X_test, y_test, le_target, feature_names)

    print("\n" + "=" * 60)
    print("  SELESAI — Terima kasih!")
    print("=" * 60)


if __name__ == "__main__":
    main()
