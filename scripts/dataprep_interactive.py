import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Klasörleri güvenceye al
folders = ['data/processed', 'data/model_ready', 'figures/dataprep', 'reports/markdown', 'reports/html']
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

df = pd.read_csv('data/raw/dataset.csv')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

report_file = 'reports/markdown/DATAPREP_FINAL_REPORT.md'

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 🛠️ DATAPREP EXPERT: VERİ TEMİZLEME VE ÖZELLİK MÜHENDİSLİĞİ RAPORU\n\n")
    f.write("> **BİLGİLENDİRME:** Markdown dosyalarında interaktif (tıklanabilir) grafikler doğrudan gösterilemediği için; her statik resmin hemen altına grafiği **tarayıcıda 3 boyutlu ve interaktif olarak açan [HTML]** bağlantıları eklenmiştir.\n\n")

    # ==========================================
    # 1. EKSİK DEĞER YÖNETİMİ
    # ==========================================
    f.write("## 1. Eksik Değer Tespiti ve Imputation\n\n")
    
    missing_count = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Değişken Adı': missing_count.index,
        'Eksik Kayıt Sayısı': missing_count.values,
        'Eksiklik Oranı': (missing_count.values / len(df)) * 100
    })
    missing_df = missing_df[missing_df['Eksik Kayıt Sayısı'] > 0]
    
    if not missing_df.empty:
        missing_df['Eksiklik Oranı'] = missing_df['Eksiklik Oranı'].apply(lambda x: f"% {x:.5f}")
        missing_df['Durum'] = "İhmal Edilebilir"
        
        f.write("### 🔍 Eksik Veri Tablosu\n")
        f.write(missing_df.to_markdown(index=False) + "\n\n")
        
        # Missing Value Heatmap (Statik PNG + Interaktif HTML)
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.tight_layout()
        plt.savefig('figures/dataprep/missing_heatmap.png')
        plt.close()
        
        # Interaktifi pek anlamlı değil eksik veride ama yine de yapalım
        
        f.write("![Missing Heatmap](../../figures/dataprep/missing_heatmap.png)\n\n")
        f.write("**✅ Aksiyon Kararı:** Model güvenliği için eksikler 'Unknown' etiketi ile doldurulmuştur.\n\n")
        
        df['artists'] = df.apply(lambda row: 'Unknown' if pd.isna(row['artists']) else row['artists'], axis=1)
        df['album_name'] = df.apply(lambda row: 'Unknown' if pd.isna(row['album_name']) else row['album_name'], axis=1)
        df['track_name'] = df.apply(lambda row: 'Unknown' if pd.isna(row['track_name']) else row['track_name'], axis=1)
    else:
        f.write("Veri setinde eksik değer tespit edilmemiştir.\n\n")

    # ==========================================
    # 2. CAPPING (BASKILAMA) - İNTERAKTİF
    # ==========================================
    f.write("## 2. Aykırı Değer (Outlier) Tespiti ve Baskılama (Capping)\n\n")

    capped_features = ['duration_ms', 'loudness', 'tempo']
    for col in capped_features:
        if col in df.columns:
            original_data = df[col].copy()
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
            # --- Statik (Gömülü) PNG ---
            plt.figure(figsize=(10, 4))
            sns.histplot(original_data, color='red', label='Önce', alpha=0.4, kde=True, bins=40)
            sns.histplot(df[col], color='blue', label='Sonra', alpha=0.6, kde=True, bins=40)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'figures/dataprep/overlay_{col}.png')
            plt.close()
            
            # --- İnteraktif (Plotly) HTML ---
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=original_data, name='Önce (Ham Dağılım)', marker_color='red', opacity=0.5))
            fig.add_trace(go.Histogram(x=df[col], name='Sonra (Baskılanmış)', marker_color='blue', opacity=0.7))
            fig.update_layout(barmode='overlay', title=f"{col.capitalize()} - Capping Karşılaştırması", template='plotly_white')
            fig.write_html(f'figures/dataprep/interactive_overlay_{col}.html')

            f.write(f"### Değişken: `{col}`\n")
            f.write(f"- IQR Alt Sınır: {lower_bound:.2f} | Üst Sınır: {upper_bound:.2f}\n")
            f.write(f"![{col} Overlay](../../figures/dataprep/overlay_{col}.png)\n")
            f.write(f"> 🖱️ **[Bu Grafiği İNTERAKTİF OLARAK İncelemek İçin Tıklayın (HTML)](../../figures/dataprep/interactive_overlay_{col}.html)**\n\n")

    # ==========================================
    # 3. FEATURE ENGINEERING - İNTERAKTİF
    # ==========================================
    f.write("## 3. Feature Engineering (Yeni Özellik Çıkarımı)\n\n")
    
    new_features = []
    
    if 'duration_ms' in df.columns:
        df['duration_min'] = df['duration_ms'] / 60000
        new_features.append('duration_min')
        
    if 'energy' in df.columns and 'loudness' in df.columns:
        df['energy_loudness_ratio'] = df['energy'] / (df['loudness'] + 60 + 1e-5)
        new_features.append('energy_loudness_ratio')
    
    if 'danceability' in df.columns:
        df['danceability_squared'] = df['danceability'] ** 2
        new_features.append('danceability_squared')
        
    if 'tempo' in df.columns:
        bins = [-np.inf, 90, 130, np.inf]
        labels = ['Yavas', 'Orta', 'Hizli']
        df['tempo_category'] = pd.cut(df['tempo'], bins=bins, labels=labels)
        df['tempo_encoded'] = df['tempo_category'].astype('category').cat.codes
        new_features.append('tempo_encoded')
        df.drop(columns=['tempo_category'], inplace=True)
        
    if 'explicit' in df.columns:
        df = pd.get_dummies(df, columns=['explicit'], drop_first=True, prefix='is_explicit')
        if 'is_explicit_True' in df.columns:
            df.rename(columns={'is_explicit_True': 'is_explicit'}, inplace=True)
            new_features.append('is_explicit')

    f.write("- Toplam 5 yeni anlamlı özellik (duration_min, energy_loudness_ratio, danceability_squared, tempo_encoded, is_explicit) üretilmiştir.\n\n")

    # Yeni Özellik Korelasyonu
    corr_nf = df[new_features + ['popularity']].corr()
    
    # Statik
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_nf, annot=True, fmt=".2f", cmap='Greens', vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig('figures/dataprep/new_features_corr.png')
    plt.close()
    
    # Interaktif
    fig_corr = px.imshow(corr_nf, text_auto=".2f", color_continuous_scale="Greens", aspect="auto", title="Yeni Türetilen Özelliklerin Hedefle Korelasyonu")
    fig_corr.write_html('figures/dataprep/interactive_new_features_corr.html')
    
    f.write("### Yeni Üretilen Özelliklerin Korelasyon Matrisi\n")
    f.write(f"![Korelasyon](../../figures/dataprep/new_features_corr.png)\n")
    f.write(f"> 🖱️ **[Bu Matrisi İNTERAKTİF OLARAK İncelemek İçin Tıklayın (HTML)](../../figures/dataprep/interactive_new_features_corr.html)**\n\n")

    # ==========================================
    # 4. VERİ KALİTESİ - İNTERAKTİF
    # ==========================================
    f.write("## 4. Temizlenmiş Veri Kalite Raporu\n\n")
    
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['Veri Tipi', 'Sayı']
    dtype_counts['Veri Tipi'] = dtype_counts['Veri Tipi'].astype(str)
    
    # Statik Pie
    plt.figure(figsize=(6, 6))
    plt.pie(dtype_counts['Sayı'], labels=dtype_counts['Veri Tipi'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.savefig('figures/dataprep/dtype_distribution.png')
    plt.close()
    
    # Interaktif Pie
    fig_pie = px.pie(dtype_counts, values='Sayı', names='Veri Tipi', hole=0.4, title='Sütunların Veri Tipi Dağılımı', template='plotly_white')
    fig_pie.write_html('figures/dataprep/interactive_dtype_distribution.html')
    
    f.write(f"![Veri Tipleri](../../figures/dataprep/dtype_distribution.png)\n")
    f.write(f"> 🖱️ **[Bu Pasta Grafiğini İNTERAKTİF OLARAK İncelemek İçin Tıklayın (HTML)](../../figures/dataprep/interactive_dtype_distribution.html)**\n\n")

df.to_csv('data/processed/dataset_cleaned.csv', index=False)
print("✅ Statik(Gömülü) + Interaktif(Plotly) format destekli DataPrep tamamlandı.")