import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Klasörleri oluştur
folders = ['figures', 'reports/markdown']
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Veriyi oku
df = pd.read_csv('data/raw/dataset.csv')

# Unnamed kolonunu gereksizse silelim
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

target = 'popularity'
# Sayısal ve Kategorik değişkenlerin ayrımı
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if target in num_cols:
    num_cols.remove(target)

with open('reports/markdown/EDA_FINAL_REPORT.md', 'w', encoding='utf-8') as f:
    f.write("# 📊 DETAYLI VE GÖRSEL EDA NİHAİ RAPORU\n\n")
    f.write("> **NOT:** Tüm figürler (grafikler) ve tablolar bu sunum dokümanı içerisine gömülmüştür.\n\n")

    # ----------------------------------------------------
    # PHASE 1: DATA OVERVIEW
    # ----------------------------------------------------
    f.write("## 🚀 PHASE 1: VERİYE GENEL BAKIŞ VE KALİTE (OVERVIEW)\n")
    f.write(f"- **Toplam Satır Sayısı:** {df.shape[0]}\n")
    f.write(f"- **Toplam Sütun Sayısı:** {df.shape[1]}\n\n")
    
    info_df = pd.DataFrame({
        'Veri_Tipi': df.dtypes,
        'Eksik_Gözlem_Sayısı': df.isnull().sum(),
        'Benzersiz_(Unique)_Değer': df.nunique()
    }).reset_index().rename(columns={'index': 'Değişken Mimarisi'})
    
    f.write("### Değişken Özeti ve Eksik Veri Tablosu\n")
    f.write(info_df.to_markdown(index=False) + "\n\n")
    f.write("**🧠 Bulgular:** Sadece `artists`, `album_name` ve `track_name` sütunlarında 1'er adet eksik boyut tespit edilmiştir. Kritik bir kalite sorunu yoktur.\n\n")

    # ----------------------------------------------------
    # PHASE 2: UNIVARIATE ANALYSIS (Tekli Değişken)
    # ----------------------------------------------------
    f.write("## 🚀 PHASE 2: TEKLİ DEĞİŞKEN (UNIVARIATE) DAĞILIMLARI\n\n")
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target], bins=50, kde=True, color='purple')
    plt.title('Hedef Değişken: Popularity Dağılımı')
    plt.tight_layout()
    plt.savefig('figures/target_dist.png')
    plt.close()
    
    f.write("### Hedef Değişken (Popularity)\n")
    f.write("![Popularity Dağılımı](../../figures/target_dist.png)\n")
    f.write("**💡 Yorum:** Popülarite değişkeni 0'da yoğunlaşmış büyük bir kitle (low-popularity outliers) barındırmakla birlikte genel olarak sağa çarpık ve yaygın bir dağılım göstermektedir.\n\n")

    # 2. Önemli Sayısal Özelliklerin Dağılımı
    important_nums = ['danceability', 'energy', 'loudness', 'tempo']
    f.write("### Kritik Ses Özellikleri (Audio Features) Dağılımları\n")
    
    for col in important_nums:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], bins=30, kde=True, color='teal')
            plt.title(f'{col} Dağılımı')
            plt.tight_layout()
            plt.savefig(f'figures/hist_{col}.png')
            plt.close()
            f.write(f"#### {col}\n")
            f.write(f"![{col} Dağılımı](../../figures/hist_{col}.png)\n\n")

    # ----------------------------------------------------
    # PHASE 3: BIVARIATE ANALYSIS (İkili Analiz - Hedef İlişkisi)
    # ----------------------------------------------------
    f.write("## 🚀 PHASE 3: HEDEF DEĞİŞKEN (POPULARITY) İLE İLİŞKİLER\n\n")
    f.write("Verinin hacmi yüksek (114k satır) olduğu için grafikler 3000 kayıtlık rastgele bir alt örneklem ile çizdirilmiştir.\n\n")
    
    sample_df = df.sample(3000, random_state=42)
    for col in important_nums:
        if col in sample_df.columns:
            plt.figure(figsize=(8, 4))
            sns.scatterplot(data=sample_df, x=col, y=target, alpha=0.4, color='darkorange')
            plt.title(f'{target} vs {col}')
            plt.tight_layout()
            plt.savefig(f'figures/scatter_{col}.png')
            plt.close()
            f.write(f"#### Popularity vs {col}\n")
            f.write(f"![{target} vs {col}](../../figures/scatter_{col}.png)\n\n")
            
    # ----------------------------------------------------
    # PHASE 4: MULTIVARIATE ANALYSIS (Çoklu Değişken Korelasyonu)
    # ----------------------------------------------------
    f.write("## 🚀 PHASE 4: ÇOKLU DEĞİŞKEN (MULTIVARIATE) ANALİZ VE KORELASYON\n\n")
    
    # Sadece sayısal + target
    corr_cols = num_cols.copy()
    if target not in corr_cols:
        corr_cols.append(target)
        
    plt.figure(figsize=(14, 10))
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Sayısal Değişkenler Korelasyon Isı Haritası')
    plt.tight_layout()
    plt.savefig('figures/corr_heatmap.png')
    plt.close()

    f.write("### Feature - Feature & Feature - Target Korelasyonları\n")
    f.write("![Korelasyon Haritası](../../figures/corr_heatmap.png)\n\n")

    f.write("#### Popularity (Hedef Değişken) İle En Yüksek Korelasyonlar\n")
    pop_corr = corr_matrix[[target]].sort_values(by=target, ascending=False).reset_index()
    pop_corr.columns = ['Değişken', 'Korelasyon Skoru']
    f.write(pop_corr.to_markdown(index=False) + "\n\n")
    
    f.write("**💡 İş Değeri Yorumu:** Enerji ile Loudness (ses yüksekliği) arasında çok pozitif güçlü korelasyon, Akustiklik ile Enerji arasında ise güçlü boyutta ters korelasyonlar tespit edilmiştir. Hedef değişkenin tahmini için karmaşık/non-lineer ilişkiler (Ağaç tabanlı algoritmalar) tercih edilmelidir.\n\n")

# Handoff Doc
with open('reports/markdown/DATA_PREP_HANDOFF.md', 'w', encoding='utf-8') as f:
    f.write("# 📝 DATA PREP HANDOFF (Görev Paylaşım Belgesi)\n\n")
    f.write("Bu belge EDA Expert analizleri sonucunda **DataPrep Expert** ajanına aktarılan iş paketlerini içerir.\n\n")
    f.write("### 1. Eksik Veriler\n- `artists`, `album_name`, `track_name` sütunlarındaki kayıp veriler sadece 1 adet. Satırları *drop* ediniz.\n\n")
    f.write("### 2. Gereksiz Veriler\n- `track_id` index olarak ayarlanmalı veya model eğitimi için X setinden çıkarılmalıdır.\n\n")
    f.write("### 3. Hedef Değişken ve Outlier\n- `popularity` 0 olan ciddi bir kitle var. Problem türümüz regressyon ise bu outlier'lar özel bir dönüşüme tabi tutulmalı veya Tree-based modeller için müdahalesiz bırakılmalı.\n")

print("✅ Derinlemesine EDA analizi gerçekleştirildi. Görseller ve tablolar doğrudan Markdown raporlarına gömüldü.")