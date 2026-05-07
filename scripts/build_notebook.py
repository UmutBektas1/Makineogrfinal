import json
import os

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🎵 Uçtan Uca (End-to-End) Spotify Hit Tahmini Pipeline\n",
        "Bu notebook, takımdaki Agent'ların parça parça yaptığı işlemlerin (Veri Yükleme, EDA, Veri Temizleme (DataPrep), Feature Engineering ve Model Training kısımlarının) **hepsini tek bir akışta** toplar."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
        "from xgboost import XGBRegressor\n",
        "from lightgbm import LGBMRegressor\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "plt.style.use('seaborn-v0_8-whitegrid')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Veri Yükleme ve Hızlı İnceleme"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df = pd.read_csv('../data/raw/dataset.csv')\n",
        "if 'Unnamed: 0' in df.columns:\n",
        "    df = df.drop(columns=['Unnamed: 0'])\n",
        "df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Veri Temizleme ve Özellik Mühendisliği (Data Preparation)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 1. Kategorik Eksikleri Doldurma\n",
        "df['artists'] = df['artists'].fillna('Unknown')\n",
        "df['album_name'] = df['album_name'].fillna('Unknown')\n",
        "df['track_name'] = df['track_name'].fillna('Unknown')\n",
        "\n",
        "# 2. Aykırı Değer Baskılama (Capping - IQR Metodu)\n",
        "for col in ['duration_ms', 'loudness', 'tempo']:\n",
        "    Q1 = df[col].quantile(0.25)\n",
        "    Q3 = df[col].quantile(0.75)\n",
        "    IQR = Q3 - Q1\n",
        "    lower_bound = Q1 - 1.5 * IQR\n",
        "    upper_bound = Q3 + 1.5 * IQR\n",
        "    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])\n",
        "    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])\n",
        "\n",
        "# 3. Yeni Özellik Çıkarımı (Feature Engineering)\n",
        "df['duration_min'] = df['duration_ms'] / 60000.0\n",
        "df['energy_loudness_ratio'] = df['energy'] / (df['loudness'] + 60 + 1e-5)\n",
        "df['danceability_squared'] = df['danceability'] ** 2\n",
        "\n",
        "bins = [-np.inf, 90, 130, np.inf]\n",
        "labels = [0, 1, 2] # 0: Yavaş, 1: Orta, 2: Hızlı\n",
        "df['tempo_encoded'] = pd.cut(df['tempo'], bins=bins, labels=labels).astype(int)\n",
        "\n",
        "if 'explicit' in df.columns:\n",
        "    df['is_explicit'] = df['explicit'].astype(int)\n",
        "    df.drop(columns=['explicit'], inplace=True)\n",
        "\n",
        "# 4. Gereksiz Verileri Çıkarma (Metinler)\n",
        "drop_cols = ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']\n",
        "df_cleaned = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')\n",
        "print(\"Veri Temizleme Tamamlandı! Yeni Boyut:\", df_cleaned.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Makine Öğrenmesi (Model Training)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Performans için %20 Sampling yapıyoruz\n",
        "df_sample = df_cleaned.sample(frac=0.2, random_state=42)\n",
        "\n",
        "X = df_sample.drop(columns=['popularity'])\n",
        "y = df_sample['popularity']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(\"Eğitim Seti Boyutu:\", X_train_scaled.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "models = {\n",
        "    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),\n",
        "    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),\n",
        "    'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)\n",
        "}\n",
        "\n",
        "results = []\n",
        "for name, model in models.items():\n",
        "    model.fit(X_train_scaled, y_train)\n",
        "    preds = model.predict(X_test_scaled)\n",
        "    r2 = r2_score(y_test, preds)\n",
        "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
        "    results.append({'Model': name, 'R2': r2, 'RMSE': rmse})\n",
        "\n",
        "res_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)\n",
        "res_df"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Sonuçların Görselleştirilmesi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.figure(figsize=(10, 5))\n",
        "sns.barplot(data=res_df, x='R2', y='Model', palette='viridis')\n",
        "plt.title('Ağaç Modellerinin Başarı Karşılaştırması (R²)')\n",
        "plt.show()"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {"name": "ipython", "version": 3},
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/End_to_End_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
