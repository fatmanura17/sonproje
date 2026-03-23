import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

print("🚀 V4.3: SHAP UYUMLU ANALİZ MOTORU BAŞLATILIYOR...")

# 1. Veri Hazırlığı (Aynı kalıyor)
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11: return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final['Komsuluk_Dizilimi'].str[i])

X = df_final.drop(columns=['Mutasyon_Adi', 'Gen', 'Komsuluk_Dizilimi', 'ETIKET']).fillna(0)
y = df_final['ETIKET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Modeller
modeller = [
    ("1. Lojistik Regresyon", LogisticRegression(max_iter=500), "linear"),
    ("2. Random Forest", RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42), "tree"),
    ("3. CatBoost", CatBoostClassifier(iterations=150, learning_rate=0.05, depth=7, random_state=42, verbose=0), "tree"),
    ("4. Optimize XGBoost", xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss'), "tree")
]

# 3. Döngü
for isim, model, tip in modeller:
    print(f"\n📊 {isim} Analiz Ediliyor...")
    model.fit(X_train, y_train)
    
    # --- FEATURE IMPORTANCE ---
    plt.figure(figsize=(10, 6))
    if tip == "linear":
        importances = model.coef_[0]
    elif isim == "3. CatBoost":
        importances = model.get_feature_importance()
    else:
        importances = model.feature_importances_
        
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title(f"{isim} - Özellik Önemi")
    plt.savefig(f"FI_{isim.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    # --- SHAP ANALİZİ (GÜNCELLENEN KISIM) ---
    print(f"🧠 {isim} SHAP Değerleri Hesaplanıyor...")
    try:
        # Daha evrensel bir Explainer yapısı
        if tip == "linear":
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

        # SHAP değerlerinin formatını kontrol edip düzeltiyoruz
        # Bazı modellerde (XGBoost/CatBoost) 1D dizi, bazılarında list döner
        if isinstance(shap_values, list):
            # Eğer listeyse genellikle [0] negatif, [1] pozitif sınıftır
            shap_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3: # Bazı özel durumlarda 3D dönebilir
            shap_to_plot = shap_values[:, :, 1]
        else:
            shap_to_plot = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_to_plot, X_test, show=False)
        plt.title(f"{isim} - SHAP Analizi", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"SHAP_{isim.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {isim} başarıyla kaydedildi.")
        
    except Exception as e:
        print(f"❌ {isim} SHAP hatası: {e}")

print("\n🎉 Tüm grafikler klasöre düştü! Dilara'ya selam, analize devam!")