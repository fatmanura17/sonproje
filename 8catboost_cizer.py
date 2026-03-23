import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import shap

print("👑 YENİ KRALIN (CATBOOST) GRAFİKLERİ ÇİZİLİYOR...")

# 1. Veriyi Hazırla (Artık ezberlediğimiz 18 kolonlu V3.0 yapısı)
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11:
        return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

komsuluk_sutunlari = []
for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    komsuluk_sutunlari.append(kolon_adi)
    df_final[kolon_adi] = df_final['Komsuluk_Dizilimi'].str[i]
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final[kolon_adi])

X = df_final[['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
              'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
              'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru'] + komsuluk_sutunlari]
y = df_final['ETIKET']

# 2. CatBoost Canavarını Tüm Veriyle Eğit
print("⏳ Model eğitiliyor (Sessiz mod)...")
cat_model = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=7, random_state=42, verbose=0)
cat_model.fit(X, y)

# ---------------------------------------------------------
# 3. YENİ "TEACHER IMPORTANT" GRAFİĞİ (Feature Importance)
# ---------------------------------------------------------
print("📊 CatBoost Feature Importance Çiziliyor...")
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(cat_model.get_feature_importance(), index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='#1f77b4')
plt.title("Hangi Özellik Ne Kadar Önemli? (CatBoost Şampiyonu)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("CatBoost_Feature_Importance.png", dpi=300)
print("✅ CatBoost_Feature_Importance.png klasöre kaydedildi!")

# ---------------------------------------------------------
# 4. YENİ "ŞAHAP" (SHAP ANALİZİ)
# ---------------------------------------------------------
print("🧠 CatBoost SHAP Analizi Yapılıyor...")
explainer = shap.Explainer(cat_model)
shap_values = explainer(X)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("CatBoost_SHAP_Analizi.png", dpi=300, bbox_inches='tight')
print("✅ CatBoost_SHAP_Analizi.png klasöre kaydedildi!")

print("\n🎉 VEEEE FİNAL! Yeni Kralın yepyeni 2 grafiği rapor için hazır.")