import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🚀 TEKNOFEST FİNAL ŞOVU: VUS TAHMİN MODÜLÜ (XGBOOST EDİSYONU) BAŞLATILIYOR...")

# ---------------------------------------------------------
# 1. ŞAMPİYON MODELİ (XGBOOST) 18 KOLONLA EĞİT
# ---------------------------------------------------------
print("🧠 Şampiyon modelimiz XGBoost eğitiliyor...")
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_egitim = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

le_gen = LabelEncoder()
df_egitim['Gen_Kodu'] = le_gen.fit_transform(df_egitim['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11: return x_str
    return "XXXXXXXXXXX"

df_egitim['Komsuluk_Dizilimi'] = df_egitim['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

komsuluk_sutunlari = []
for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    komsuluk_sutunlari.append(kolon_adi)
    df_egitim[kolon_adi] = df_egitim['Komsuluk_Dizilimi'].str[i]
    df_egitim[kolon_adi] = LabelEncoder().fit_transform(df_egitim[kolon_adi])

X_egitim = df_egitim[['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
                      'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
                      'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru'] + komsuluk_sutunlari]
y_egitim = df_egitim['ETIKET']

# XGBoost'u tüm veriyle eğitiyoruz
model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42, eval_metric='logloss')
model.fit(X_egitim, y_egitim)

# ---------------------------------------------------------
# 2. VUS VERİSİNİ TAHMİN ETME ZAMANI! (YENİ AKILLI DOLDURMA)
# ---------------------------------------------------------
print("🔍 İçi dopdolu yeni VUS tablosu (VUS_Dolu_Veriler.xlsx) okunuyor...")
df_vus = pd.read_excel("VUS_Dolu_Veriler.xlsx") # Dosyayı okuyacağımız DOĞRU YER burası!

# XGBoost çökmesin diye VUS verisini X_egitim ile eşleştiriyoruz
X_vus = pd.DataFrame(index=df_vus.index, columns=X_egitim.columns)

gen_mapping = dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))
# Fabrikada kolon adını 'Gen' yaptığımız için burada da 'Gen' arıyoruz
X_vus['Gen_Kodu'] = df_vus['Gen'].astype(str).map(gen_mapping).fillna(0)

print("⚙️ VUS verilerinin biyokimyasal ve evrimsel özellikleri eşleştiriliyor...")

# Liste düzgünce kapatıldı:
sayisal_kolonlar = ['Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 
                    'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']

for col in sayisal_kolonlar:
    if col in df_vus.columns:
        X_vus[col] = df_vus[col].fillna(X_egitim[col].mean())
    else:
        print(f"⚠️ DİKKAT: VUS dosyasında '{col}' yok! Mecburen ortalama atandı.")
        X_vus[col] = X_egitim[col].mean()

if 'Polarite_Degisimi' in df_vus.columns:
    X_vus['Polarite_Degisimi'] = df_vus['Polarite_Degisimi'].fillna(X_egitim['Polarite_Degisimi'].mode()[0])
else:
    X_vus['Polarite_Degisimi'] = X_egitim['Polarite_Degisimi'].mode()[0]

if 'Komsuluk_Dizilimi' in df_vus.columns:
    df_vus['Komsuluk_Dizilimi'] = df_vus['Komsuluk_Dizilimi'].apply(guvenli_dizilim)
    for i in range(11):
        kolon_adi = f'Komsu_{i-5}'
        X_vus[kolon_adi] = LabelEncoder().fit_transform(df_vus['Komsuluk_Dizilimi'].str[i])
else:
    print("⚠️ DİKKAT: VUS dosyasında 'Komsuluk_Dizilimi' yok! Komşuluk etkisi sıfırlandı.")
    for col in komsuluk_sutunlari:
        X_vus[col] = 0 

print("🤖 XGBoost, 18 boyutta bilinmeyen varyantları inceliyor...")

# TAHMİNLERİ YAP! (0.30 EŞİĞİ DEVREDE!)
vus_olasilik = model.predict_proba(X_vus)[:, 1]
vus_tahminleri = (vus_olasilik >= 0.30).astype(int)

df_vus['YAPAY_ZEKA_TAHMINI'] = vus_tahminleri
df_vus['KANSER_RISK_OLASILIGI'] = (vus_olasilik * 100).round(2).astype(str) + "%"

final_dosya = "TEKNOFEST_VUS_Tahmin_XGBoost.xlsx"
df_vus.to_excel(final_dosya, index=False)

print("\n🎉 VEEEE FİNAL! Bilinmeyen tüm varyantlar XGBoost zekasıyla tahmin edildi!")
print(f"📁 Sonuçlar '{final_dosya}' adıyla kaydedildi.")
print("\n--- İŞTE DOKTORLARIN BULAMADIĞI, SENİN YAPAY ZEKANIN BULDUĞU İLK 5 TAHMİN ---")
# Fabrika çıktısına uygun kolon isimleriyle yazdırıyoruz:
print(df_vus[['Gen', 'Mutasyon_Adi', 'YAPAY_ZEKA_TAHMINI', 'KANSER_RISK_OLASILIGI']].head(5))