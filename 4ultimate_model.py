import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

print("🧠 ULTIMATE YAPAY ZEKA (V2.0) BAŞLATILIYOR...")

# 1. HER İKİ BEYNİ YÜKLE VE BİRLEŞTİR (Data Fusion)
print("🔄 Kimyasal veriler ile Evrimsel skorlar birleştiriliyor...")
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")

# Sadece evrimsel skorları alıp kimya tablosuna 'Mutasyon_Adi' üzerinden yapıştırıyoruz
df_evrim_secili = df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']]
df_final = pd.merge(df_kimya, df_evrim_secili, on='Mutasyon_Adi', how='left')

# O eski, içi boş ve bizi %73'te bırakan GVGD ve Prior skorlarını tarihin çöplüğüne atıyoruz
df_final = df_final.drop(columns=['Prior_Skoru', 'Align_GVGD_Skoru'])

# 2. YAPAY ZEKA İÇİN DİL ÇEVİRİSİ (Encoding)
le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

le_komsuluk = LabelEncoder()
df_final['Komsuluk_Kodu'] = le_komsuluk.fit_transform(df_final['Komsuluk_Dizilimi'].astype(str))

# 3. YENİ VE ÇOK DAHA GÜÇLÜ ÖZNİTELİKLERİMİZ (X)
X = df_final[['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
              'Molekuler_Agirlik_Farki', 'Polarite_Degisimi', 'Komsuluk_Kodu',
              'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']]

y = df_final['ETIKET']

# 4. VERİYİ BÖL (%80 Eğitim, %20 Sınav)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. XGBOOST MOTORU (Hiperparametreleri V2.0 için güçlendirdik!)
print("⚙️ XGBoost V2.0 çalışıyor... Kimya, Komşuluk ve Evrim Tarihi öğreniliyor...")
model = xgb.XGBClassifier(
    n_estimators=150,      # Daha fazla karar ağacı
    learning_rate=0.05,    # Daha yavaş ve detaylı öğrensin
    max_depth=6,           # Daha derin bağlantılar kursun
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ V2.0 Eğitim tamamlandı!\n")

# 6. SINAV ZAMANI
y_tahmin = model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin)

# 7. YENİ KARNEYİ YAZDIR
print(f"🏆 ULTIMATE MODEL DOĞRULUK ORANI: %{dogruluk * 100:.2f}")
print("\n--- 📋 Detaylı Performans Raporu ---")
print(classification_report(y_test, y_tahmin))

print("\n--- 🎯 Hata Matrisi ---")
print(confusion_matrix(y_test, y_tahmin))
print("------------------------------------\n")
print("🎉 Eğer başarı oranımız arttıysa, o boş VUS tablosunu tahmin etmeye tam olarak hazırız!")