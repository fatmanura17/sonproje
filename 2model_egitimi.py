import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

print("🧠 Yapay Zeka (XGBoost) eğitim modülü başlatılıyor...")

# 1. TEMİZLENMİŞ VERİYİ YÜKLE
df = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")

# 2. YAPAY ZEKA İÇİN DİL ÇEVİRİSİ (Encoding)
# XGBoost sadece sayılardan anlar. "BRCA1" veya "CERTLKYFLGI" gibi metinleri
# algoritmanın anlayabileceği sayısal kodlara çeviriyoruz.
le_gen = LabelEncoder()
df['Gen_Kodu'] = le_gen.fit_transform(df['Gen'])

le_komsuluk = LabelEncoder()
df['Komsuluk_Kodu'] = le_komsuluk.fit_transform(df['Komsuluk_Dizilimi'])

# 3. HEDEF VE ÖZNİTELİKLERİ BELİRLE (X ve y)
# Modeli eğiteceğimiz özellikler (X)
X = df[['Gen_Kodu', 'Popülasyon_Frekansi', 'Prior_Skoru', 'Align_GVGD_Skoru', 
        'Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 'Polarite_Degisimi', 'Komsuluk_Kodu']]

# Tahmin etmeye çalıştığımız şey (y) -> Patojenik mi (1) Zararsız mı (0)?
y = df['ETIKET']

# 4. VERİYİ EĞİTİM (%80) VE TEST (%20) OLARAK BÖL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"📊 Veri bölündü: {len(X_train)} satır eğitim için, {len(X_test)} satır test (sınav) için ayrıldı.")

# 5. XGBOOST MODELİNİ YARAT VE EĞİT
print("⚙️ XGBoost motoru çalışıyor, model biyokimyasal kuralları öğreniyor...")
model = xgb.XGBClassifier(
    n_estimators=100,      # Kaç tane karar ağacı kurulacak?
    learning_rate=0.1,     # Öğrenme hızı
    max_depth=5,           # Ağaçların derinliği
    random_state=42,
    eval_metric='logloss'
)

# İşte o sihirli satır: Modelin veriyi ezberlemeden öğrendiği an!
model.fit(X_train, y_train)
print("✅ Eğitim tamamlandı!")

# 6. MODELİ TEST ET (HİÇ GÖRMEDİĞİ %20'LİK VERİYLE)
y_tahmin = model.predict(X_test)

# 7. SONUÇLARI (KARNEYİ) EKRANA YAZDIR
dogruluk = accuracy_score(y_test, y_tahmin)
print(f"\n🏆 Modelin Doğruluk (Accuracy) Oranı: %{dogruluk * 100:.2f}")

print("\n--- 📋 Detaylı Performans Raporu ---")
print(classification_report(y_test, y_tahmin))

print("\n--- 🎯 Hata Matrisi (Confusion Matrix) ---")
print(confusion_matrix(y_test, y_tahmin))
print("------------------------------------------\n")