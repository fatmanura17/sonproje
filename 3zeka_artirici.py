import pandas as pd
import re

print("🧬 Evrimsel Zeka (BLOSUM62) ve Risk Skorlama Modülü Başlatılıyor...")

# 1. VERİYİ YÜKLE (İlk baştaki temizlenmemiş olanı alıyoruz ki üzerine yazalım)
try:
    df = pd.read_excel("-Dengeli_Veriseti_800.xlsx")
except FileNotFoundError:
    print("❌ HATA: Dengeli_Veriseti_800.xlsx bulunamadı!")
    exit()

# BLOSUM62 Evrimsel Değişim Matrisi (Evrimsel Korunmuşluk Skoru)
# Pozitif değerler: Evrimsel olarak kabul edilebilir (Zararsız eğilimli)
# Negatif değerler: Evrimsel olarak çok tehlikeli, doğaya aykırı (Patojenik eğilimli)
blosum62 = {
    ('A', 'V'): 0, ('A', 'D'): -2, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'P'): -1,
    ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'K'): 2, ('R', 'W'): -3, ('R', 'C'): -3,
    ('N', 'D'): 1, ('N', 'H'): 1, ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'K'): 0,
    ('D', 'E'): 2, ('D', 'N'): 1, ('D', 'G'): -1, ('D', 'V'): -3, ('D', 'A'): -2,
    ('C', 'Y'): -2, ('C', 'R'): -3, ('C', 'S'): -1, ('C', 'G'): -3, ('C', 'W'): -2,
    ('E', 'D'): 2, ('E', 'K'): 1, ('E', 'Q'): 2, ('E', 'A'): -1, ('E', 'V'): -2,
    ('Q', 'R'): 1, ('Q', 'K'): 1, ('Q', 'E'): 2, ('Q', 'P'): -1, ('Q', 'H'): 0,
    ('G', 'A'): 0, ('G', 'S'): 0, ('G', 'D'): -1, ('G', 'R'): -2, ('G', 'C'): -3,
    ('H', 'Y'): 2, ('H', 'N'): 1, ('H', 'Q'): 0, ('H', 'R'): 0, ('H', 'P'): -2,
    ('I', 'L'): 2, ('I', 'V'): 3, ('I', 'M'): 1, ('I', 'F'): 0, ('I', 'T'): -1,
    ('L', 'I'): 2, ('L', 'V'): 1, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('K', 'R'): 2, ('K', 'Q'): 1, ('K', 'E'): 1, ('K', 'T'): -1, ('K', 'A'): -1,
    ('M', 'L'): 2, ('M', 'I'): 1, ('M', 'V'): 1, ('M', 'T'): -1, ('M', 'R'): -1,
    ('F', 'Y'): 3, ('F', 'W'): 1, ('F', 'L'): 0, ('F', 'I'): 0, ('F', 'S'): -2,
    ('P', 'A'): -1, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'L'): -3, ('P', 'R'): -2,
    ('S', 'T'): 1, ('S', 'A'): 1, ('S', 'N'): 1, ('S', 'G'): 0, ('S', 'P'): -1,
    ('T', 'S'): 1, ('T', 'A'): 0, ('T', 'I'): -1, ('T', 'M'): -1, ('T', 'V'): 0,
    ('W', 'Y'): 2, ('W', 'F'): 1, ('W', 'R'): -3, ('W', 'C'): -2, ('W', 'L'): -2,
    ('Y', 'F'): 3, ('Y', 'W'): 2, ('Y', 'H'): 2, ('Y', 'C'): -2, ('Y', 'S'): -2,
    ('V', 'I'): 3, ('V', 'L'): 1, ('V', 'M'): 1, ('V', 'A'): 0, ('V', 'T'): 0
}

aa_cevirici = {'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Glu':'E', 'Gln':'Q', 'Gly':'G', 'His':'H', 'Ile':'I', 'Leu':'L', 'Lys':'K', 'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W', 'Tyr':'Y', 'Val':'V'}

def evrimsel_skor_hesapla(mutasyon_str):
    if pd.isna(mutasyon_str):
        return pd.Series([None, None]) # Evrimsel_Skor, InSilico_Risk
        
    mutasyon_str = str(mutasyon_str).replace('p.', '')
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    
    if not match:
        return pd.Series([0, 0])
        
    ilk_aa_kod, son_aa_kod = match.group(1), match.group(3)
    ilk_aa = aa_cevirici.get(ilk_aa_kod, ilk_aa_kod)
    son_aa = aa_cevirici.get(son_aa_kod, son_aa_kod)

    # BLOSUM62'den skoru çek, eğer tabloda yoksa (çok nadir bir değişimse) otomatik -4 (çok riskli) ver
    skor = blosum62.get((ilk_aa, son_aa), -4)
    
    # In Silico Risk Algoritması: Negatif evrimsel skor = Yüksek Kanser Riski
    risk_skoru = abs(skor) * 1.5 if skor < 0 else 0.5
    
    return pd.Series([skor, risk_skoru])

print("⚙️ Mutasyonların milyonlarca yıllık evrimsel geçmişi taranıyor...")
df[['Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']] = df['Mutasyon_Adi'].apply(evrimsel_skor_hesapla)

# Eskiden boş olan o işe yaramaz sütunları, bu havalı yeni zeka sütunlarıyla değiştiriyoruz
df = df.drop(columns=['Prior_Skoru', 'Align_GVGD_Skoru'])

print("✅ Evrimsel Skorlar ve Risk Analizleri Başarıyla Eklendi!")
print(df[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru', 'ETIKET']].head(10))

# Yeni veriyi kaydet
df.to_excel("YapayZeka_Evrimsel_Veri.xlsx", index=False)
print("\n📁 Yeni veri seti 'YapayZeka_Evrimsel_Veri.xlsx' olarak kaydedildi.")
print("🚀 Şimdi bu veriyi alıp hidrofobiklik ve komşuluk kodlarıyla birleştirme vakti!")