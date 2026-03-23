import pandas as pd
import requests
import time

# 1. VERİYİ YÜKLEME
print("Veri yükleniyor, lütfen bekleyin...")
try:
    df_egitim = pd.read_excel("-Dengeli_Veriseti_800.xlsx")
    print("✅ Eğitim verisi başarıyla yüklendi!")
    print("\n--- Tablonun İlk 5 Satırı ---")
    print(df_egitim.head()) # display() yerine print() kullanıyoruz!
    print("-----------------------------\n")
except FileNotFoundError:
    print("❌ HATA: Excel dosyası bulunamadı!")

# 2. KİMYA LABORATUVARI SÖZLÜĞÜ
aa_ozellikleri = {
    'A': {'isim': 'Alanine', 'hidrofobiklik': 1.8, 'agirlik': 89.1, 'polarite': 'Nonpolar'},
    'R': {'isim': 'Arginine', 'hidrofobiklik': -4.5, 'agirlik': 174.2, 'polarite': 'Basic polar'},
    'N': {'isim': 'Asparagine', 'hidrofobiklik': -3.5, 'agirlik': 132.1, 'polarite': 'Polar'},
    'D': {'isim': 'Aspartic acid', 'hidrofobiklik': -3.5, 'agirlik': 133.1, 'polarite': 'Acidic polar'},
    'C': {'isim': 'Cysteine', 'hidrofobiklik': 2.5, 'agirlik': 121.2, 'polarite': 'Nonpolar'},
    'E': {'isim': 'Glutamic acid', 'hidrofobiklik': -3.5, 'agirlik': 147.1, 'polarite': 'Acidic polar'},
    'Q': {'isim': 'Glutamine', 'hidrofobiklik': -3.5, 'agirlik': 146.2, 'polarite': 'Polar'},
    'G': {'isim': 'Glycine', 'hidrofobiklik': -0.4, 'agirlik': 75.1, 'polarite': 'Nonpolar'},
    'H': {'isim': 'Histidine', 'hidrofobiklik': -3.2, 'agirlik': 155.2, 'polarite': 'Basic polar'},
    'I': {'isim': 'Isoleucine', 'hidrofobiklik': 4.5, 'agirlik': 131.2, 'polarite': 'Nonpolar'},
    'L': {'isim': 'Leucine', 'hidrofobiklik': 3.8, 'agirlik': 131.2, 'polarite': 'Nonpolar'},
    'K': {'isim': 'Lysine', 'hidrofobiklik': -3.9, 'agirlik': 146.2, 'polarite': 'Basic polar'},
    'M': {'isim': 'Methionine', 'hidrofobiklik': 1.9, 'agirlik': 149.2, 'polarite': 'Nonpolar'},
    'F': {'isim': 'Phenylalanine', 'hidrofobiklik': 2.8, 'agirlik': 165.2, 'polarite': 'Nonpolar'},
    'P': {'isim': 'Proline', 'hidrofobiklik': -1.6, 'agirlik': 115.1, 'polarite': 'Nonpolar'},
    'S': {'isim': 'Serine', 'hidrofobiklik': -0.8, 'agirlik': 105.1, 'polarite': 'Polar'},
    'T': {'isim': 'Threonine', 'hidrofobiklik': -0.7, 'agirlik': 119.1, 'polarite': 'Polar'},
    'W': {'isim': 'Tryptophan', 'hidrofobiklik': -0.9, 'agirlik': 204.2, 'polarite': 'Nonpolar'},
    'Y': {'isim': 'Tyrosine', 'hidrofobiklik': -1.3, 'agirlik': 181.2, 'polarite': 'Polar'},
    'V': {'isim': 'Valine', 'hidrofobiklik': 4.2, 'agirlik': 117.1, 'polarite': 'Nonpolar'}
}

aa_cevirici = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}

print("✅ Biyokimya modülü başarıyla yüklendi! Yapay zekamız artık kimya biliyor.")
print("🔥 Sistem bir sonraki aşama olan API bağlantıları için hazır!")

import re

# 3. GENETİK KOD ÇÖZÜCÜ VE KİMYASAL HESAPLAMA MODÜLÜ
def mutasyon_cozucu(mutasyon_str):
    # Eğer hücre boşsa pas geç
    if pd.isna(mutasyon_str) or not isinstance(mutasyon_str, str):
        return pd.Series([None, None, None])

    # "p." kısmını temizle (Örn: p.K1702Q -> K1702Q)
    mutasyon_str = mutasyon_str.replace('p.', '')

    # Regex ile harfleri ve sayıları ayır (Önceki AA, Pozisyon, Sonraki AA)
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    
    if not match:
        return pd.Series([None, None, None])
        
    ilk_aa_kod = match.group(1)
    pozisyon = match.group(2) # Komşuluk API'si için bunu birazdan kullanacağız!
    son_aa_kod = match.group(3)

    # 3 harfli format geldiyse (Met) 1 harfliye (M) çevir
    ilk_aa = aa_cevirici.get(ilk_aa_kod, ilk_aa_kod)
    son_aa = aa_cevirici.get(son_aa_kod, son_aa_kod)

    # Sözlüğümüzde bu amino asitlerin özellikleri var mı?
    if ilk_aa in aa_ozellikleri and son_aa in aa_ozellikleri:
        ozellik_ilk = aa_ozellikleri[ilk_aa]
        ozellik_son = aa_ozellikleri[son_aa]

        # Teknofest Şartnamesi: Kimyasal Farkların Hesaplanması
        hidro_fark = ozellik_son['hidrofobiklik'] - ozellik_ilk['hidrofobiklik']
        agirlik_fark = ozellik_son['agirlik'] - ozellik_ilk['agirlik']
        
        # Polarite değişti mi? (Değiştiyse 1, aynı kaldıysa 0)
        polarite_degisimi = 1 if ozellik_ilk['polarite'] != ozellik_son['polarite'] else 0

        return pd.Series([round(hidro_fark, 2), round(agirlik_fark, 2), polarite_degisimi])
    else:
        return pd.Series([None, None, None])

print("🧬 Mutasyonlar analiz ediliyor ve kimyasal farklar hesaplanıyor...")

# Pandas 'apply' büyüsü: Tüm 800 satırı saniyeler içinde yukarıdaki fonksiyona sokuyoruz
df_egitim[['Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 'Polarite_Degisimi']] = df_egitim['Mutasyon_Adi'].apply(mutasyon_cozucu)

print("✅ Biyokimyasal hesaplamalar tamamlandı!")
print("\n--- Teknofest Şartnamesine Uygun Yeni Sütunlar ---")
print(df_egitim[['Mutasyon_Adi', 'Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 'Polarite_Degisimi', 'ETIKET']].head())
print("--------------------------------------------------\n")

# 4. PROTEOMİK KOMŞULUK (+/- 5 AMİNO ASİT) MODÜLÜ
print("\n--- Adım 4: UniProt API Bağlantısı Başlıyor ---")
# BRCA1 (P38398) ve BRCA2 (P51587) için resmi UniProt ID'leri
uniprot_id = {'BRCA1': 'P38398', 'BRCA2': 'P51587'}
protein_dizilimleri = {}

print("🌐 İsviçre UniProt veritabanına bağlanılıyor... (Tam sekanslar çekiliyor)")
for gen, id_kod in uniprot_id.items():
    url = f"https://rest.uniprot.org/uniprotkb/{id_kod}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        # Fasta formatını temizle (ilk satırı at, kalanları birleştir)
        fasta_satirlari = response.text.split('\n')[1:]
        tam_sekans = "".join(fasta_satirlari)
        protein_dizilimleri[gen] = tam_sekans
        print(f"✅ {gen} başarıyla çekildi! (Uzunluk: {len(tam_sekans)} amino asit)")
    else:
        print(f"❌ {gen} sekansı çekilirken hata oluştu! İnternet bağlantını kontrol et.")

def komsuluk_bulucu(row):
    gen = row['Gen']
    mutasyon_str = str(row['Mutasyon_Adi']).replace('p.', '')
    
    # Mutasyon pozisyonunu bul (Örn: R1699G -> 1699)
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    if not match or gen not in protein_dizilimleri:
        return None
        
    pozisyon = int(match.group(2)) - 1 # Python'da indeksler 0'dan başlar, o yüzden -1
    sekans = protein_dizilimleri[gen]
    
    # +/- 5 amino asit penceresini (window) hesapla
    baslangic = max(0, pozisyon - 5)
    bitis = min(len(sekans), pozisyon + 6) # +6 çünkü Python slicing'de son sınır dahil edilmez
    
    # Dizilimi kes ve al
    komsu_dizilim = sekans[baslangic:bitis]
    return komsu_dizilim

print("🔍 Her varyant için +/- 5 amino asitlik genomik komşuluk dilimleniyor...")
# apply ile her satıra bu fonksiyonu uygula
df_egitim['Komsuluk_Dizilimi'] = df_egitim.apply(komsuluk_bulucu, axis=1)

print("✅ Komşuluk verisi başarıyla eklendi!")
print("\n--- Teknofest Şartnamesi: Proteomik Komşuluk ---")
print(df_egitim[['Gen', 'Mutasyon_Adi', 'Komsuluk_Dizilimi']].head())
print("------------------------------------------------\n")

import numpy as np
from sklearn.impute import KNNImputer

# 5. VERİ TEMİZLİĞİ VE KNN İLE BOŞLUK DOLDURMA (IMPUTATION)
print("\n--- Adım 5: Veri Temizliği ve Eksik Veri Tamamlama ---")

# Yüzde işaretlerini temizleyip saf ondalık sayıya (float) çeviriyoruz
def frekans_temizle(deger):
    if pd.isna(deger):
        return np.nan
    deger = str(deger).replace('%', '').strip()
    try:
        return float(deger) / 100
    except:
        return np.nan

df_egitim['Popülasyon_Frekansi'] = df_egitim['Popülasyon_Frekansi'].apply(frekans_temizle)

# GVGD Skorundaki "C" harflerini silip sadece sayıyı alıyoruz (Örn: C15 -> 15.0)
df_egitim['Align_GVGD_Skoru'] = df_egitim['Align_GVGD_Skoru'].astype(str).str.extract(r'(\d+)').astype(float)

# Modellerin hata vermemesi için NaN (boş) hücreleri 5 en yakın komşunun ortalamasıyla dolduruyoruz
sayisal_sutunlar = ['Popülasyon_Frekansi', 'Prior_Skoru', 'Align_GVGD_Skoru', 'Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 'Polarite_Degisimi']

print("🤖 KNN Algoritması devreye giriyor, boş hücreler sentetik olarak dolduruluyor...")
imputer = KNNImputer(n_neighbors=5)
df_egitim[sayisal_sutunlar] = imputer.fit_transform(df_egitim[sayisal_sutunlar])

print("✅ Tüm veriler makine öğrenmesi standartlarına getirildi!")

# 6. YAPAY ZEKAYA HAZIR VERİYİ EXCEL OLARAK KAYDETME
dosya_adi = "YapayZeka_Hazir_Veri.xlsx"
df_egitim.to_excel(dosya_adi, index=False)

print(f"\n🎉 TAA DAA! Veri ön işleme kusursuz tamamlandı.")
print(f"📁 Dosyanız '{dosya_adi}' adıyla TEKNOFEST klasörüne kaydedildi!")