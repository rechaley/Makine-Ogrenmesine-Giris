Açıklamalar
Veri Setinin Yüklenmesi:

datasets.load_iris() fonksiyonu ile Iris veri seti yüklenir.
X değişkeni özellikleri (features) ve y değişkeni sınıf etiketlerini (labels) içerir.
Verilerin Standartlaştırılması:

StandardScaler kullanılarak veriler standartlaştırılır (her bir özellik, ortalaması 0 ve standart sapması 1 olacak şekilde ölçeklenir).
Veri Setinin Eğitim ve Test Setlerine Ayrılması:

train_test_split() fonksiyonu ile veri seti eğitim ve test setlerine ayrılır. Bu örnekte, veri setinin %30'u test seti olarak kullanılır.
Etiketlerin Gizlenmesi:

Eğitim setindeki etiketlerin %90'ı gizlenir. Gizlenen etiketler -1 değeriyle değiştirilir.
Label Spreading Modelinin Oluşturulması ve Eğitilmesi:

LabelSpreading algoritması kullanılarak model oluşturulur ve fit() fonksiyonu ile eğitilir.
Test Verisi Üzerinde Tahminlerin Yapılması:

predict() fonksiyonu ile test seti üzerinde tahminler yapılır.
Modelin Doğruluğunun Hesaplanması:

accuracy_score fonksiyonu ile modelin doğruluğu hesaplanır ve ekrana yazdırılır.
Sonuçların Görselleştirilmesi:

matplotlib kullanılarak sonuçlar görselleştirilir. Hatalı tahminler kırmızı 'x' ile işaretlenir.
Bu örnek, yarı denetimli öğrenmenin nasıl uygulanacağını ve etiketlenmemiş verilerin nasıl kullanılabileceğini göstermektedir.