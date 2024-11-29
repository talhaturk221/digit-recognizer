# digit-recognizer
 
# El Yazısı Rakam Tanıma Projesi

DigitR, el yazısı rakamları tanımak için kullanılan bir makine öğrenmesi ve derin öğrenme projesidir. Bu proje, hem Lojistik Regresyon hem de Convolutional Neural Network (CNN) modellerini kullanarak,  veri setiyle el yazısı rakamlarını sınıflandırır.

## Proje İçeriği

### 1. **Veri Seti**
Proje, el yazısı rakamları içeren veri setini kullanır. Veri seti şu dosyalarda bulunur:
- `train.csv`: Eğitim verisi (etiketli el yazısı rakamları).
- `test.csv`: Test verisi (etiketli el yazısı rakamları).

Veriler, her bir rakam için 28x28 piksel değerleri içeren görsellerden oluşur. Her piksel değeri 0-255 arasında bir değere sahiptir.

### 2. **Modeller**
Proje iki ana model içerir:
- **Lojistik Regresyon (Logistic Regression)**: Basit bir doğrusal model olan lojistik regresyon kullanılarak eğitilir.
- **CNN (Convolutional Neural Network)**: Derin öğrenme modelidir ve daha karmaşık özellikleri öğrenme yeteneğine sahiptir. Bu model, 2D konvolüsyon katmanları ile görselleri işler.

### 3. **Kullanım**
Projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

#### 1. Gerekli Kütüphaneleri Kurma
Proje için gereken Python kütüphanelerini yüklemek için:

pip install pandas numpy tensorflow scikit-learn matplotlib

#### 2. Projeyi Çalıştırma
Projeyi çalıştırmak için terminal veya komut satırında aşağıdaki komutu kullanabilirsiniz:

python src/digit_recognition.py

### 4. **Sonuçlar ve Çıktılar**
Eğitim ve test süreci sonunda, aşağıdaki çıktılar gözlemlenecektir:
-**Doğruluk Oranları**: Lojistik Regresyon ve CNN doğruluk oranlarını gösterir.
-**Eğitim Görselleri**: Eğitim verilerin görselleri.
- **Yanlış Tahmin Görselleri**: Lojistik Regresyon ve CNN modellerinin yanlış tahmin ettiği görseller.
- **Eğitim Doğruluğu ve Kayıplar Grafikleri**: CNN modelinin eğitim doğruluğu ve kayıp değerleri.
- 

### 5. **Teknik Detaylar**
- **Lojistik Regresyon**: Bu model, 28x28'lik görselleri düzleştirerek (flatten) kullanır ve basit bir doğrusal sınıflandırıcı olarak çalışır.
- **CNN**: Derin öğrenme yaklaşımını kullanan bu model, görsel verileri doğrudan işleyerek daha yüksek doğruluk oranları elde etmeyi amaçlar.

### 6. **Lisans**
Bu proje [MIT Lisansı] ile lisanslanmıştır.
