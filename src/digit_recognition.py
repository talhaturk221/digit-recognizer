import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

# Klasör ve dosya yollarını tanımlayalım
data_dir = os.path.join('..', 'data')  # 'data' klasörüne ulaşmak için '../data'
train_path = os.path.join(data_dir, 'train.csv')  # train.csv dosyasının yolu

# Veriyi yükle
data = pd.read_csv(train_path)

# Veriyi inceleyelim
print(data.head())

# Veriyi hazırlama
X = data.drop('label', axis=1).values  # 'label' hariç tüm sütunlar (özellikler)
y = data['label'].values  # 'label' sütunu hedef değişken (etiketler)

# Veriyi normalize et
X = X / 255.0  # 0-255 arası değerleri 0-1 aralığına dönüştür

# Lojistik Regresyon için düzleştirme (flatten)
X_flat = X.reshape(-1, 28 * 28)  # 28x28'lik görselleri düzleştiriyoruz

# CNN Modeli
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelleri Eğitme
def train_models():
    # Lojistik Regresyon Eğitimi
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_flat, y)
    
    # CNN Eğitimi
    cnn_history = cnn_model.fit(X.reshape(-1, 28, 28), y, epochs=5, batch_size=64, validation_split=0.2)

    return logreg_model, cnn_history

# Eğitim Görsellerini Çizme
def plot_combined_images(title, X_data, y_labels, predictions=None, num_per_digit=5):
    """0'dan 9'a kadar her rakamdan num_per_digit kadar örnek ve tahminleri tek bir figürde gösterir."""
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)

    digit_count = 0
    for digit in range(10):
        # İlgili rakamların indekslerini al
        indices = np.where(y_labels == digit)[0][:num_per_digit]
        for i, idx in enumerate(indices):
            digit_count += 1
            plt.subplot(10, num_per_digit, digit_count)
            plt.imshow(X_data[idx].reshape(28, 28), cmap='gray')
            if predictions is not None:
                plt.title(f"True: {digit}\nPred: {predictions[idx]}", fontsize=8)
            else:
                plt.title(f"True: {digit}", fontsize=8)
            plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Yanlış Tahmin Görsellerini Çizme
def plot_wrong_predictions(title, X_data, y_true, y_pred, num_per_digit=3):
    """Yanlış tahmin edilen rakamları tek bir figürde gösterir."""
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)

    digit_count = 0
    for digit in range(10):
        # Yanlış tahmin edilen ilgili rakamların indekslerini al
        indices = np.where((y_true == digit) & (y_true != y_pred))[0][:num_per_digit]
        for i, idx in enumerate(indices):
            digit_count += 1
            plt.subplot(10, num_per_digit, digit_count)
            plt.imshow(X_data[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {digit}\nPred: {y_pred[idx]}", fontsize=8)
            plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Modelleri Eğit ve Görselleri Çıkar
logreg_model, cnn_history = train_models()

# Eğitim Görsellerini Göster (0-9 Arası Her Rakamdan Örnekler)
plot_combined_images("Eğitim Verilerinden Örnekler (0'dan 9'a)", X, y, num_per_digit=5)

# Lojistik Regresyon Doğruluğu
logreg_accuracy = logreg_model.score(X_flat, y)
print(f"Lojistik Regresyon Doğruluğu: {logreg_accuracy * 100:.2f}%")

# Lojistik Regresyon Yanlış Tahminlerini Göster
logreg_predictions = logreg_model.predict(X_flat)
logreg_wrong_predictions = logreg_predictions != y
print(f"Lojistik Regresyon Yanlış Tahmin Edilen Görselleri Gösteriliyor...")
plot_wrong_predictions("Lojistik Regresyon Yanlış Tahmin Edilen Görseller", X, y, logreg_predictions, num_per_digit=3)

# CNN Tahminleri
cnn_predictions = np.argmax(cnn_model.predict(X.reshape(-1, 28, 28)), axis=1)

# CNN Doğruluğu
cnn_accuracy = np.mean(cnn_predictions == y)
print(f"CNN Doğruluğu: {cnn_accuracy * 100:.2f}%")

# CNN Yanlış Tahminlerini Göster
print(f"CNN Yanlış Tahmin Edilen Görselleri Gösteriliyor...")
plot_wrong_predictions("CNN Yanlış Tahmin Edilen Görseller", X, y, cnn_predictions, num_per_digit=3)

# Eğitim ve Doğrulama Grafikleri
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(cnn_history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('CNN Doğruluk')  # CNN başlığını ekledim
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Eğitim Kayıpları')
plt.plot(cnn_history.history['val_loss'], label='Doğrulama Kayıpları')
plt.title('CNN Kayıplar')  # CNN başlığını ekledim
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()
