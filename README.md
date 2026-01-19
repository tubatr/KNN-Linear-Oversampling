# KNN Linear Oversampling
### Synthetic Sample Generation for Imbalanced Datasets / Dengesiz Veri Kümeleri İçin Sentetik Örnek Üretimi

This project implements a **custom oversampling algorithm** using **K-Nearest Neighbours (KNN)** and **linear interpolation** to handle class imbalance problems in classification tasks.  
The algorithm is fully **from scratch**, without using SMOTE or other oversampling libraries.

Bu proje, sınıf dengesizliği problemlerini çözmek için **K-En Yakın Komşu (KNN)** ve **lineer enterpolasyon** yöntemlerini kullanarak özel bir oversampling algoritması sunar.  
Algoritma tamamen **sıfırdan geliştirilmiştir**, SMOTE veya hazır oversampling kütüphaneleri kullanılmamıştır.

---

## ✨ Key Features / Öne Çıkan Özellikler

- Custom implementation: KNN + Linear Interpolation  
- Multiple manually created datasets (binary & multi-class)  
- Before/After visualization for data distribution  
- Classification performance evaluation (Precision, Recall, F1-Score, AUC-ROC)  
- Easy to extend for new datasets  

- Özel implementasyon: KNN + Lineer Enterpolasyon  
- Çeşitli manuel oluşturulmuş veri kümeleri (binary ve multi-class)  
- Oversampling öncesi ve sonrası veri dağılım görselleştirme  
- Sınıflandırma performans değerlendirmesi (Precision, Recall, F1-Score, AUC-ROC)  
- Yeni veri kümeleri için kolay genişletilebilir yapı  

---

## 🧠 Oversampling Algorithm / Oversampling Algoritması

**Algorithm Steps / Adımlar:**

1. Randomly select a minority class instance / Azınlık sınıfından rastgele bir örnek seçilir  
2. Find k nearest neighbours within the minority class / K en yakın komşu bulunur  
3. Randomly pick one neighbour / Bir komşu rastgele seçilir  
4. Generate a synthetic sample using linear interpolation / Lineer enterpolasyon ile sentetik örnek üretilir  
5. Repeat until all classes have the same number of instances / Tüm sınıflar eşit sayıya ulaşana kadar tekrar edilir  

**Mathematical Formula / Matematiksel Formül:**

\[
s = p + \lambda \cdot (n - p)
\]

- \( p \) = minority sample / azınlık örneği  
- \( n \) = selected neighbour / seçilen komşu  
- \( \lambda \in [0,1] \) = random interpolation factor / rastgele enterpolasyon katsayısı  

---

## 🧪 Core Algorithm Code / Temel Algoritma Kodu

```python
def custom_oversample(X, y, k=5):
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    X_resampled, y_resampled = [X], [y]

    for cls, count in zip(classes, counts):
        if count < max_count:
            n_needed = max_count - count
            X_minority = X[y == cls]
            actual_k = min(k, len(X_minority) - 1)
            if actual_k < 1: actual_k = 1
            nbrs = NearestNeighbors(n_neighbors=actual_k + 1).fit(X_minority)
            synthetic_samples = []
            for _ in range(n_needed):
                p = X_minority[np.random.randint(0, len(X_minority))]
                indices = nbrs.kneighbors(p.reshape(1, -1), return_distance=False)[0][1:]
                neighbor = X_minority[np.random.choice(indices)]
                s = p + np.random.random() * (neighbor - p)
                synthetic_samples.append(s)
            X_resampled.append(np.array(synthetic_samples))
            y_resampled.append(np.full(len(synthetic_samples), cls))
    X_final = np.vstack(X_resampled)
    y_final = np.hstack(y_resampled)
    return X_final, y_final

def plot_dashboard(X_old, y_old, probs_old, X_new, y_new, probs_new, name, n_classes, y_test_old, y_test_new):
    # Generates 2x2 dashboard:
    # Top-left: Before scatter
    # Top-right: After scatter
    # Bottom-left: ROC before
    # Bottom-right: ROC after

clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
clf.fit(X_train_res, y_train_res)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)
