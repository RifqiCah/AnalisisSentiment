# Proyek Analisis Sentimen Ulasan E-commerce

Proyek ini melakukan analisis sentimen pada kumpulan data ulasan produk dari platform e-commerce menggunakan berbagai algoritma Machine Learning klasik. Proses ini melibatkan serangkaian tahap Pra-pemrosesan Bahasa Alami (Natural Language Processing/NLP) untuk mempersiapkan data teks, diikuti dengan pembangunan, evaluasi, dan optimasi model klasifikasi sentimen.

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Tujuan Proyek](#tujuan-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
  - [Pra-pemrosesan Teks (NLP)](#pra-pemrosesan-teks-nlp)
  - [Vektorisasi Fitur](#vektorisasi-fitur)
  - [Model Machine Learning](#model-machine-learning)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
  - [Alur Evaluasi dan Optimasi Model](#alur-evaluasi-dan-optimasi-model)
  - [Kesimpulan Hasil](#kesimpulan-hasil)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
- [Kebutuhan (Dependencies)](#kebutuhan-dependencies)

## Latar Belakang

Ulasan pelanggan adalah sumber informasi yang kaya mengenai pengalaman dan preferensi konsumen terhadap suatu produk atau layanan. Analisis sentimen memungkinkan kita untuk secara otomatis mengidentifikasi dan mengekstrak opini (positif, negatif) dari teks-teks ulasan, yang dapat memberikan wawasan berharga bagi bisnis untuk pengambilan keputusan, peningkatan produk, dan strategi pemasaran.

## Tujuan Proyek

Tujuan utama dari proyek ini adalah:
1.  Membangun *pipeline* pra-pemrosesan teks yang robust untuk ulasan berbahasa Indonesia.
2.  Mengembangkan dan mengevaluasi beberapa model klasifikasi sentimen menggunakan algoritma Machine Learning klasik.
3.  Melakukan optimasi hyperparameter untuk menemukan pengaturan terbaik bagi setiap model.
4.  Membandingkan performa model-model yang telah dioptimalkan untuk menentukan algoritma terbaik dalam mengklasifikasikan sentimen ulasan.

## Dataset

Dataset yang digunakan dalam proyek ini berisi 1925 ulasan produk dari e-commerce. Kolom utama yang digunakan adalah:
-   `Ulasan`: Berisi teks ulasan pelanggan.
-   `label`: Berisi label sentimen yang sudah ditentukan (misalnya, 0 untuk negatif, 1 untuk positif).

*Catatan: Dataset ini seharusnya ditempatkan di dalam folder `data/` di root proyek ini.*

## Metodologi

### Pra-pemrosesan Teks (NLP)

Langkah-langkah pra-pemrosesan teks dilakukan untuk membersihkan dan menstandarisasi data ulasan, memastikan kualitas input untuk model Machine Learning. Tahapan yang dilakukan meliputi:
1.  **Penanganan Missing Values:** Mengisi ulasan kosong dengan nilai 'tidak ada komentar'.
2.  **Case Folding:** Mengubah semua teks menjadi huruf kecil.
3.  **Normalisasi Singkatan/Slang:** Mengubah kata-kata tidak baku atau singkatan menjadi bentuk bakunya menggunakan kamus kustom. Contoh: 'dgn' menjadi 'dengan', 'tdk' menjadi 'tidak', 'good' menjadi 'bagus'.
4.  **Stopwords Removal:** Menghapus kata-kata umum yang tidak membawa makna sentimen signifikan (contoh: 'yang', 'dan'), dengan pengecualian kata-kata negasi (`tidak`) untuk menjaga makna sentimen.
5.  **Tokenisasi:** Memecah setiap ulasan menjadi unit-unit kata (token).
6.  **Stemming:** Mengubah kata-kata berimbuhan menjadi kata dasar menggunakan pustaka Sastrawi untuk Bahasa Indonesia. Contoh: 'dicoba' menjadi 'coba', 'pengiriman' menjadi 'kirim'.

### Vektorisasi Fitur

Setelah pra-pemrosesan, teks diubah menjadi representasi numerik yang dapat dipahami oleh model Machine Learning menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer**. TF-IDF memberikan bobot pada kata berdasarkan frekuensi kemunculannya dalam sebuah ulasan dan seberapa unik kata tersebut di seluruh korpus ulasan.

### Model Machine Learning

Tiga algoritma klasifikasi Machine Learning klasik diimplementasikan, dievaluasi, dan dioptimalkan:
1.  **Gaussian Naive Bayes:** Sebuah model probabilistik yang bekerja baik untuk klasifikasi teks.
2.  **Logistic Regression:** Model linear yang sering digunakan sebagai *baseline* yang kuat dan memberikan probabilitas kelas.
3.  **Support Vector Machine (LinearSVC):** Efektif dalam data berdimensi tinggi dan sering memberikan performa tinggi dalam klasifikasi teks.

## Hasil dan Evaluasi

### Alur Evaluasi dan Optimasi Model

Untuk mendapatkan hasil yang valid dan andal, proses evaluasi dan optimasi model dilakukan melalui dua tahap utama:

1.  **Validasi Silang (Cross-Validation):** Alih-alih menggunakan satu kali pembagian data `train_test_split`, proyek ini mengadopsi metode **K-Fold Cross-Validation** (menggunakan `StratifiedKFold` dengan 5 lipatan) untuk mengevaluasi performa dasar setiap model. Metode ini memberikan estimasi performa yang lebih stabil dan mengurangi faktor "keberuntungan" dalam pembagian data.

2.  **Optimasi Hyperparameter (Hyperparameter Tuning):** Untuk menemukan potensi maksimal dari setiap algoritma, dilakukan proses pencarian hyperparameter terbaik menggunakan **`GridSearchCV`**. `GridSearchCV` secara sistematis menguji berbagai kombinasi hyperparameter dan menggunakan skema K-Fold Cross-Validation secara internal untuk menentukan kombinasi mana yang menghasilkan skor akurasi rata-rata tertinggi.

### Kesimpulan Hasil

Setelah melalui proses evaluasi dan optimasi yang ketat, berikut adalah rekapitulasi performa dari setiap model yang telah di-tuning:

| Model | Hyperparameter Terbaik | Akurasi Cross-Validated |
| :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | `{'C': 1}` | **90.7%** |
| **Logistic Regression** | `[Isi dengan param terbaik LR Anda]` | `[Isi dengan skor terbaik LR Anda]` |
| **Gaussian Naive Bayes** | `[Isi dengan param terbaik NB Anda]`| `[Isi dengan skor terbaik NB Anda]` |

Dari ketiga algoritma yang diuji, model **Support Vector Machine (SVM) menunjukkan performa yang paling unggul dan konsisten**.

Proses *hyperparameter tuning* mengonfirmasi bahwa parameter `C=1` merupakan pengaturan yang paling optimal untuk model SVM, menghasilkan akurasi rata-rata *cross-validation* yang terverifikasi sebesar **90.7%**. Performa ini terbukti lebih superior dibandingkan model-model lainnya setelah melalui proses optimasi yang adil dan sebanding.

Dengan demikian, dapat disimpulkan bahwa **model Support Vector Machine (SVM) dengan parameter C=1 adalah model yang paling akurat, stabil, dan andal** untuk diimplementasikan dalam sistem analisis sentimen pada kasus ini.

## Cara Menjalankan Proyek

Untuk menjalankan proyek ini secara lokal di komputer Anda:

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/RifqiCah/AnalisisSentiment.git](https://github.com/RifqiCah/AnalisisSentiment.git)
    cd AnalisisSentiment
    ```

2.  **Buat dan Aktifkan Lingkungan Conda:**
    Disarankan untuk membuat lingkungan virtual baru untuk menghindari konflik dependensi.
    ```bash
    conda create -n analisis_sentiment_env python=3.12 pandas numpy scikit-learn matplotlib jupyter seaborn sastrawi wordcloud
    conda activate analisis_sentiment_env
    ```
    *Catatan: Pastikan semua library yang disebutkan di atas terinstal. Jika ada yang terlewat, Anda bisa menambahkannya nanti dengan `conda install <nama_library>` atau `pip install <nama_library>`.*

3.  **Siapkan Dataset:**
    * Pastikan file dataset `data_real.csv` (atau nama file dataset Anda) ditempatkan di dalam subfolder `data/` di root proyek.
    * Contoh path: `AnalisisSentiment/data/data_real.csv`

4.  **Jalankan Jupyter Notebook:**
    * Buka Visual Studio Code dan buka folder proyek `AnalisisSentiment`.
    * Pilih interpreter Python yang benar di VS Code (klik di pojok kanan bawah, lalu pilih `Python 3.12.x ('analisis_sentiment_env': conda)`).
    * Buka file `analisis_sentimen_ulasan.ipynb` (atau nama file notebook Anda) dan jalankan setiap sel secara berurutan.

## Kebutuhan (Dependencies)

Proyek ini membutuhkan library Python berikut:
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `Sastrawi`
-   `wordcloud`