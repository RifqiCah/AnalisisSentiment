# Proyek Analisis Sentimen Ulasan E-commerce

Proyek ini melakukan analisis sentimen pada kumpulan data ulasan produk dari platform e-commerce menggunakan berbagai algoritma Machine Learning klasik. Proses ini melibatkan serangkaian tahap Pra-pemrosesan Bahasa Alami (Natural Language Processing/NLP) untuk mempersiapkan data teks, diikuti dengan pembangunan dan evaluasi model klasifikasi sentimen.

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Tujuan Proyek](#tujuan-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
  - [Pra-pemrosesan Teks (NLP)](#pra-pemrosesan-teks-nlp)
  - [Vektorisasi Fitur](#vektorisasi-fitur)
  - [Model Machine Learning](#model-machine-learning)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
- [Kebutuhan (Dependencies)](#kebutuhan-dependencies)
- [Lisensi](#lisensi)

## Latar Belakang

Ulasan pelanggan adalah sumber informasi yang kaya mengenai pengalaman dan preferensi konsumen terhadap suatu produk atau layanan. Analisis sentimen memungkinkan kita untuk secara otomatis mengidentifikasi dan mengekstrak opini (positif, negatif, netral) dari teks-teks ulasan, yang dapat memberikan wawasan berharga bagi bisnis untuk pengambilan keputusan, peningkatan produk, dan strategi pemasaran.

## Tujuan Proyek

Tujuan utama dari proyek ini adalah:
1.  Membangun *pipeline* pra-pemrosesan teks yang robust untuk ulasan berbahasa Indonesia.
2.  Mengembangkan dan mengevaluasi beberapa model klasifikasi sentimen menggunakan algoritma Machine Learning klasik.
3.  Membandingkan performa model-model tersebut untuk menentukan algoritma terbaik dalam mengklasifikasikan sentimen ulasan.

## Dataset

Dataset yang digunakan dalam proyek ini berisi ulasan produk dari e-commerce. Kolom utama yang digunakan adalah:
-   `Ulasan`: Berisi teks ulasan pelanggan.
-   `label`: Berisi label sentimen yang sudah ditentukan (misalnya, 0 untuk negatif/netral, 1 untuk positif).

*Catatan: Dataset ini seharusnya ditempatkan di dalam folder `data/` di root proyek ini. Jika dataset berukuran besar, disarankan untuk tidak mengunggahnya ke GitHub dan memberikan tautan unduhan di sini.*

## Metodologi

### Pra-pemrosesan Teks (NLP)

Langkah-langkah pra-pemrosesan teks dilakukan untuk membersihkan dan menstandarisasi data ulasan, memastikan kualitas input untuk model Machine Learning. Tahapan yang dilakukan meliputi:
1.  **Penanganan Missing Values:** Mengisi ulasan kosong dengan nilai 'tidak ada komentar'.
2.  **Case Folding:** Mengubah semua teks menjadi huruf kecil.
3.  **Normalisasi Singkatan/Slang:** Mengubah kata-kata tidak baku atau singkatan menjadi bentuk bakunya menggunakan kamus kustom. Contoh: 'dgn' menjadi 'dengan', 'tdk' menjadi 'tidak', 'good' menjadi 'bagus'.
4.  **Penghapusan Tanda Baca, Angka, dan Karakter Khusus:** Membersihkan teks dari elemen non-alfabetik yang tidak relevan.
5.  **Tokenisasi:** Memecah setiap ulasan menjadi unit-unit kata (token).
6.  **Stopwords Removal:** Menghapus kata-kata umum yang tidak membawa makna sentimen signifikan (contoh: 'yang', 'dan'), dengan pengecualian kata-kata negasi (`tidak`, `bukan`) untuk menjaga makna sentimen.
7.  **Stemming:** Mengubah kata-kata berimbuhan menjadi kata dasar menggunakan pustaka Sastrawi untuk Bahasa Indonesia. Contoh: 'dicoba' menjadi 'coba', 'pengiriman' menjadi 'kirim'.

### Vektorisasi Fitur

Setelah pra-pemrosesan, teks diubah menjadi representasi numerik yang dapat dipahami oleh model Machine Learning menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer**. TF-IDF memberikan bobot pada kata berdasarkan frekuensi kemunculannya dalam sebuah ulasan dan seberapa unik kata tersebut di seluruh korpus ulasan.

### Model Machine Learning

Tiga algoritma klasifikasi Machine Learning klasik diimplementasikan dan dilatih pada data yang sudah divetorisasi:
1.  **Naive Bayes (Multinomial Naive Bayes):** Sebuah model probabilistik yang bekerja baik untuk klasifikasi teks.
2.  **Logistic Regression:** Model linear yang sering digunakan sebagai *baseline* yang kuat dan memberikan probabilitas kelas.
3.  **Support Vector Machine (LinearSVC):** Efektif dalam data berdimensi tinggi dan sering memberikan performa tinggi dalam klasifikasi teks.

## Hasil dan Evaluasi

Model-model dievaluasi menggunakan metrik seperti Akurasi, Precision, Recall, F1-Score, Confusion Matrix, dan Kurva ROC. Data dibagi menjadi 90% set pelatihan dan 10% set pengujian (`test_size=0.1`) dengan `stratify` berdasarkan label untuk menjaga proporsi kelas.

Berikut adalah ringkasan akurasi yang diperoleh untuk setiap model pada data pengujian:

-   **Naive Bayes:** 92%
-   **Logistic Regression:** 97%
-   **Support Vector Machine (SVM):** 97%

Berdasarkan hasil akurasi, model **Logistic Regression** dan **SVM** menunjukkan performa yang sangat kuat dan setara, melampaui Naive Bayes dalam mengklasifikasikan sentimen ulasan e-commerce pada dataset ini.

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

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](https://opensource.org/licenses/MIT) - lihat file [LICENSE](LICENSE) untuk detailnya.
