# EN
## 🤖 Advanced RAG Question-Answer System

A modern RAG (Retrieval-Augmented Generation) system that generates intelligent answers using AI by collecting information from the web.

## 🌟 Features

### 🔍 Smart Web Search
- Automatic web search with DuckDuckGo
- Turkish language optimization
- Rate limiting and error handling

### 🧠 Advanced AI Response Generation
- Google Gemini 1.5 Flash integration
- Turkish language model support
- Cross-encoder re-ranking

### 📊 Modern Streamlit Interface
- Responsive and user-friendly design
- Real-time system monitoring
- Detailed analytics and visualization

### 🔧 Advanced Technical Features
- FAISS vector indexing
- Sentence-BERT embedding
- Intelligent document chunking
- Confidence score calculation

## 🚀 Installation

### 1. Requirements

```bash
pip install streamlit
pip install sentence-transformers
pip install faiss-cpu
pip install duckduckgo-search
pip install google-generativeai
pip install plotly
pip install nltk
pip install scipy
pip install numpy
pip install pandas
```

### 2. Project Structure

```
rag-system/
├── rag_app.py              # Streamlit interface
├── rag_gemini.py       # RAG pipeline
├── data/               # Document storage
│   └── documents.txt   # Auto-generated
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### 3. API Key Setup

Set your Google Gemini API key in `rag_gemini.py`:

```python
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY_HERE'
```

You can get a free API key from Google AI Studio: https://makersuite.google.com/

## 📋 Requirements File

Save the following content as `requirements.txt`:

```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
duckduckgo-search>=3.9.0
google-generativeai>=0.3.0
plotly>=5.15.0
nltk>=3.8.1
scipy>=1.11.0
numpy>=1.24.3
pandas>=2.0.3
```

## 🎯 Usage

### 1. Launch the Application

```bash
streamlit run rag_app.py
```

### 2. Basic Usage

1. **Question Input**: Enter your question on the main page
2. **Web Search**: Click "🔍 Search Web and Prepare" button
3. **Get Answer**: Click "💬 Get Answer" button
4. **Review Results**: View answer, confidence score, and sources

### 3. Advanced Settings

You can configure the following settings from the sidebar:

- **Web Result Count**: Number of search results (5-20)
- **Document Count to Use**: Top-K retrieval (1-10)
- **Similarity Threshold**: Minimum similarity score (0.1-0.8)
- **Document Chunk Size**: Chunk size (200-500)
- **Maximum Sentence Count**: Sentences per chunk (2-6)

## 🏗️ Architecture

### System Components

1. **Web Scraper**: Web search with DuckDuckGo
2. **Document Processor**: Text cleaning and chunking
3. **Embedding Engine**: Turkish BERT model
4. **Vector Store**: FAISS indexing
5. **Retrieval System**: Semantic search
6. **Re-ranker**: Cross-encoder re-ranking
7. **Generation Engine**: Google Gemini
8. **UI Layer**: Streamlit interface

### Data Flow

```
User Query → Web Search → Document Processing → Embedding → 
FAISS Index → Retrieval → Re-ranking → Gemini → Response
```

## 🔧 Code Structure

### `rag_app.py` - Streamlit Interface
- Modern UI components
- System status monitoring
- Real-time analytics
- User experience optimization

### `rag_gemini.py` - RAG Pipeline
- `ImprovedRAGPipeline`: Base RAG class
- `OptimizedGeminiRAG`: Gemini integration
- `search_web_improved`: Web search function
- Advanced error handling

## 📊 Feature Details

### Smart Document Processing
- HTML tag cleaning
- Turkish sentence segmentation
- Adaptive chunk sizing
- Metadata tracking

### Advanced Retrieval
- Semantic search
- Query expansion
- Hybrid scoring
- Confidence calculation

### UI/UX Features
- Responsive design
- Progress tracking
- Real-time metrics
- Interactive visualizations

## 🎨 Visual Features

- **Gradient header**: Modern appearance
- **Color-coded confidence**: Confidence score coloring
- **Interactive charts**: Plotly charts
- **Status indicators**: System status display
- **Responsive design**: Mobile-friendly

## 📈 Performance Metrics

The system tracks the following metrics:

- **Total Query Count**: Number of processed queries
- **Success Rate**: High-confidence response rate
- **Average Confidence Score**: Response quality
- **System Status**: Pipeline health
- **Document Statistics**: Number of loaded documents

## 🛠️ Customization

### Model Change

To change the embedding model:

```python
self.embedding_model = SentenceTransformer("your-model-name")
```

### Gemini Settings

To adjust generation parameters:

```python
generation_config = {
    "temperature": 0.3,        # Creativity level
    "top_p": 0.95,            # Nucleus sampling
    "top_k": 64,              # Top-K sampling
    "max_output_tokens": 1024  # Maximum response length
}
```

### UI Theme

You can customize CSS styles in the `rag_app.py` file.

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**: Check your Google API key
2. **FAISS Installation**: Install `faiss-cpu` package
3. **NLTK Data**: Punkt tokenizer will be downloaded on first run
4. **Memory Error**: Reduce document count

### Debugging

To increase logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)

## 📞 Contact

Feel free to open an issue for questions or suggestions.

---

**🚀 Experience Turkish question-answering with advanced RAG technology!**

# TR
## 🤖 Gelişmiş RAG Soru-Cevap Sistemi

Web'den bilgi toplayarak yapay zeka ile akıllı cevaplar üreten modern bir RAG (Retrieval-Augmented Generation) sistemi.

## 🌟 Özellikler

### 🔍 Akıllı Web Araması
- DuckDuckGo ile otomatik web araması
- Türkçe optimizasyonu
- Rate limiting ve hata yönetimi

### 🧠 Gelişmiş AI Yanıt Üretimi
- Google Gemini 1.5 Flash entegrasyonu
- Türkçe dil modeli desteği
- Cross-encoder ile yeniden sıralama

### 📊 Modern Streamlit Arayüzü
- Responsive ve kullanıcı dostu tasarım
- Gerçek zamanlı sistem izleme
- Detaylı analitik ve görselleştirme

### 🔧 Gelişmiş Teknik Özellikler
- FAISS vektör indeksleme
- Sentence-BERT embedding
- Akıllı belge parçalama
- Güven skoru hesaplama

## 🚀 Kurulum

### 1. Gereksinimler

```bash
pip install streamlit
pip install sentence-transformers
pip install faiss-cpu
pip install duckduckgo-search
pip install google-generativeai
pip install plotly
pip install nltk
pip install scipy
pip install numpy
pip install pandas
```

### 2. Proje Yapısı

```
rag-system/
├── rag_app.py              # Streamlit arayüzü
├── rag_gemini.py       # RAG pipeline
├── data/               # Belge depolama
│   └── documents.txt   # Otomatik oluşturulur
├── requirements.txt    # Gereksinimler
└── README.md          # Bu dosya
```

### 3. API Anahtarı Ayarları

`rag_gemini.py` dosyasında Google Gemini API anahtarınızı ayarlayın:

```python
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY_HERE'
```

Google AI Studio'dan ücretsiz API anahtarı alabilirsiniz: https://makersuite.google.com/

## 📋 Gereksinimler Dosyası

Aşağıdaki içeriği `requirements.txt` dosyası olarak kaydedin:

```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
duckduckgo-search>=3.9.0
google-generativeai>=0.3.0
plotly>=5.15.0
nltk>=3.8.1
scipy>=1.11.0
numpy>=1.24.3
pandas>=2.0.3
```

## 🎯 Kullanım

### 1. Uygulamayı Başlatın

```bash
streamlit run rag_app.py
```

### 2. Temel Kullanım

1. **Soru Girişi**: Ana sayfada sorunuzu yazın
2. **Web Araması**: "🔍 Web'de Ara ve Hazırla" butonuna tıklayın
3. **Cevap Alın**: "💬 Cevap Al" butonuna tıklayın
4. **Sonuçları İnceleyin**: Cevap, güven skoru ve kaynakları görüntüleyin

### 3. Gelişmiş Ayarlar

Yan menüden aşağıdaki ayarları yapılandırabilirsiniz:

- **Web Sonuç Sayısı**: Arama sonucu sayısı (5-20)
- **Kullanılacak Belge Sayısı**: Top-K retrieval (1-10)
- **Benzerlik Eşiği**: Minimum benzerlik skoru (0.1-0.8)
- **Belge Parça Boyutu**: Chunk size (200-500)
- **Maksimum Cümle Sayısı**: Chunk'taki cümle sayısı (2-6)

## 🏗️ Mimari

### Sistem Bileşenleri

1. **Web Scraper**: DuckDuckGo ile web araması
2. **Document Processor**: Metin temizleme ve parçalama
3. **Embedding Engine**: Türkçe BERT modeli
4. **Vector Store**: FAISS indeksleme
5. **Retrieval System**: Semantik arama
6. **Re-ranker**: Cross-encoder ile yeniden sıralama
7. **Generation Engine**: Google Gemini
8. **UI Layer**: Streamlit arayüzü

### Veri Akışı

```
Kullanıcı Sorusu → Web Araması → Belge İşleme → Embedding → 
FAISS İndeksi → Retrieval → Re-ranking → Gemini → Yanıt
```

## 🔧 Kod Yapısı

### `rag_app.py` - Streamlit Arayüzü
- Modern UI bileşenleri
- Sistem durumu izleme
- Gerçek zamanlı analitik
- Kullanıcı deneyimi optimizasyonu

### `rag_gemini.py` - RAG Pipeline
- `ImprovedRAGPipeline`: Temel RAG sınıfı
- `OptimizedGeminiRAG`: Gemini entegrasyonu
- `search_web_improved`: Web arama fonksiyonu
- Gelişmiş hata yönetimi

## 📊 Özellikler Detayı

### Akıllı Belge İşleme
- HTML tag temizleme
- Türkçe cümle segmentasyonu
- Adaptif chunk boyutlandırma
- Metadata izleme

### Gelişmiş Retrieval
- Semantic search
- Query expansion
- Hybrid scoring
- Confidence calculation

### UI/UX Özellikleri
- Responsive tasarım
- Progress tracking
- Real-time metrics
- Interactive visualizations

## 🎨 Görsel Özellikler

- **Gradient header**: Modern görünüm
- **Color-coded confidence**: Güven skoru renklendirmesi
- **Interactive charts**: Plotly grafikleri
- **Status indicators**: Sistem durumu gösterimi
- **Responsive design**: Mobil uyumlu

## 📈 Performans Metrikleri

Sistem aşağıdaki metrikleri takip eder:

- **Toplam Sorgu Sayısı**: İşlenen sorgu sayısı
- **Başarı Oranı**: Yüksek güvenli yanıt oranı
- **Ortalama Güven Skoru**: Yanıt kalitesi
- **Sistem Durumu**: Pipeline sağlığı
- **Belge İstatistikleri**: Yüklenen belge sayısı

## 🛠️ Özelleştirme

### Model Değişikliği

Embedding modelini değiştirmek için:

```python
self.embedding_model = SentenceTransformer("your-model-name")
```

### Gemini Ayarları

Generation parametrelerini ayarlamak için:

```python
generation_config = {
    "temperature": 0.3,        # Yaratıcılık seviyesi
    "top_p": 0.95,            # Nucleus sampling
    "top_k": 64,              # Top-K sampling
    "max_output_tokens": 1024  # Maksimum yanıt uzunluğu
}
```

### UI Teması

CSS stillerini `rag_app.py` dosyasında özelleştirebilirsiniz.

## 🔍 Sorun Giderme

### Yaygın Sorunlar

1. **API Key Hatası**: Google API anahtarınızı kontrol edin
2. **FAISS Kurulum**: `faiss-cpu` paketini yükleyin
3. **NLTK Data**: İlk çalışmada punkt tokenizer indirilecek
4. **Memory Error**: Belge sayısını azaltın

### Hata Ayıklama

Loglama seviyesini artırmak için:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)

## 📞 İletişim

Sorularınız veya önerileriniz için issue açabilirsiniz.

---

**🚀 Gelişmiş RAG teknolojisi ile Türkçe soru-cevap deneyimini yaşayın!**
