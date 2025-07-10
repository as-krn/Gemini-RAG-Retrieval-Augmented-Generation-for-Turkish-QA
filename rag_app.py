import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from rag_gemini import ImprovedRAGPipeline, search_web_improved

# Streamlit konfigürasyonu
st.set_page_config(
    page_title="Gelişmiş RAG Soru-Cevap Sistemi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .answer-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    
    .source-container {
        background-color: #f1f3f4;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #4285f4;
    }
    
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Session state'i initialize et"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_confidence': 0.0,
            'total_documents': 0
        }

def create_sidebar():
    """Gelişmiş sidebar oluştur"""
    with st.sidebar:
        st.markdown("## ⚙️ Sistem Ayarları")
        
        # Arama parametreleri
        st.markdown("### 🔍 Arama Ayarları")
        num_results = st.slider("Web sonuç sayısı", 5, 20, 10)
        top_k = st.slider("Kullanılacak belge sayısı", 1, 10, 5)
        similarity_threshold = st.slider("Benzerlik eşiği", 0.1, 0.8, 0.2, 0.1)
        
        # Gelişmiş ayarlar
        with st.expander("🔧 Gelişmiş Ayarlar"):
            chunk_size = st.slider("Belge parça boyutu", 200, 500, 300)
            max_sentences = st.slider("Maksimum cümle sayısı", 2, 6, 4)
        
        # Sistem durumu
        st.markdown("### 📊 Sistem Durumu")
        
        # Belge sayısı
        doc_count = 0
        if os.path.exists("data/documents.txt"):
            try:
                with open("data/documents.txt", "r", encoding="utf-8") as f:
                    doc_count = len([line for line in f.readlines() if line.strip()])
            except:
                doc_count = 0
        
        st.metric("Yüklü Ham Belge", doc_count)
        
        # Pipeline durumu
        if st.session_state.rag_pipeline is not None:
            rag = st.session_state.rag_pipeline
            if rag.index is not None:
                st.markdown('<div class="status-success">✅ Sistem Hazır</div>', unsafe_allow_html=True)
                st.metric("Belge Parçası", len(rag.documents))
                st.metric("Embed Boyutu", rag.index.d if rag.index else 0)
            else:
                st.markdown('<div class="status-warning">⚠️ İndeks Hazır Değil</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Pipeline Başlatılmadı</div>', unsafe_allow_html=True)
        
        # Sistem istatistikleri
        stats = st.session_state.system_stats
        st.metric("Toplam Sorgu", stats['total_queries'])
        if stats['total_queries'] > 0:
            success_rate = (stats['successful_queries'] / stats['total_queries']) * 100
            st.metric("Başarı Oranı", f"{success_rate:.1f}%")
            st.metric("Ort. Güven", f"{0.1*stats['avg_confidence']:.1%}")
        
        # Sistem sıfırlama
        if st.button("🔄 Sistemi Sıfırla", type="secondary"):
            st.session_state.rag_pipeline = None
            if os.path.exists("data/documents.txt"):
                os.remove("data/documents.txt")
            st.success("Sistem sıfırlandı!")
            st.rerun()
        
        return {
            'num_results': num_results,
            'top_k': top_k,
            'similarity_threshold': similarity_threshold,
            'chunk_size': chunk_size,
            'max_sentences': max_sentences
        }

def create_main_interface(config):
    """Ana arayüz oluştur"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Gelişmiş RAG Soru-Cevap Sistemi</h1>
        <p>Web'den bilgi toplayarak yapay zeka ile akıllı cevaplar üretir</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline başlatma
    if st.session_state.rag_pipeline is None:
        st.info("🚀 Sistem ilk kez başlatılıyor...")
        try:
            st.session_state.rag_pipeline = ImprovedRAGPipeline()
            st.success("✅ Pipeline başarıyla başlatıldı!")
        except Exception as e:
            st.error(f"❌ Pipeline başlatılamadı: {str(e)}")
            return
    
    rag = st.session_state.rag_pipeline
    
    # Ana sorgu alanı
    st.markdown("## ❓ Sorunuzu Yazın")
    
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "",
            placeholder="Sorunuzu buraya yazın...",
            key="main_query",
            help="Detaylı sorular daha iyi sonuç verir"
        )
    
    
    
    # Aksiyon butonları
    col_search, col_ask, col_history = st.columns(3)
    
    with col_search:
        search_clicked = st.button("🔍 Web'de Ara ve Hazırla", type="primary", use_container_width=True)
    
    with col_ask:
        ask_clicked = st.button("💬 Cevap Al", type="secondary", use_container_width=True)
    
    with col_history:
        history_clicked = st.button("📈 Geçmiş", use_container_width=True)
    
    # Arama işlemi
    if search_clicked:
        if query:
            handle_search(query, config)
        else:
            st.warning("⚠️ Lütfen bir soru yazın.")
    
    # Cevap alma işlemi
    if ask_clicked:
        if query:
            handle_query(query, config)
        else:
            st.warning("⚠️ Lütfen bir soru yazın.")
    
    # Geçmiş görüntüleme
    if history_clicked:
        show_query_history()

def handle_search(query, config):
    """Arama işlemini handle et"""
    rag = st.session_state.rag_pipeline
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Web araması
        status_text.text("🔍 Web'de aranıyor...")
        progress_bar.progress(20)
        
        results = search_web_improved(query, config['num_results'])
        
        if not results:
            st.error("❌ Arama sonucu bulunamadı.")
            return
        
        progress_bar.progress(40)
        status_text.text("📄 Belgeler yükleniyor...")
        
        # 2. Belgeleri yükle
        if not rag.load_documents():
            st.error("❌ Belgeler yüklenemedi.")
            return
        
        progress_bar.progress(70)
        status_text.text("🔧 FAISS indeksi oluşturuluyor...")
        
        # 3. İndeks oluştur
        if not rag.build_faiss_index():
            st.error("❌ İndeks oluşturulamadı.")
            return
        
        progress_bar.progress(100)
        status_text.text("✅ Sistem hazır!")
        
        # Başarı mesajı
        st.success(f"✅ {len(results)} web sonucu işlendi ve {len(rag.documents)} belge parçası oluşturuldu!")
        
        # İstatistikleri güncelle
        st.session_state.system_stats['total_documents'] = len(rag.documents)
        
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Arama sırasında hata: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def handle_query(query, config):
    """Sorgu işlemini handle et"""
    rag = st.session_state.rag_pipeline
    
    if rag.index is None:
        st.warning("⚠️ Önce web'de arama yapın ve sistemi hazırlayın.")
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🧠 Cevap üretiliyor...")
        progress_bar.progress(50)
        
        # Cevap üret
        response = rag.generate_answer(query)
        
        progress_bar.progress(100)
        status_text.text("✅ Cevap hazır!")
        
        # Sonucu kaydet
        st.session_state.last_response = response
        st.session_state.last_query = query
        
        # Geçmişe ekle
        st.session_state.query_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # İstatistikleri güncelle
        stats = st.session_state.system_stats
        stats['total_queries'] += 1
        if response.get('confidence', 0) > 0.3:
            stats['successful_queries'] += 1
        
        # Ortalama güven hesapla
        confidences = [item['response'].get('confidence', 0) for item in st.session_state.query_history]
        stats['avg_confidence'] = np.mean(confidences) if confidences else 0
        
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Cevap üretilirken hata: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def show_results():
    """Sonuçları göster"""
    if not hasattr(st.session_state, 'last_response'):
        return
    
    response = st.session_state.last_response
    query = st.session_state.last_query
    
    st.divider()
    
    # Soru başlığı
    st.markdown(f"### 🔍 Soru: *{query}*")
    
    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    confidence = response.get('confidence', 0)
    
    with col1:
        if confidence > 0.7:
            st.success(f"🟢 Güven: {0.1*confidence:.1%}")
        elif confidence > 0.4:
            st.warning(f"🟡 Güven: {confidence:.1%}")
        else:
            st.error(f"🔴 Güven: {confidence:.1%}")
    
    with col2:
        st.info(f"📊 Kaynak: {len(response.get('sources', []))}")
    
    with col3:
        st.info(f"🔍 Bulunan: {response.get('retrieved_count', 0)}")
    
    with col4:
        answer_length = len(response.get('answer', ''))
        st.info(f"📝 Uzunluk: {answer_length}")
    
    # Cevap
    st.markdown("### 📄 Cevap")
    
    answer = response.get('answer', 'Cevap bulunamadı.')
    
    # Cevap kalitesi analizi
    if confidence > 0.7:
        answer_style = "background-color: #000000; border-left: 4px solid #28a745;"
    elif confidence > 0.4:
        answer_style = "background-color: #000000; border-left: 4px solid #ffc107;"
    else:
        answer_style = "background-color: #000000; border-left: 4px solid #dc3545;"
    
    st.markdown(f"""
    <div style="{answer_style} padding: 20px; border-radius: 10px; margin: 10px 0;">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Kaynaklar
    sources = response.get('sources', [])
    if sources:
        st.markdown("### 🔗 Kaynaklar")
        
        for i, source in enumerate(sources, 1):
            similarity = source.get('similarity', 0)
            
            # Benzerlik rengini belirle
            if similarity > 0.7:
                sim_color = "🟢"
            elif similarity > 0.4:
                sim_color = "🟡"
            else:
                sim_color = "🔴"
            
            with st.expander(f"{sim_color} Kaynak {i} - Benzerlik: {0.1*similarity:.1%}"):
                st.markdown(f"**Metin:** {source.get('text', '')}")
                
                url = source.get('url', '')
                if url and url != "Bilinmeyen kaynak":
                    st.markdown(f"**URL:** [{url}]({url})")
                else:
                    st.markdown("**URL:** Bilinmeyen kaynak")
                
                # Kaynak istatistikleri
                text_length = len(source.get('text', ''))
                st.markdown(f"**Metin Uzunluğu:** {text_length} karakter")
    
    # Aksiyonlar
    st.markdown("### 🔧 Aksiyonlar")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Yeni Soru", use_container_width=True):
            if 'last_response' in st.session_state:
                del st.session_state.last_response
            if 'last_query' in st.session_state:
                del st.session_state.last_query
            st.rerun()
    
    with col2:
        if st.button("📋 Cevabı Kopyala", use_container_width=True):
            st.code(answer, language="text")
    
    with col3:
        if st.button("⭐ Beğen", use_container_width=True):
            st.success("Teşekkürler! Geri bildiriminiz kaydedildi.")
    
    with col4:
        if st.button("👎 Beğenme", use_container_width=True):
            st.info("Geri bildiriminiz kaydedildi. Sistem geliştirilmeye devam edecek.")

def show_query_history():
    """Sorgu geçmişini göster"""
    if not st.session_state.query_history:
        st.info("Henüz sorgu geçmişi yok.")
        return
    
    st.markdown("### 📈 Sorgu Geçmişi")
    
    # Geçmiş istatistikleri
    history = st.session_state.query_history
    
    if len(history) > 0:
        # Güven skoru grafiği
        confidences = [item['response'].get('confidence', 0) for item in history]
        timestamps = [item['timestamp'] for item in history]
        
        fig = px.line(
            x=timestamps, 
            y=confidences,
            title="Güven Skoru Trend",
            labels={'x': 'Zaman', 'y': 'Güven Skoru'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Geçmiş listesi
        for i, item in enumerate(reversed(history[-10:]), 1):  # Son 10 sorgu
            with st.expander(f"Sorgu {i}: {item['query'][:50]}..."):
                st.write(f"**Zaman:** {item['timestamp']}")
                st.write(f"**Soru:** {item['query']}")
                st.write(f"**Cevap:** {item['response'].get('answer', '')[:200]}...")
                st.write(f"**Güven:** {item['response'].get('confidence', 0):.1%}")

def show_document_management():
    """Belge yönetimi bölümü"""
    with st.expander("📄 Belge Yönetimi ve İstatistikler"):
        tab1, tab2, tab3 = st.tabs(["Yüklü Belgeler", "İstatistikler", "Analiz"])
        
        with tab1:
            if os.path.exists("data/documents.txt"):
                with open("data/documents.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        st.text_area("Ham Belgeler", content, height=200)
                        
                        # Belge sayısı
                        doc_count = len([line for line in content.split('\n') if line.strip()])
                        st.metric("Toplam Ham Belge", doc_count)
                        
                        # Belge silme
                        if st.button("🗑️ Tüm Belgeleri Sil"):
                            os.remove("data/documents.txt")
                            st.session_state.rag_pipeline = None
                            st.success("Belgeler silindi!")
                            st.rerun()
                    else:
                        st.info("Henüz belge yüklenmedi.")
            else:
                st.info("Henüz belge yüklenmedi.")
        
        with tab2:
            if st.session_state.rag_pipeline and st.session_state.rag_pipeline.documents:
                docs = st.session_state.rag_pipeline.documents
                doc_lengths = [len(doc) for doc in docs]
                
                # Temel istatistikler
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Toplam Belge Parçası", len(docs))
                    st.metric("Ortalama Uzunluk", f"{np.mean(doc_lengths):.0f} karakter")
                    st.metric("Medyan Uzunluk", f"{np.median(doc_lengths):.0f} karakter")
                
                with col2:
                    st.metric("En Uzun Belge", f"{max(doc_lengths)} karakter")
                    st.metric("En Kısa Belge", f"{min(doc_lengths)} karakter")
                    st.metric("Toplam Karakter", f"{sum(doc_lengths):,}")
                
                # Uzunluk dağılımı histogram
                fig = px.histogram(
                    x=doc_lengths,
                    nbins=20,
                    title="Belge Uzunluk Dağılımı",
                    labels={'x': 'Karakter Sayısı', 'y': 'Belge Sayısı'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Henüz belge yüklenmedi.")
        
        with tab3:
            if st.session_state.rag_pipeline and st.session_state.rag_pipeline.documents:
                st.markdown("#### 📊 Belge İçerik Analizi")
                
                # En uzun ve en kısa belgeleri göster
                docs = st.session_state.rag_pipeline.documents
                doc_lengths = [len(doc) for doc in docs]
                
                longest_idx = np.argmax(doc_lengths)
                shortest_idx = np.argmin(doc_lengths)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**En Uzun Belge:**")
                    st.text_area("", docs[longest_idx][:500] + "...", height=150)
                
                with col2:
                    st.markdown("**En Kısa Belge:**")
                    st.text_area("", docs[shortest_idx], height=150)
                
            else:
                st.info("Henüz belge yüklenmedi.")

def main():
    """Ana uygulama"""
    # Session state'i initialize et
    initialize_session_state()
    
    # Sidebar konfigürasyonu
    config = create_sidebar()
    
    # Ana arayüz
    create_main_interface(config)
    
    # Sonuçları göster
    show_results()
    
    # Belge yönetimi
    show_document_management()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
        🤖 <strong>Gelişmiş RAG Soru-Cevap Sistemi</strong><br>
        Web araması ile desteklenen yapay zeka yanıtları | 
        Türkçe optimizasyonu ile geliştirilmiş
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()