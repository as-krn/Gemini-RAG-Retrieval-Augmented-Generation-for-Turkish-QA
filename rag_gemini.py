from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict
import time
import math
import logging
import scipy.special
import google.generativeai as genai

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Google Gemini ayarları
GOOGLE_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
genai.configure(api_key=GOOGLE_API_KEY)

def upload_to_gemini(path, mime_type=None):
    """Gemini'ye dosya yükleme fonksiyonu"""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Dosya yüklendi: '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"Dosya yükleme hatası: {str(e)}")
        return None

def wait_for_files_active(files):
    """Dosya işleme tamamlanana kadar bekle"""
    print("Dosya işleniyor...")
    for file in files:
        if file is None:
            continue
        retry_count = 0
        max_retries = 20
        while file.state.name == "PROCESSING" and retry_count < max_retries:
            print(".", end="", flush=True)
            time.sleep(3)
            try:
                file = genai.get_file(file.name)
                retry_count += 1
            except Exception as e:
                print(f"Dosya durumu kontrol hatası: {e}")
                break
        if file.state.name != "ACTIVE":
            print(f"Dosya {file.name} işlenemedi, durum: {file.state.name}")
        else:
            print(f"Dosya {file.name} hazır")
    print("\nTüm dosyalar hazır")

# Gelişmiş sistem talimatları
system_instruction = """
Sen Türkçe konuşan bir AI asistanısın. Kullanıcının sorularını verilen bağlam ve belgeler doğrultusunda yanıtla.

KURALLARIN:
1. Sadece verilen bağlam ve belgelerden yararlan
2. Eğer bağlamda yeterli bilgi yoksa bunu açıkça belirt
3. Yanıtlarını kısa, net ve anlaşılır tut
4. Türkçe dilbilgisi kurallarına uy
5. Kaynak belirtmeyi unutma
6. Spekülasyon yapma, sadece belgelerden elde ettiğin bilgileri kullan

YANIT FORMATI:
- Doğrudan cevap ver
- Gerekirse madde madde listele
- Kaynakları belirt
- Belirsizlik varsa bunu söyle
"""

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

class ImprovedRAGPipeline:
    def __init__(self, docs_path="data/documents.txt"):
        self.docs_path = docs_path
        self.embedding_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        self.re_ranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        self.documents = []
        self.document_metadata = []
        self.index = None
        self.doc_embeddings = None
        
        # Logging kurulumu
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_documents(self) -> bool:
        """Belgeleri yükle ve işle"""
        if not os.path.exists(self.docs_path):
            self.logger.warning(f"Dosya bulunamadı: {self.docs_path}")
            return False
        
        try:
            with open(self.docs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            self.documents, self.document_metadata = [], []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # URL ayrıştırma
                if " ||| " in line:
                    content, url = line.split(" ||| ", 1)
                    url = url.strip("() ")
                else:
                    content = line
                    url = "Bilinmeyen kaynak"
                
                # Metin temizleme
                cleaned = self._clean_text(content)
                if len(cleaned) > 50:
                    # Metin parçalara ayırma
                    chunks = self._split_into_chunks(cleaned, max_sentences=5)
                    for chunk in chunks:
                        self.documents.append(chunk)
                        self.document_metadata.append({
                            "url": url, 
                            "length": len(chunk),
                            "original_content": content[:200] + "..." if len(content) > 200 else content
                        })
            
            self.logger.info(f"{len(self.documents)} belge parçası yüklendi.")
            return True
            
        except Exception as e:
            self.logger.error(f"Belge yükleme hatası: {str(e)}")
            return False

    def _split_into_chunks(self, text: str, max_sentences: int = 5) -> List[str]:
        """Metni cümle bazında parçalara ayır"""
        try:
            sentences = sent_tokenize(text, language='turkish')
            chunks = []
            for i in range(0, len(sentences), max_sentences):
                chunk = " ".join(sentences[i:i+max_sentences])
                if len(chunk.strip()) > 30:  # Çok kısa parçaları filtrele
                    chunks.append(chunk)
            return chunks
        except Exception as e:
            self.logger.error(f"Metin parçalama hatası: {str(e)}")
            return [text]

    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        # HTML etiketlerini kaldır
        text = re.sub(r'<[^>]+>', '', text)
        # Fazla boşlukları düzelt
        text = re.sub(r'\s+', ' ', text)
        # Sadece istenen karakterleri tut
        text = re.sub(r'[^\w\s.,!?;:()\-üğıöçşÜĞIÖÇŞ]', '', text)
        return text.strip()

    def build_faiss_index(self) -> bool:
        """FAISS indeksi oluştur"""
        if not self.documents:
            self.logger.warning("Yüklenecek doküman yok.")
            return False
        
        try:
            self.logger.info("Embedding'ler hesaplanıyor...")
            embeddings = self.embedding_model.encode(
                self.documents, 
                convert_to_numpy=True, 
                show_progress_bar=True, 
                batch_size=32
            )
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # FAISS indeks oluştur
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            self.doc_embeddings = embeddings
            
            self.logger.info("FAISS indeksi başarıyla oluşturuldu.")
            return True
            
        except Exception as e:
            self.logger.error(f"İndeks oluşturma hatası: {str(e)}")
            return False

    def expand_query(self, query: str) -> str:
        """Sorguyu genişlet"""
        replacements = {
            "yapay zeka": "yapay zeka artificial intelligence AI",
            "makine öğrenmesi": "makine öğrenmesi machine learning ML",
            "veri bilimi": "veri bilimi data science",
            "derin öğrenme": "derin öğrenme deep learning",
            "algoritma": "algoritma algorithm",
            "model": "model algoritma sistem"
        }
        
        query_lower = query.lower()
        for key, val in replacements.items():
            if key in query_lower:
                query = query.replace(key, val)
        
        return query

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """İlgili belgeleri getir"""
        if self.index is None:
            self.logger.warning("FAISS indeksi yok.")
            return []
        
        try:
            # Sorguyu genişlet
            expanded_query = self.expand_query(query)
            
            # Query embedding'ini hesapla
            query_vec = self.embedding_model.encode([expanded_query], convert_to_numpy=True)
            faiss.normalize_L2(query_vec)
            
            # Arama yap
            similarities, indices = self.index.search(query_vec, top_k)
            
            # Sonuçları düzenle
            docs = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0 and sim > 0.1:  # Minimum benzerlik eşiği
                    doc_content = self.documents[idx]
                    doc_url = self.document_metadata[idx]["url"]
                    docs.append((doc_content, float(sim), doc_url))
            
            # Re-ranking uygula
            if docs:
                try:
                    rerank_pairs = [(query, doc) for doc, _, _ in docs]
                    rerank_scores = self.re_ranker.predict(rerank_pairs)
                    
                    # Skorları güncelleyerek yeniden sırala
                    docs = [(doc, float(score), url) for (doc, _, url), score in zip(docs, rerank_scores)]
                    docs.sort(key=lambda x: x[1], reverse=True)
                    
                except Exception as e:
                    self.logger.warning(f"Re-ranking hatası: {str(e)}")
            
            return docs
            
        except Exception as e:
            self.logger.error(f"Retrieval hatası: {str(e)}")
            return []

class OptimizedGeminiRAG(ImprovedRAGPipeline):
    def __init__(self, docs_path="data/documents.txt", pdf_path=None):
        super().__init__(docs_path)
        self.pdf_path = pdf_path
        self.gemini_file = None
        self.chat_session = None
        self.conversation_history = []
        self.api_call_count = 0
        self.max_api_calls_per_minute = 60

    def prepare_gemini(self):
        """Gemini'yi hazırla"""
        try:
            history = []
            
            # PDF varsa yükle
            if self.pdf_path and os.path.exists(self.pdf_path):
                self.logger.info(f"PDF yükleniyor: {self.pdf_path}")
                self.gemini_file = upload_to_gemini(self.pdf_path, mime_type="application/pdf")
                if self.gemini_file:
                    wait_for_files_active([self.gemini_file])
                    history.append({
                        "role": "user",
                        "parts": [self.gemini_file],
                    })
            
            # Chat session başlat
            self.chat_session = model.start_chat(history=history)
            self.logger.info("Gemini hazırlandı.")
            
        except Exception as e:
            self.logger.error(f"Gemini hazırlama hatası: {str(e)}")
            raise

    def _build_context(self, retrieved_docs: List[Tuple[str, float, str]], max_context_length: int = 3000) -> str:
        """Retrieved belgelerden context oluştur"""
        context_parts = []
        total_length = 0
        
        for i, (doc, score, url) in enumerate(retrieved_docs):
            if total_length + len(doc) > max_context_length:
                break
            
            context_parts.append(f"BELGE {i+1} (Kaynak: {url}):\n{doc}\n")
            total_length += len(doc)
        
        return "\n".join(context_parts)

    def _create_enhanced_prompt(self, query: str, context: str) -> str:
        """Gelişmiş prompt oluştur"""
        prompt = f"""
BAĞLAM BİLGİLERİ:
{context}

KULLANICI SORUSU: {query}

GÖREV:
Yukarıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu yanıtla.

KURALLARIN:
1. Sadece verilen bağlam bilgilerini kullan
2. Bağlamda yeterli bilgi yoksa bunu açıkça belirt
3. Yanıtını kısa ve öz tut (maksimum 3-4 paragraf)
4. Hangi belgeden hangi bilgiyi aldığını belirt
5. Emin olmadığın konularda spekülasyon yapma

YANIT:
"""
        return prompt

    def _rate_limit_check(self):
        """API rate limit kontrolü"""
        if self.api_call_count >= self.max_api_calls_per_minute:
            self.logger.info("Rate limit reached, waiting...")
            time.sleep(60)
            self.api_call_count = 0

    def _fallback_response(self, query: str) -> Dict[str, any]:
        """Fallback yanıt"""
        return {
            "answer": "Üzgünüm, bu soruya yanıt veremiyorum. Lütfen daha sonra tekrar deneyin.",
            "sources": [],
            "confidence": 0.0,
            "retrieved_count": 0,
            "method": "fallback"
        }

    def _validate_response(self, response_text: str, retrieved_docs: List[Tuple[str, float, str]]) -> float:
        """Yanıt kalitesini değerlendir"""
        if not response_text or len(response_text.strip()) < 20:
            return 0.1
        
        # Kaynak belgelerdeki kelimelerin yanıtta geçme oranı
        if not retrieved_docs:
            return 0.5
        
        try:
            doc_words = set()
            for doc, _, _ in retrieved_docs:
                doc_words.update(doc.lower().split())
            
            response_words = set(response_text.lower().split())
            overlap = len(doc_words.intersection(response_words))
            
            if len(doc_words) > 0:
                relevance_score = min(overlap / len(doc_words), 1.0)
                return max(relevance_score, 0.3)
            
        except Exception as e:
            self.logger.warning(f"Yanıt validation hatası: {str(e)}")
        
        return 0.5

    def generate_answer(self, query: str, use_retrieval: bool = True) -> Dict[str, any]:
        """Gemini kullanarak yanıt üret"""
        if self.chat_session is None:
            try:
                self.prepare_gemini()
            except Exception as e:
                self.logger.error(f"Gemini hazırlanamadı: {str(e)}")
                return self._fallback_response(query)

        try:
            # Rate limit kontrolü
            self._rate_limit_check()
            
            retrieved_docs = []
            context = ""
            
            # Belge retrieval kullan
            if use_retrieval:
                retrieved_docs = self.retrieve(query, top_k=5)
                if retrieved_docs:
                    context = self._build_context(retrieved_docs)
                    self.logger.info(f"{len(retrieved_docs)} belge getirildi.")
                else:
                    self.logger.warning("Relevant belge bulunamadı.")
            
            # Prompt oluştur
            if context:
                prompt = self._create_enhanced_prompt(query, context)
            else:
                prompt = f"""
KULLANICI SORUSU: {query}

Eğer bu soruyla ilgili bilginiz varsa yanıtlayın. Yoksa "Bu konuda yeterli bilgim yok" deyin.
Yanıtınızı kısa ve öz tutun.
"""

            # Gemini'ye gönder
            self.api_call_count += 1
            response = self.chat_session.send_message(prompt)
            answer_text = response.text.strip()
            
            # Yanıt validation
            confidence = self._validate_response(answer_text, retrieved_docs)
            
            # Yanıt yapılandır
            sources = []
            for doc, score, url in retrieved_docs[:3]:  # İlk 3 kaynağı göster
                sources.append({
                    "text": doc[:200] + "..." if len(doc) > 200 else doc,
                    "url": url,
                    "similarity": round(score, 3)
                })
            
            result = {
                "answer": answer_text if answer_text.endswith('.') else answer_text + ".",
                "sources": sources,
                "confidence": confidence,
                "retrieved_count": len(retrieved_docs),
                "method": "gemini_with_retrieval" if use_retrieval else "gemini_only"
            }
            
            # Conversation history güncelle
            self.conversation_history.append({
                "query": query,
                "response": answer_text,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini yanıt üretme hatası: {str(e)}")
            return self._fallback_response(query)

    def get_conversation_summary(self) -> str:
        """Konuşma özetini al"""
        if not self.conversation_history:
            return "Henüz konuşma geçmişi yok."
        
        summary = f"Toplam {len(self.conversation_history)} soru-cevap:\n\n"
        for i, conv in enumerate(self.conversation_history[-5:], 1):  # Son 5 konuşma
            summary += f"{i}. S: {conv['query'][:100]}...\n"
            summary += f"   C: {conv['response'][:100]}...\n\n"
        
        return summary

def search_web_improved(query: str, num_results: int = 10, save_path: str = "data/documents.txt") -> List[Tuple[str, str]]:
    """Gelişmiş web arama"""
    results = []
    max_retries = 3
    retry_wait = 5

    print(f"Web araması: '{query}'")

    for attempt in range(1, max_retries + 1):
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                
                for r in search_results:
                    snippet = r.get('body', '').replace('\n', ' ').strip()
                    url = r.get('href', '')
                    title = r.get('title', '')

                    full_content = f"{title}. {snippet}" if title else snippet

                    if len(full_content) > 100:
                        results.append((full_content, url))

            if results:
                # Sonuçları kaydet
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    for content, url in results:
                        f.write(f"{content} ||| {url}\n")
                print(f"{len(results)} sonuç kaydedildi.")
                return results

        except Exception as e:
            print(f"[Deneme {attempt}] Hata: {str(e)}")
            if "Ratelimit" in str(e) and attempt < max_retries:
                print(f"{retry_wait} saniye bekleniyor...")
                time.sleep(retry_wait)
                retry_wait *= 2  # Exponential backoff
            else:
                break

    print("Web araması başarısız.")
    return []

def main():
    """Ana fonksiyon"""
    # RAG sistemini başlat
    rag = OptimizedGeminiRAG(docs_path="data/documents.txt")
    
    # Test sorguları
    test_queries = [
        "Yapay zeka nedir ve nasıl çalışır?",
        "Makine öğrenmesi algoritmaları nelerdir?",
        "Derin öğrenme ile geleneksel makine öğrenmesi arasındaki fark nedir?",
        "Doğal dil işleme nedir?"
    ]
    
    print("=== GELİŞMİŞ GEMİNİ RAG SİSTEMİ ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print('='*60)
        
        # Web araması yap
        search_results = search_web_improved(query, num_results=8)
        
        if search_results:
            # Belgeleri yükle ve indeks oluştur
            if rag.load_documents() and rag.build_faiss_index():
                # Yanıt üret
                response = rag.generate_answer(query)
                
                print(f"\n📝 YANIT:")
                print(f"{response['answer']}\n")
                
                print(f"📊 BİLGİLER:")
                print(f"• Güven skoru: {response['confidence']:.2f}")
                print(f"• Bulunan belge: {response['retrieved_count']}")
                print(f"• Yöntem: {response['method']}")
                
                if response['sources']:
                    print(f"\n📚 KAYNAKLAR:")
                    for j, source in enumerate(response['sources'], 1):
                        print(f"{j}. {source['url']} (Benzerlik: {source['similarity']})")
                        print(f"   {source['text'][:100]}...")
                
            else:
                print("❌ Belge yükleme veya indeks oluşturma hatası")
        else:
            print("❌ Web araması başarısız")
        
        print(f"\n{'='*60}")
        time.sleep(2)  # Rate limiting için bekle
    
    # Konuşma özetini göster
    print(f"\n📋 KONUŞMA ÖZETİ:")
    print(rag.get_conversation_summary())

if __name__ == "__main__":
    main()