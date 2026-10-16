import re

def fixed_size_chunking(text: str, chunk_size: int) -> list[str]:
    """Splits text into chunks of a fixed character size."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def fixed_size_chunking_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into chunks of a fixed character size with a specified overlap."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
        i += chunk_size - overlap
    return chunks

def basic_sentence_aware_chunking(text: str, max_chunk_size: int) -> list[str]:
    """
    Splits text into sentences and then groups sentences into chunks,
    trying not to break sentences and respecting max_chunk_size.
    Uses a basic regex for sentence splitting (not as robust as NLTK).
    """
    # A very basic sentence splitter for demonstration purposes
    # Splits by common sentence terminators, keeping the terminator with the sentence
    # and also handles multiple spaces between sentences.
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s*', text.strip()) if s.strip()]
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_length = 0

    for sentence in sentences:
        # Estimate length if added to current chunk (including a space separator)
        # Add 1 for a space if it's not the first sentence in the current_chunk_sentences
        potential_length = current_chunk_length + len(sentence) + (1 if current_chunk_sentences else 0)

        if potential_length <= max_chunk_size:
            current_chunk_sentences.append(sentence)
            current_chunk_length = potential_length
        else:
            # If current_chunk_sentences is not empty, save it
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            
            # Start a new chunk with the current sentence
            # If the sentence itself is larger than max_chunk_size, it will form its own chunk
            # and potentially exceed max_chunk_size (a known limitation of simple sentence chunkers)
            current_chunk_sentences = [sentence]
            current_chunk_length = len(sentence)

    # Add any remaining chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks


if __name__ == "__main__":
    # Sample text from the article context (original Turkish).
    # This text discusses LLMs, RAG, and the importance of chunking.
    sample_text = """
    Büyük Dil Modelleri (Large Language Models - LLM) çağında, kullanıcıların karmaşık sorularına doğru ve güncel yanıtlar verebilmek, bu modellerin yeteneklerini gerçek dünya bilgileriyle birleştirmekten geçiyor. Ancak LLM'lerin sınırlı bağlam pencereleri (context window) ve sürekli güncellenmeyen eğitim verileri, bu birleşimi zorlu bir görev haline getiriyor. İşte tam bu noktada, Geri Kazanım Artırılmış Üretim (Retrieval Augmented Generation - RAG) mimarisi devreye giriyor. RAG, bir sorguya yanıt vermeden önce ilgili bilgileri harici bir bilgi tabanından alarak (retrieve) LLM'e sunar ve böylece modelin daha bilinçli ve doğru yanıtlar üretmesini sağlar. Bu sürecin en kritik adımlarından biri ise "parçalama (chunking)" işlemidir. Veri parçalama, büyük metin belgelerini yönetilebilir, anlamlı ve LLM'in bağlam penceresine sığacak küçük parçalara ayırma sanatıdır. Peki, bu parçalama işlemini ne kadar verimli yapıyoruz? Doğru parçalama stratejisi seçimi, RAG sisteminizin performansı üzerinde doğrudan bir etkiye sahiptir. Bu makalede, RAG sistemlerinizin etkinliğini artıracak 5 farklı parçalama stratejisini detaylıca inceleyecek ve her biri için Python'da uygulanabilir kod örnekleri sunacağız.
    """.strip()

    print("Original Text Length:", len(sample_text), "characters\n")

    # --- Strategy 1: Fixed-size chunking ---
    # This method splits the text into chunks of a predefined character length.
    print("--- Fixed-size Chunking (chunk_size=200) ---")
    fixed_chunks = fixed_size_chunking(sample_text, 200)
    for i, chunk in enumerate(fixed_chunks):
        print(f"Chunk {i+1} (len={len(chunk)}): {chunk[:100]}...") # Show first 100 chars
    print(f"Total chunks: {len(fixed_chunks)}\n")

    # --- Strategy 2: Fixed-size chunking with overlap ---
    # Similar to fixed-size, but subsequent chunks share a portion of the previous chunk,
    # which can help maintain context across chunk boundaries.
    print("--- Fixed-size Chunking with Overlap (chunk_size=200, overlap=50) ---")
    overlap_chunks = fixed_size_chunking_with_overlap(sample_text, 200, 50)
    for i, chunk in enumerate(overlap_chunks):
        print(f"Chunk {i+1} (len={len(chunk)}): {chunk[:100]}...") # Show first 100 chars
    print(f"Total chunks: {len(overlap_chunks)}\n")

    # --- Strategy 3: Basic Sentence-aware Chunking ---
    # This method attempts to split text into sentences first, then groups sentences
    # into chunks up to a maximum size, avoiding breaking sentences mid-way.
    print("--- Basic Sentence-aware Chunking (max_chunk_size=250) ---")
    sentence_chunks = basic_sentence_aware_chunking(sample_text, 250)
    for i, chunk in enumerate(sentence_chunks):
        print(f"Chunk {i+1} (len={len(chunk)}): {chunk[:100]}...") # Show first 100 chars
    print(f"Total chunks: {len(sentence_chunks)}\n")

    print("\nNote: The '...' indicates truncated output for brevity. Full chunks are larger.")
    print("      Sentence-aware chunking aims to keep sentences intact, potentially leading to chunks slightly larger than max_chunk_size if a single sentence exceeds it.")
