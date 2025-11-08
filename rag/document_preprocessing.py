# Complete RAG Document Preprocessing Python Examples
# Best practices for preparing documents for Retrieval-Augmented Generation systems

# =============================================================================
# 1. DOCUMENT LOADING AND BASIC PREPROCESSING
# =============================================================================

import re
import unicodedata
from typing import List
import pandas as pd


# Basic text cleaning function
def clean_text(text: str) -> str:
    """
    Remove noise and normalize text for RAG systems.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    # Normalize unicode characters (handles special characters, accents, etc.)
    text = unicodedata.normalize('NFKD', text)

    # Remove URLs
    text = re.sub(r'http\\S+|www\\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\\S+@\\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\\s.!?\\-]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()

    return text


def remove_headers_footers(text: str) -> str:
    """
    Remove common headers, footers, and page numbers from PDFs.
    """
    # Remove page numbers (formats like "Page 1", "1.", etc.)
    text = re.sub(r'\\b(?:Page|page)?\\s*\\d+\\b', '', text)

    # Remove common headers/footers patterns
    text = re.sub(r'^.*?(?:CONFIDENTIAL|PROPRIETARY|©).*?$', '',
                  text, flags=re.MULTILINE)

    return text


# Example usage
raw_text = """
Visit https://example.com for more info. 
Contact us at info@example.com.
This is important content. © 2024
Page 1
"""
cleaned = clean_text(raw_text)
print("Cleaned text:", cleaned)

# =============================================================================
# 2. TEXT SPLITTING AND CHUNKING STRATEGIES
# =============================================================================

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter
)


def fixed_size_chunking(text: str, chunk_size: int = 500,
                        chunk_overlap: int = 50) -> List[str]:
    """
    Simple fixed-size chunking with overlap.
    Best for: Quick prototyping, unstructured text

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk (characters)
        chunk_overlap: Overlap between chunks (characters)

    Returns:
        List of text chunks
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "
    )
    chunks = splitter.split_text(text)
    return chunks


def recursive_chunking(text: str, chunk_size: int = 1000,
                       chunk_overlap: int = 100) -> List[str]:
    """
    Recursive chunking respects document structure.
    Best for: Mixed document types, preserving semantics

    Splits by multiple separators in order:
    1. Paragraphs (\\n\\n)
    2. Sentences (\\n)
    3. Words ( )
    4. Characters (fallback)

    Args:
        text: Input text to chunk
        chunk_size: Maximum chunk size (characters)
        chunk_overlap: Overlap between chunks (characters)

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def token_based_chunking(text: str, chunk_size: int = 256) -> List[str]:
    """
    Split text based on token count rather than characters.
    Best for: Ensuring chunks fit within LLM context windows

    Args:
        text: Input text to chunk
        chunk_size: Size in tokens

    Returns:
        List of text chunks
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        encoding_name="cl100k_base"  # For OpenAI models
    )
    chunks = splitter.split_text(text)
    return chunks


def markdown_aware_chunking(markdown_text: str) -> List[dict]:
    """
    Split markdown while preserving header hierarchy metadata.
    Best for: Documentation, markdown files

    Args:
        markdown_text: Markdown formatted text

    Returns:
        List of chunks with metadata
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    splits = splitter.split_text(markdown_text)

    # Convert to dict format
    return [
        {
            "content": split.page_content,
            "metadata": split.metadata
        }
        for split in splits
    ]


# Example usage
sample_text = """
Paragraph 1. This is the first paragraph with multiple sentences.
It contains important information that should stay together.

Paragraph 2. This is the second paragraph. It has different content.
Understanding the structure is critical for RAG systems.
"""

recursive_chunks = recursive_chunking(sample_text, chunk_size=100, chunk_overlap=20)
print("\\nRecursive chunks:")
for i, chunk in enumerate(recursive_chunks):
    print(f"Chunk {i + 1}: {chunk[:50]}...")


# =============================================================================
# 3. SEMANTIC CHUNKING (Advanced)
# =============================================================================

def semantic_chunking(text: str, threshold: float = 0.75) -> List[str]:
    """
    Split text into semantically coherent chunks using embeddings.
    Best for: High-quality retrieval, domain-specific content

    Uses sentence-level similarity to determine chunk boundaries.

    Args:
        text: Input text to chunk
        threshold: Cosine similarity threshold (0-1)

    Returns:
        List of semantically meaningful chunks
    """
    try:
        import spacy
        from sentence_transformers import SentenceTransformer, util

        nlp = spacy.load("en_core_web_sm")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Segment into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents
                     if sent.text.strip()]

        chunks = []
        current_chunk_sentences = []
        current_chunk_embedding = None

        for sentence in sentences:
            # Generate embedding for current sentence
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)

            if current_chunk_embedding is None:
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_embedding
            else:
                # Compute similarity
                sim_score = util.cos_sim(sentence_embedding,
                                         current_chunk_embedding)

                if sim_score.item() >= threshold:
                    # Add to current chunk and update embedding
                    current_chunk_sentences.append(sentence)
                    num_sents = len(current_chunk_sentences)
                    current_chunk_embedding = (
                                                      (current_chunk_embedding * (num_sents - 1)) +
                                                      sentence_embedding
                                              ) / num_sents
                else:
                    # Finalize chunk and start new one
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentence]
                    current_chunk_embedding = sentence_embedding

        # Add final chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    except ImportError:
        print("Install: pip install spacy sentence-transformers")
        return []


# =============================================================================
# 4. METADATA EXTRACTION AND PRESERVATION
# =============================================================================

from langchain.schema import Document


def add_metadata_to_chunks(chunks: List[str],
                           source: str,
                           doc_type: str = "text",
                           author: str = None) -> List[Document]:
    """
    Convert chunks to LangChain Document objects with metadata.
    Metadata helps with filtering and ranking in retrieval.

    Args:
        chunks: List of text chunks
        source: Document source (filename, URL, etc.)
        doc_type: Type of document (pdf, markdown, web, etc.)
        author: Document author if available

    Returns:
        List of Document objects with metadata
    """
    documents = []

    for idx, chunk in enumerate(chunks):
        metadata = {
            "source": source,
            "chunk_id": idx,
            "doc_type": doc_type,
            "chunk_length": len(chunk),
            "char_count": len(chunk),
        }

        if author:
            metadata["author"] = author

        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)

    return documents


# Example: Processing PDF with metadata
def process_pdf_document(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
    """
    Complete pipeline: Load -> Clean -> Chunk -> Add Metadata
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Install: pip install PyPDF2")
        return []

    # Extract text from PDF
    extracted_text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted_text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    # Clean text
    cleaned_text = clean_text(extracted_text)
    cleaned_text = remove_headers_footers(cleaned_text)

    # Chunk text
    chunks = recursive_chunking(cleaned_text, chunk_size=chunk_size)

    # Add metadata
    documents = add_metadata_to_chunks(
        chunks,
        source=pdf_path,
        doc_type="pdf"
    )

    return documents


# =============================================================================
# 5. DEDUPLICATION (Removing Similar/Duplicate Content)
# =============================================================================

def remove_duplicate_chunks(documents: List[Document],
                            similarity_threshold: float = 0.95) -> List[Document]:
    """
    Remove near-duplicate chunks using SimHash.
    Reduces redundant content and improves retrieval efficiency.

    Args:
        documents: List of Document objects
        similarity_threshold: Threshold for considering documents duplicates

    Returns:
        Deduplicated list of Document objects
    """
    try:
        from datasketch import MinHash
    except ImportError:
        print("Install: pip install datasketch")
        return documents

    def get_minhash(text: str, num_perm: int = 128) -> MinHash:
        """Generate MinHash for text"""
        m = MinHash(num_perm=num_perm)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    unique_documents = []
    seen_hashes = []

    for doc in documents:
        text_hash = get_minhash(doc.page_content)
        is_duplicate = False

        for seen_hash in seen_hashes:
            # Calculate Jaccard similarity
            similarity = text_hash.jaccard(seen_hash)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_documents.append(doc)
            seen_hashes.append(text_hash)

    return unique_documents


# =============================================================================
# 6. COMPLETE RAG PREPROCESSING PIPELINE
# =============================================================================

class RAGPreprocessor:
    """
    Complete preprocessing pipeline for RAG systems.
    Handles loading, cleaning, chunking, and metadata extraction.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 chunking_strategy: str = "recursive"):
        """
        Initialize preprocessor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: 'fixed', 'recursive', 'token', or 'semantic'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

    def process_text(self, text: str, source: str) -> List[Document]:
        """Process raw text through complete pipeline"""
        # Step 1: Clean
        text = clean_text(text)
        text = remove_headers_footers(text)

        # Step 2: Chunk based on strategy
        if self.chunking_strategy == "recursive":
            chunks = recursive_chunking(text, self.chunk_size, self.chunk_overlap)
        elif self.chunking_strategy == "token":
            chunks = token_based_chunking(text, self.chunk_size)
        elif self.chunking_strategy == "semantic":
            chunks = semantic_chunking(text)
        else:  # fixed
            chunks = fixed_size_chunking(text, self.chunk_size, self.chunk_overlap)

        # Step 3: Add metadata
        documents = add_metadata_to_chunks(chunks, source=source)

        # Step 4: Deduplicate (optional)
        documents = remove_duplicate_chunks(documents, similarity_threshold=0.9)

        return documents


# Example usage
preprocessor = RAGPreprocessor(
    chunk_size=800,
    chunk_overlap=80,
    chunking_strategy="recursive"
)

sample_document = """
Introduction. This is a sample document for RAG preprocessing.
It contains multiple paragraphs and sections.

Main Content. The main content section explains key concepts.
Understanding preprocessing is crucial for RAG performance.

Conclusion. RAG systems require careful document preparation.
Proper chunking and metadata enhance retrieval accuracy.
"""

processed_docs = preprocessor.process_text(
    sample_document,
    source="sample_doc.txt"
)

print(f"\\nProcessed {len(processed_docs)} documents")
for i, doc in enumerate(processed_docs):
    print(f"\\nDoc {i + 1}:")
    print(f"  Content preview: {doc.page_content[:60]}...")
    print(f"  Metadata: {doc.metadata}")


# =============================================================================
# 7. EMBEDDING AND VECTOR DATABASE STORAGE
# =============================================================================

def create_vector_store(documents: List[Document]):
    """
    Create and store embeddings in vector database.
    Uses LangChain with Chroma or other backends.
    """
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma

        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Persist to disk
        vector_store.persist()

        return vector_store

    except ImportError:
        print("Install: pip install langchain chromadb sentence-transformers")
        return None


# =============================================================================
# Usage Example: Complete RAG Pipeline
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG DOCUMENT PREPROCESSING EXAMPLES")
    print("=" * 70)

    # Create sample document
    sample_doc = """
    Artificial Intelligence in Healthcare.

    Introduction. AI is transforming healthcare through various applications.
    Machine learning models can detect diseases early and improve treatment outcomes.

    Clinical Applications. AI systems assist in diagnosis, drug discovery, and 
    personalized medicine. Computer vision helps analyze medical imaging. 
    Natural language processing extracts insights from medical records.

    Challenges and Considerations. Privacy and data security remain critical concerns.
    Regulatory compliance and ethical considerations must guide AI deployment.
    Ensuring model interpretability builds clinician trust and adoption.

    Conclusion. The integration of AI in healthcare continues to evolve.
    Proper implementation requires collaboration between technologists and 
    healthcare professionals to maximize patient benefits.
    """

    # Process using the preprocessor
    preprocessor = RAGPreprocessor(
        chunk_size=300,
        chunk_overlap=50,
        chunking_strategy="recursive"
    )

    documents = preprocessor.process_text(sample_doc, source="healthcare_ai.txt")

    print(f"\\nTotal documents created: {len(documents)}")
    print("\\nDocument chunks and metadata:")

    for i, doc in enumerate(documents, 1):
        print(f"\\n--- Chunk {i} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")