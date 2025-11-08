from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import re
import unicodedata

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
)


class DocumentPreprocessor:
    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            chunking_strategy: Literal["recursive", "token", "character", "markdown"] = "recursive",
            clean_text: bool = True,
            remove_duplicates: bool = False,
            similarity_threshold: float = 0.95,
            encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.clean_text = clean_text
        self.remove_duplicates = remove_duplicates
        self.similarity_threshold = similarity_threshold
        self.encoding_name = encoding_name
        self._text_splitter = self._get_text_splitter()

    def _get_text_splitter(self):
        if self.chunking_strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name=self.encoding_name,
            )
        elif self.chunking_strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n",
            )
        elif self.chunking_strategy == "markdown":
            headers_to_split = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split,
                strip_headers=False,
            )

    def _clean_document_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_document(self, file_path: str) -> List[Document]:
        file_extension = Path(file_path).suffix.lower()

        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
        }

        loader_class = loaders.get(file_extension)
        if not loader_class:
            raise ValueError(f"Unsupported file format: {file_extension}")

        loader = loader_class(file_path)
        return loader.load()

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        try:
            from datasketch import MinHash
        except ImportError:
            raise ImportError(
                "datasketch is required for deduplication. "
                "Install it with: pip install datasketch"
            )

        def get_minhash(text: str, num_perm: int = 128) -> MinHash:
            m = MinHash(num_perm=num_perm)
            for word in text.split():
                m.update(word.encode('utf8'))
            return m

        unique_docs = []
        seen_hashes = []

        for doc in documents:
            text_hash = get_minhash(doc.page_content)
            is_duplicate = False

            for seen_hash in seen_hashes:
                similarity = text_hash.jaccard(seen_hash)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_hashes.append(text_hash)

        return unique_docs

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        if self.chunking_strategy == "markdown":
            splits = []
            for doc in documents:
                markdown_splits: List[Document] = self._text_splitter.split_text(doc.page_content)
                for split_doc in markdown_splits:
                    splits.append(Document(
                        page_content=split_doc.page_content,
                        metadata={**doc.metadata, **split_doc.metadata}
                    ))

            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = recursive_splitter.split_documents(splits)
        else:
            splits = self._text_splitter.split_documents(documents)

        return splits

    def process_file(
            self,
            file_path: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        documents = self._load_document(file_path)

        if self.clean_text:
            for doc in documents:
                doc.page_content = self._clean_document_text(doc.page_content)

        splits = self._split_documents(documents)

        if metadata:
            for doc in splits:
                doc.metadata.update(metadata)

        for doc in splits:
            doc.metadata.setdefault("source", file_path)
            doc.metadata.setdefault("chunk_size", self.chunk_size)

        if self.remove_duplicates:
            splits = self._deduplicate_documents(splits)

        return splits

    def process_text(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        if self.clean_text:
            text = self._clean_document_text(text)

        if self.chunking_strategy == "markdown":
            markdown_splits: List[Document] = self._text_splitter.split_text(text)
            documents = [
                Document(
                    page_content=split_doc.page_content,
                    metadata={**(metadata or {}), **split_doc.metadata}
                )
                for split_doc in markdown_splits
            ]
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            documents = recursive_splitter.split_documents(documents)
        else:
            splits = self._text_splitter.split_text(text)
            documents = [
                Document(
                    page_content=chunk,
                    metadata=metadata or {}
                )
                for chunk in splits
            ]

        if self.remove_duplicates:
            documents = self._deduplicate_documents(documents)

        return documents

    def process_documents(
            self,
            documents: List[Document]
    ) -> List[Document]:
        if self.clean_text:
            for doc in documents:
                doc.page_content = self._clean_document_text(doc.page_content)

        splits = self._split_documents(documents)

        if self.remove_duplicates:
            splits = self._deduplicate_documents(splits)

        return splits