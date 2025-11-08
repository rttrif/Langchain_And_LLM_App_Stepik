from typing import List, Optional, Dict, Any, Literal, Tuple
from pathlib import Path
import re
import unicodedata
import logging
import hashlib

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
    """
    Препроцессор документов для RAG систем с поддержкой различных форматов и стратегий чанкинга.

    Функциональность:
    - Загрузка документов (PDF, DOCX, TXT, MD, CSV, HTML, XLSX)
    - Очистка текста (удаление URL, email, HTML тегов)
    - Разбиение на чанки (recursive, token, character, markdown)
    - Обогащение метаданных (source, page, topic, date, language)
    - Дедупликация документов (exact hash, minhash)
    - Специальная обработка таблиц (сохранение, слияние фрагментов)
    - Логирование метаданных и статистики
    """

    def __init__(
            self,
            file_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 100,
            chunking_strategy: Literal["recursive", "token", "character", "markdown"] = "recursive",
            clean_text: bool = True,
            remove_duplicates: bool = False,
            deduplication_method: Literal["exact", "minhash"] = "exact",
            similarity_threshold: float = 0.95,
            encoding_name: str = "cl100k_base",
            doc_language: str = "en",
            log_metadata_sample: bool = False,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.clean_text = clean_text
        self.remove_duplicates = remove_duplicates
        self.deduplication_method = deduplication_method
        self.similarity_threshold = similarity_threshold
        self.encoding_name = encoding_name
        self.doc_language = doc_language
        self.log_metadata_sample = log_metadata_sample
        self._text_splitter = self._get_text_splitter()
        self.logger = logging.getLogger(__name__)

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

    def _is_table_fragment(self, text: str) -> bool:
        return bool(re.search(r'\|.*\|', text)) or \
            bool(re.search(r'\t{2,}', text)) or \
            bool(re.search(r'^\s*[\w\s]+\s{2,}[\w\s]+', text))

    def _deduplicate_exact(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, int]]:
        unique_texts = set()
        unique_docs = []
        stats = {'duplicates': 0, 'tables_preserved': 0}

        for doc in documents:
            text = doc.page_content.strip()
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

            if text_hash not in unique_texts:
                unique_texts.add(text_hash)
                unique_docs.append(doc)
                if self._is_table_fragment(text):
                    stats['tables_preserved'] += 1
            else:
                stats['duplicates'] += 1

        return unique_docs, stats

    def _deduplicate_minhash(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, int]]:
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            raise ImportError(
                "MinHash deduplication requires 'datasketch' package. "
                "Install it with: pip install datasketch"
            )

        def create_minhash(text: str, num_perm: int = 128) -> MinHash:
            mh = MinHash(num_perm=num_perm)
            for word in text.lower().split():
                mh.update(word.encode('utf-8'))
            return mh

        unique_docs = []
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        stats = {'duplicates': 0, 'tables_preserved': 0}

        for idx, doc in enumerate(documents):
            text = doc.page_content.strip()
            minhash = create_minhash(text)
            similar = lsh.query(minhash)

            if not similar:
                lsh.insert(f"doc_{idx}", minhash)
                unique_docs.append(doc)
                if self._is_table_fragment(text):
                    stats['tables_preserved'] += 1
            else:
                stats['duplicates'] += 1

        return unique_docs, stats

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        if self.deduplication_method == "exact":
            unique_docs, stats = self._deduplicate_exact(documents)
        else:
            unique_docs, stats = self._deduplicate_minhash(documents)

        self.logger.info(f"🔍 Дедупликация ({self.deduplication_method}):")
        self.logger.info(f"   Всего чанков: {len(documents)}")
        self.logger.info(f"   Дубликатов: {stats['duplicates']}")
        self.logger.info(f"   Сохранено таблиц: {stats['tables_preserved']}")
        self.logger.info(f"   Осталось: {len(unique_docs)} чанков")

        return unique_docs

    def merge_table_fragments(self, documents: List[Document]) -> List[Document]:
        merged = []
        current_table = []

        for doc in documents:
            if self._is_table_fragment(doc.page_content):
                current_table.append(doc)
            else:
                if current_table:
                    merged_doc = Document(
                        page_content="\n".join([d.page_content for d in current_table]),
                        metadata=current_table[0].metadata
                    )
                    merged_doc.metadata['content_type'] = 'table'
                    merged_doc.metadata['merged_fragments'] = len(current_table)
                    merged.append(merged_doc)
                    current_table = []
                merged.append(doc)

        if current_table:
            merged_doc = Document(
                page_content="\n".join([d.page_content for d in current_table]),
                metadata=current_table[0].metadata
            )
            merged_doc.metadata['content_type'] = 'table'
            merged_doc.metadata['merged_fragments'] = len(current_table)
            merged.append(merged_doc)

        if len(merged) < len(documents):
            self.logger.info(f"🔗 Объединено фрагментов таблиц: {len(documents) - len(merged)}")

        return merged

    def _log_metadata_sample(self, documents: List[Document]) -> None:
        if not self.log_metadata_sample or not documents:
            return

        self.logger.info(f"Total documents: {len(documents)}")
        self.logger.info("=" * 80)
        self.logger.info("FIRST DOCUMENT:")
        self.logger.info(f"Content preview: {documents[0].page_content[:200]}...")
        self.logger.info(f"Metadata: {documents[0].metadata}")
        self.logger.info("=" * 80)

        if len(documents) > 1:
            self.logger.info("LAST DOCUMENT:")
            self.logger.info(f"Content preview: {documents[-1].page_content[:200]}...")
            self.logger.info(f"Metadata: {documents[-1].metadata}")
            self.logger.info("=" * 80)

    def add_table_metadata(
            self,
            doc: Document,
            table_info: Dict[str, Any]
    ) -> Document:
        doc.metadata.update({
            'content_type': 'table',
            'table_name': table_info.get('title', 'Unknown'),
            'columns': list(table_info.get('columns', [])),
            'row_count': table_info.get('rows', 0),
            'table_context': table_info.get('preceding_text', ''),
        })
        return doc

    def _enrich_metadata(
            self,
            documents: List[Document],
            source: Optional[str] = None,
            topic: Optional[str] = None,
            date: Optional[str] = None,
            additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        for doc in documents:
            if source:
                doc.metadata["source"] = source

            if topic:
                doc.metadata["topic"] = topic

            if date:
                doc.metadata["date"] = date

            doc.metadata.setdefault("language", self.doc_language)
            doc.metadata.setdefault("chunk_size", self.chunk_size)

            if additional_metadata:
                doc.metadata.update(additional_metadata)

        return documents

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

    def _process_pipeline(
            self,
            documents: List[Document],
            source: Optional[str] = None,
            topic: Optional[str] = None,
            date: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        if self.clean_text:
            for doc in documents:
                doc.page_content = self._clean_document_text(doc.page_content)

        splits = self._split_documents(documents)
        splits = self._enrich_metadata(
            splits,
            source=source,
            topic=topic,
            date=date,
            additional_metadata=metadata
        )

        if self.remove_duplicates:
            splits = self._deduplicate_documents(splits)

        self._log_metadata_sample(splits)
        return splits

    def process_file(
            self,
            file_path: Optional[str] = None,
            topic: Optional[str] = None,
            date: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        path = file_path or self.file_path
        if not path:
            raise ValueError("file_path must be provided either in __init__ or process_file")

        documents = self._load_document(path)
        return self._process_pipeline(documents, source=path, topic=topic, date=date, metadata=metadata)

    def process_text(
            self,
            text: str,
            source: Optional[str] = None,
            topic: Optional[str] = None,
            date: Optional[str] = None,
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

        documents = self._enrich_metadata(
            documents,
            source=source,
            topic=topic,
            date=date,
            additional_metadata=metadata
        )

        if self.remove_duplicates:
            documents = self._deduplicate_documents(documents)

        self._log_metadata_sample(documents)
        return documents

    def process_documents(
            self,
            documents: List[Document],
            source: Optional[str] = None,
            topic: Optional[str] = None,
            date: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return self._process_pipeline(documents, source=source, topic=topic, date=date, metadata=metadata)