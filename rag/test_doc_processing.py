from DocumentPreprocessor import DocumentPreprocessor


preprocessor = DocumentPreprocessor(
    remove_duplicates=True,
    deduplication_method="minhash",
    similarity_threshold=0.95,
    log_metadata_sample=True
)

docs = preprocessor.process_file("ORDER_FLOW_Trading_Setups.pdf")

print(len(docs))
