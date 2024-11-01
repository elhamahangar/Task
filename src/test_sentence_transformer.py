from sentence_transformer import EnhancedSentenceTransformer

# Initialize the enhanced model
enhanced_sentence_transformer = EnhancedSentenceTransformer()

# Sample sentences
sentences = ["I love machine learning.", "Sentence transformers are powerful.", "Transformers can encode text effectively."]
embeddings = enhanced_sentence_transformer(sentences)
print("Enhanced Embeddings:", embeddings)
