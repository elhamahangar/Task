import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import logging

#  reduce warnings
logging.basicConfig(level=logging.ERROR)

#I will use the DistilBERT model as a base due to its efficiency.

class EnhancedSentenceTransformer(tf.keras.Model):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=256, dropout_rate=0.3):
        super(EnhancedSentenceTransformer, self).__init__()
        
        # Load pre-trained transformer model and tokenizer
        self.transformer = TFAutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Additional dense layers to process embeddings
        self.dense1 = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embedding_dim, activation="relu")
        
        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Final projection layer for output embedding dimension
        self.output_layer = tf.keras.layers.Dense(embedding_dim)

    def call(self, sentences):
        # Tokenize input sentences
        inputs = self.tokenizer(sentences, return_tensors="tf", padding=True, truncation=True)
        
        # Pass through transformer model
        outputs = self.transformer(inputs).last_hidden_state
        
        # Mean pooling
        embeddings = tf.reduce_mean(outputs, axis=1)
        
        # Process embeddings through additional dense layers and dropout
        x = self.dense1(embeddings)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.output_layer(x)  # Final embedding layer
        
        return x
