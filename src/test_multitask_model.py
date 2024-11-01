from multitask_model import MultiTaskSentenceTransformer

# Initialize the model with 4 classes for classification and 3 for sentiment
model = MultiTaskSentenceTransformer(num_classes=4, num_sentiment_classes=3)

# Sample 
test_sentences = [
    "The Flower is amazing and I love it!",
    "The weather is sunny and bright.",
    "I am not satisfied with the service."
]

# Test Task A
print("Testing Sentence Classification Task")
classification_output = model(test_sentences, task="classification")
print("Sentence Classification Output (probabilities):")
print(classification_output)

# Test Task B, Sentiment Analysis
print("\nTesting Sentiment Analysis Task")
sentiment_output = model(test_sentences, task="sentiment")
print("Sentiment Analysis Output (probabilities):")
print(sentiment_output)
