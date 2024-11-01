from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

# Define the multi-task model
class MultiTaskSentenceTransformer(tf.keras.Model):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=256, dropout_rate=0.3, num_classes=3, num_sentiment_classes=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.transformer = TFAutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dense_shared = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classification_head = tf.keras.layers.Dense(num_classes, activation="softmax")
        self.sentiment_head = tf.keras.layers.Dense(num_sentiment_classes, activation="softmax")

    def call(self, inputs, task="classification"):
        inputs = self.tokenizer(inputs, return_tensors="tf", padding=True, truncation=True, return_attention_mask=True)
        transformer_outputs = self.transformer(inputs).last_hidden_state
        shared_embeddings = tf.reduce_mean(transformer_outputs, axis=1)
        x_shared = self.dense_shared(shared_embeddings)
        x_shared = self.dropout(x_shared)
        if task == "classification":
            return self.classification_head(x_shared)
        elif task == "sentiment":
            return self.sentiment_head(x_shared)

# Define compute_loss function for each task
def compute_loss(model, inputs, labels, task):
    if task == "classification":
        predictions = model(inputs, task="classification")
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss_fn(labels, predictions)
    elif task == "sentiment":
        predictions = model(inputs, task="sentiment")
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss_fn(labels, predictions)

# Define layer-specific learning rates
layer_learning_rates = {
    "transformer_layer_1_to_4": 1e-5,
    "transformer_layer_5_to_8": 2e-5,
    "dense_shared": 2e-5,
    "classification_head": 3e-5,
    "sentiment_head": 3e-5,
}

# Define the custom training step with layer-wise learning rates
@tf.function
def train_step(model, inputs, labels_classification, labels_sentiment, optimizer):
    with tf.GradientTape() as tape:
        classification_loss = compute_loss(model, inputs, labels_classification, task="classification")
        sentiment_loss = compute_loss(model, inputs, labels_sentiment, task="sentiment")
        total_loss = classification_loss + sentiment_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)

    for i, var in enumerate(model.trainable_variables):
        layer_name = var.name.split("/")[0]
        learning_rate = layer_learning_rates.get(layer_name, 1e-5)
        gradients[i] = gradients[i] * learning_rate
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return classification_loss, sentiment_loss, total_loss
