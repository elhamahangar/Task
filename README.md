# Enhanced Sentence Transformer

This project implements an enhanced sentence transformer model using TensorFlow and the Hugging Face `transformers` library. The model leverages a pre-trained DistilBERT transformer to encode input sentences into fixed-length embeddings with additional dense layers for improved processing and flexibility.

## Model Overview

The `EnhancedSentenceTransformer` class:
- Uses **DistilBERT** as the base transformer model for efficiency and effectiveness in sentence representation.
- Includes additional **dense layers** and a **dropout** layer to refine and regularize the generated embeddings.
- Outputs embeddings of a specified dimensionality for downstream NLP tasks.

### Key Features
- **Transformer Backbone**: Uses DistilBERT for fast, high-quality text embeddings.
- **Dense Layers**: Additional dense layers to enhance the representation capabilities.
- **Dropout**: Added for regularization and to prevent overfitting.
- **Mean Pooling**: Applies mean pooling to obtain sentence-level embeddings from token-level outputs.

## Installation

To run this project, ensure have the following Python packages installed:

- TensorFlow
- transformers



# Multi-Task Transformer Model

This project implements a multi-task transformer model in TensorFlow for **sentence classification** and **sentiment analysis**. The project covers multiple tasks and includes Docker support for easy setup.

## Project Structure

- **src/multitask_model.py**: Main model and training implementation.
- **src/test_multitask_model.py**: Testing script for the multi-task model.
- **src/test_sentence_transformer.py**: Testing script for sentence transformer.
- **src/training_considerations.md**
- **src/layerwise_learning.md** 
- **requirements.txt**
- **Dockerfile**: Docker setup for containerization.

- **Python 3.7+**
- **Docker** 

