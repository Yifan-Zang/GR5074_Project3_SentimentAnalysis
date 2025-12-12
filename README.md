# GR5074_Group2_Project3_SentimentAnalysis
**Columbia University · Advanced Machine Learning (GR5074)**

**Team Members:**
- Sofia Giorgianni
- Xingyu Shi
- Yifan Zang

## Project Overview

This project develops a comprehensive sentiment classification pipeline for movie reviews using the Stanford Sentiment Treebank (SST-2) dataset. We implemented and compared multiple approaches ranging from traditional machine learning (Logistic Regression, SVM, Random Forest) to deep neural networks (Feed-Forward, CNN, Bi-LSTM) and state-of-the-art transfer learning (GloVe embeddings, DistilBERT fine-tuning). The project analyzes model performance through statistical validation (McNemar's test), conducts detailed error analysis on misclassified examples, and explores data augmentation and sentiment lexicon integration. 

## Objectives
1. Perform text preprocessing and exploratory data analysis (EDA) on movie review sentiment.
2. Build baseline traditional machine learning models with TF-IDF features.
3. Implement neural network architectures (Feed-Forward, 1D-CNN, Bi-LSTM) for text classification.
4. Apply transfer learning with pre-trained GloVe embeddings and fine-tune DistilBERT.
5. Optimize hyperparameters using Keras Tuner for the GloVe CNN model.
6. Compare all models under consistent evaluation metrics and statistical significance testing.
7. Conduct comprehensive error analysis identifying linguistic artifacts that confuse models.
8. Investigate data augmentation (synonym replacement, back-translation) and VADER lexicon integration.

## Dataset

Source: Stanford Sentiment Treebank (SST-2)
Task: Binary sentiment classification of movie reviews

**Classes:**
- Negative (0)
- Positive (1)

**Split Strategy:** Stratified 80/10/10 split
- Training: 53,879 samples (80%)
- Validation: 6,735 samples (10%)
- Test: 6,735 samples (10%)

**Preprocessing:**
- HTML tag removal, lowercasing, punctuation stripping
- Vocabulary pruning (min frequency threshold = 3)
- TF-IDF vectorization (max features = 5000, bigrams)
- Neural: Tokenization with padding/truncation (max length = 50)

**Class Balance:** Preserved through stratified sampling (~50% positive, ~50% negative)

## Methods

### 1. Traditional Machine Learning (Part 3)

**Features:** TF-IDF (5000 features, unigrams + bigrams)

| Model                     | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression (CV)** | 79.4%   | 79.3%     | 81.5%  | 80.4%    | 87.2%   |
| **Linear SVM**              | 78.8%   | 79.0%     | 80.3%  | 79.6%    | 87.8%   |
| **Random Forest**           | 74.1%   | 73.1%     | 78.9%  | 75.9%    | 80.0%   |
| **Gradient Boosting**       | 70.1%   | 67.5%     | 81.1%  | 73.7%    | 77.6%   |

### 2. Neural Networks (Part 4)

**Preprocessing:** Keras Tokenizer (vocab size = 20,000, max length = 50)

| Model | Architecture | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|-------------|----------|----------|---------|---------------|
| **Feed-Forward (Trainable)** | Embedding(128) → Flatten → Dense(128, 64) | 76.8% | 75.7% | 84.0% | 45s |
| **Feed-Forward (Frozen)** | Embedding(128, frozen) → Flatten → Dense(128, 64) | 52.6% | 57.1% | 55.6% | 38s |
| **1D-CNN** | Embedding(128) → Conv1D(128, k=3) × 2 → GlobalMaxPool | 78.7% | 79.1% | 86.9% | 62s |
| **Bi-LSTM** | Embedding(128) → BiLSTM(64, 32) → Dense(64) | 77.1% | 79.1% | 87.6% | 184s |

### 3. Transfer Learning (Part 5)

#### 3.1 GloVe Embeddings + CNN

**Configuration:**
- Pre-trained GloVe 100-dimensional embeddings
- Frozen embedding layer + 1D-CNN (128 filters, kernel size 5)
- Dropout (0.5) and Dense layers (64 units)

#### 3.2 DistilBERT Fine-tuning

**Configuration:**
- Model: `distilbert-base-uncased` (66M parameters)
- Learning rate: 2e-5 (Adam optimizer)
- Batch size: 16
- Epochs: 3
- Max sequence length: 64

### 4. Hyperparameter Optimization (Part 6)

**Model:** GloVe CNN  
**Method:** Random Search (Keras Tuner, 15 trials)

**Search Space:**
- Filters: [64, 128, 192, 256]
- Kernel size: [3, 5, 7]
- Dropout rate: [0.2, 0.3, 0.4, 0.5, 0.6]
- Dense units: [32, 64, 96, 128]
- Learning rate: [1e-4, 5e-3] (log scale)

**Best Configuration:**
- Filters: 256
- Kernel size: 3
- Dropout: 0.5
- Dense units: 128
- Learning rate: 6.65e-4

**Results:**
- **Accuracy:** 78.5%
- **ROC-AUC:** 87.5%
- **Improvement:** +0.5% over default hyperparameters
- Validates kernel size 3 as optimal for capturing trigrams in sentiment analysis

### 5. Statistical Validation (Part 7)

**McNemar's Test:** DistilBERT vs. Bi-LSTM

| Metric | Value |
|--------|-------|
| Test Statistic | 56.0 |
| **P-value** | **< 0.000001** |
| Accuracy Difference | +9.98% |
| Net Improvement | +96 correct predictions |

**Conclusion:** DistilBERT's superiority over Bi-LSTM is **statistically significant** (p < 0.05), not due to random chance.

### 6. Error Analysis (Part 7)

**DistilBERT Misclassifications:** 109/962 test samples (11.3% error rate)

| Error Type | Count | Percentage |
|------------|-------|------------|
| False Positives | 54 | 49.5% |
| False Negatives | 55 | 50.5% |

### 7. Bonus Implementations (Part 8)

#### 7.1 Data Augmentation

**Techniques:**
- Synonym replacement (WordNet)
- Random insertion/deletion/swap
- Back-translation (English → French → English)

#### 7.2 VADER Sentiment Lexicon

**Integration:** TF-IDF + VADER features (4 dimensions: neg, neu, pos, compound)

## Model Comparison Summary

| Model | Accuracy | ROC-AUC | Category |
|-------|----------|---------|----------|
| **DistilBERT Fine-tuned** | **88.7%** | **96.1%** | Transfer Learning |
| **Bi-LSTM** | 77.1% | 87.6% | Neural Network |
| **Linear SVM** | 78.9% | 87.6% | Traditional ML |
| **Optimized GloVe CNN (Hyperparameter Tuned)** | 78.5% | 87.5% | Hyperparameter Tuned |
| **GloVe CNN (100d, Frozen)** | 78.9% | 87.0% | Transfer Learning |
| **1D-CNN** | 78.7% | 86.9% | Neural Network |
| **Feed-Forward (Trainable)** | 76.8% | 84.0% | Neural Network |
| **Logistic Regression** | 79.3% | 87.1% | Traditional ML |
| **Random Forest** | 72.6% | 80.0% | Traditional ML |
| **Gradient Boosting** | 70.9% | 77.6% | Traditional ML |
| **Feed-Forward (Frozen)** | 52.6% | 55.6% | Neural Network |

## References

- Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. *EMNLP*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS Workshop*.
