# Word2Vec from Scratch
### Skip-Gram with Negative Sampling (SGNS)

> A complete end-to-end implementation of the landmark **2013 Word2Vec** paper,
> **"Distributed Representations of Words and Phrases and their Compositionality"**
> by **Tomas Mikolov et al.**

Built completely from scratch using a **high-performance C training engine** and a **lightweight Python inference API** that emulates the core functionality of `gensim`.

---

## Features

- ⚡ High-performance training engine written in C and Python
- 🧠 Skip-Gram with Negative Sampling (SGNS)
- 📚 Streaming dataset preprocessing pipeline
- 🎯 Vocabulary construction & frequency-based subsampling
- 🔍 Fast cosine similarity search
- ➕ Word analogies via vector arithmetic
- 🎲 Odd-one-out detection
- 🐍 Lightweight Python inference library (NumPy only)

---

# Background

The original Word2Vec paper transformed Natural Language Processing by replacing sparse one-hot word representations with dense vector embeddings learned directly from massive text corpora.

Instead of representing words independently,

```
king
queen
apple
```

Word2Vec learns a continuous embedding space where semantic relationships emerge naturally.

For example,

```
king ───── queen

 man  ───── woman
```

Words with similar meanings occupy nearby regions in this learned vector space.

---

# Project Architecture

```
                 Raw Wikipedia Corpus
                         │
                         ▼
             Data Preprocessing Pipeline
                         │
         ┌───────────────┼────────────────┐
         │               │                │
 Vocabulary        Phrase Detection   Subsampling
         │
         ▼
 Skip-Gram Training Pair Generation
         │
         ▼
   Skip-Gram + Negative Sampling
         │
         ▼
 Learned Embedding Matrix (.npy)
         │
         ▼
    Python Inference Engine
```

---

# Repository Structure

```text
word2vec/

├── data/
│   ├── Raw datasets
│   └── Preprocessing utilities
│
├── data_loader/
│
├── data_preprocessing/
│
├── embedding_layer/
│
├── training_engine/
│   ├── Core SGNS implementation
│   ├── Gradient updates
│   ├── Negative sampling
│   └── Memory management
│
├── utils/
│
├── interface/
│   ├── custom_word2vec.py
│   ├── playground.py
│   └── models/        (Git ignored)
│
├── docs/
│
└── README.md
```

---

# Training Pipeline

The complete pipeline consists of:

### 1. Data Cleaning

- Tokenization
- Lowercasing
- Noise removal

↓

### 2. Phrase Detection

Learns phrases such as

```
new york
machine learning
artificial intelligence
```

↓

### 3. Vocabulary Construction

- Word counts
- Frequency statistics

↓

### 4. Subsampling

Removes extremely frequent words such as

```
the
of
and
is
```

to improve training quality and speed.

↓

### 5. Skip-Gram Dataset Generation

Creates

```
Target → Context
```

training examples using a dynamic context window.

↓

### 6. SGNS Training

Optimizes the embedding matrices using

- Negative Sampling
- Stochastic Gradient Descent
- Sigmoid activation
- Backpropagation

---

# Inference Engine

Once training completes, the C engine exports

- embedding matrix
- vocabulary
- lookup dictionaries

The Python interface loads these files directly and provides an API similar to `gensim`.

Supported operations include

- Word lookup
- Cosine similarity
- Most similar words
- Vector arithmetic
- Odd-one-out detection

---

# Quick Start

## Clone

```bash
git clone https://github.com/your-username/word2vec-from-scratch.git

cd word2vec-from-scratch/interface
```

---

## Install Dependencies

```bash
pip install numpy
```

---

## Download Model Files

The trained embedding matrix is approximately **900 MB**, so it is hosted separately from GitHub.

Download

- `best_target_matrix.npy`
- `word_to_id.json`
- `id_to_word.json`

from:

**👉 Google Drive Link**

Place them inside

```
interface/models/
```

---

## Run

```bash
python playground.py
```

---

# Python API

```python
from custom_word2vec import CustomWord2Vec

model = CustomWord2Vec()

# Retrieve an embedding
vector = model["anarchism"]

# Cosine similarity
print(model.similarity("anarchism", "state"))

# Analogies
print(
    model.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=3
    )
)

# Odd-one-out
print(
    model.doesnt_match(
        ["breakfast", "cereal", "dinner", "car"]
    )
)
```

---

# Example

```
>>> model.similarity("dog", "cat")

0.83

>>> model.most_similar(
        positive=["king","woman"],
        negative=["man"]
    )

queen
princess
monarch
```

---

# Technologies Used

| Component | Technology |
|-----------|------------|
| Training Engine | C |
| Inference API | Python |
| Numerical Operations | NumPy |
| Dataset | Wikipedia (text8 / enwik9) |

---

# Future Improvements

- [ ] Multi-threaded training
- [ ] SIMD optimizations
---

# References

Tomas Mikolov et al.

**Distributed Representations of Words and Phrases and their Compositionality**

NeurIPS 2013

https://arxiv.org/abs/1310.4546
