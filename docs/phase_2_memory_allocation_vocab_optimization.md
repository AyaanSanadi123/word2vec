# Phase 2 — Vocabulary Construction & Memory Optimization

> *Transforming the cleaned corpus into efficient data structures optimized for high-performance Skip-Gram training.*

---

# Overview

Once preprocessing is complete, the pipeline constructs the vocabulary and prepares every data structure required by the C training engine.

This phase performs three major tasks:

- Build the vocabulary
- Compute subsampling probabilities
- Construct the unigram table used for negative sampling

These structures remain fixed throughout training and are shared directly with the C engine.

---

# Pipeline

```
Phrased Corpus

↓

Count Words

↓

Remove Rare Words

↓

Assign Integer IDs

↓

Compute Discard Probabilities

↓

Build Unigram Table

↓

Allocate Embedding Matrices
```

---

# Vocabulary Construction

Every token is counted using Python's `Counter`.

After counting, words occurring fewer than the configured minimum frequency are removed.

```
dog

1245

✓
```

```
xylophonically

2

✗
```

Rare words contribute little statistical information and unnecessarily increase memory consumption.

Removing them produces a smaller, denser vocabulary while improving training efficiency. :contentReference[oaicite:0]{index=0}

---

# Integer Encoding

Each remaining word receives a unique integer identifier.

```
king

↓

1045
```

```
queen

↓

2088
```

These mappings are stored inside

```
word_to_id

id_to_word
```

Integer encoding dramatically reduces memory usage compared to repeatedly storing strings.

---

# Frequent Word Subsampling

Words such as

```
the

of

is

and
```

appear extremely frequently but contribute relatively little semantic information.

Instead of training on every occurrence, Word2Vec probabilistically removes these words according to Mikolov's subsampling formula.

This increases the effective context window while reducing redundant computation.

The resulting discard probabilities are stored inside a contiguous `float32` NumPy array for efficient access by the C engine. :contentReference[oaicite:1]{index=1}

---

# The Unigram Table

Negative sampling requires drawing words according to a non-uniform probability distribution.

Sampling uniformly would overrepresent rare words, while sampling purely by frequency would overwhelmingly select common stop words.

Instead, Word2Vec samples according to

\[
P(w_i)=
\frac{f(w_i)^{0.75}}
{\sum_j f(w_j)^{0.75}}
\]

Rather than evaluating this equation during every SGD update, the probabilities are precomputed once in Python.

A large lookup table containing millions of entries is then constructed.

```
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

3

3

3

7

7

10

...
```

Each slot stores only a word ID.

Sampling therefore becomes

```
Random Integer

↓

Array Lookup

↓

Word ID
```

an **O(1)** operation that eliminates repeated probability calculations during training. :contentReference[oaicite:2]{index=2}

---

# Why Precompute?

The C training engine performs billions of negative-sampling operations.

Even a small amount of extra computation inside the inner training loop would significantly reduce throughput.

Precomputing the unigram table shifts this cost to preprocessing, allowing training to perform nothing more than a random array lookup.

---

# Design Decisions

| Decision | Benefit |
|-----------|---------|
| Minimum frequency pruning | Smaller vocabulary |
| Integer encoding | Faster lookup |
| Subsampling | Better semantic learning |
| Contiguous NumPy arrays | Zero-copy C integration |
| Precomputed unigram table | Constant-time negative sampling |

---

# Key Takeaways

This phase transforms a cleaned text corpus into compact, cache-friendly data structures optimized for high-performance SGNS training.

By performing expensive statistical computations once during preprocessing, the C engine is free to focus exclusively on numerical optimization during training.