# Phase 1 — High-Throughput Streaming Data Pipeline

> *Building a scalable preprocessing pipeline capable of handling billion-token corpora without exhausting system memory.*

---

# Overview

Training modern embedding models requires processing extremely large text corpora.

For this project, the primary dataset is **enwik9**, a compressed Wikipedia dump containing over **1 GB of raw text** and more than **150 million tokens** after preprocessing.

Attempting to load a dataset of this size entirely into memory quickly becomes impractical on consumer hardware.

Instead of treating RAM as permanent storage, this project adopts a **streaming architecture**, where text flows through the pipeline one chunk at a time.

```
                Raw Dataset
                     │
                     ▼
              Read One Line
                     │
                     ▼
             Clean & Tokenize
                     │
                     ▼
            Detect Phrases
                     │
                     ▼
          Write Back To Disk
                     │
                     ▼
             Next Chunk...
```

This approach keeps memory usage nearly constant regardless of corpus size.

---

# Objectives

The streaming pipeline performs four primary tasks.

| Component | Responsibility |
|-----------|----------------|
| Tokenizer | Standardize raw Wikipedia text |
| Cleaner | Remove unwanted formatting and punctuation |
| Phrase Miner | Detect statistically significant bigrams |
| Vocabulary Builder | Produce a cleaned corpus for later stages |

---

# Why Streaming?

Earlier versions of the project stored the entire corpus inside Python objects.

```
Raw Text

↓

List

↓

List

↓

List

↓

RAM
```

This approach worked well for small datasets containing only a few million words.

However, scaling to Wikipedia-sized corpora revealed several bottlenecks.

| Problem | Consequence |
|----------|-------------|
| Entire corpus stored in memory | Extremely high RAM usage |
| Millions of unique bigrams | Dictionary explosion |
| Large temporary structures | Frequent MemoryError exceptions |

The solution was to redesign the pipeline around sequential disk streaming.

Instead of

```
Dataset

↓

RAM

↓

Processing
```

the architecture became

```
Dataset

↓

RAM (Temporary Workspace)

↓

Output File
```

Memory is now treated as a temporary processing buffer rather than long-term storage.

---

# Pipeline Architecture

```
Raw Wikipedia Dump
        │
        ▼
Tokenizer
        │
        ▼
Cleaner
        │
        ▼
Sentence Extraction
        │
        ▼
Phrase Mining
        │
        ▼
Vocabulary Construction
        │
        ▼
Training Corpus
```

Every component performs exactly one transformation before passing the data to the next stage.

---

# Tokenization

The tokenizer is responsible for converting noisy Wikipedia markup into a consistent sequence of tokens.

Its responsibilities include

- Lowercasing
- Removing HTML artifacts
- Removing punctuation
- Normalizing whitespace
- Producing consistent word boundaries

The output is a standardized corpus suitable for statistical processing.

---

# Phrase Mining

Rather than treating every token independently, Word2Vec improves embedding quality by identifying common multi-word expressions.

Examples include

```
new york

machine learning

artificial intelligence

san francisco
```

These phrases are merged into single vocabulary entries.

```
new york

↓

new_york
```

This allows the model to learn semantic representations for complete concepts instead of individual words.

Phrase detection is performed over multiple passes using Mikolov's statistical scoring function.

---

# Engineering Challenge — Memory Explosion

The largest engineering challenge during preprocessing was phrase mining.

A corpus containing more than **150 million words** can generate **tens of millions of unique bigrams**.

```
(word1, word2)

↓

Dictionary

↓

50+ million entries
```

A naïve implementation rapidly consumes all available memory.

---

# Solution — Periodic Dictionary Cleanup

Instead of allowing the dictionary to grow indefinitely, the pipeline periodically performs a cleanup pass.

Whenever the bigram table reaches a predefined capacity:

```
Bigram Count

↓

15 Million Entries

↓

Cleanup

↓

Remove Frequency = 1

↓

Continue Streaming
```

Most one-off bigrams represent statistical noise and are extremely unlikely to survive phrase scoring.

Removing them early dramatically reduces memory usage while preserving meaningful phrase candidates.

---

# Engineering Challenge — Sentence Boundaries

Streaming introduces another subtle problem.

```
One Line

≠

One Sentence
```

Wikipedia lines frequently contain multiple complete sentences.

Even more challenging, a sentence may begin on one line and finish on the next.

For example,

```
The dog died. So
the owner got a new puppy.
```

Ideally, the pipeline would maintain a rolling buffer capable of reconstructing sentences spanning multiple lines.

For simplicity, the current implementation intentionally ignores this edge case.

Although this occasionally breaks semantic continuity between adjacent lines, no words are lost, and the impact on phrase quality is minimal.

Support for buffered sentence reconstruction is planned for a future release.

---

# Multi-Pass Phrase Detection

Some phrases require multiple iterations before they can emerge.

Consider

```
new york times
```

Pass One

```
new york

↓

new_york
```

Pass Two

```
new_york times

↓

new_york_times
```

Because phrase frequencies decrease as phrases become longer, later passes operate with progressively lower scoring thresholds.

Example:

```
[100, 50, 25]
```

This allows increasingly specific phrases to be discovered.

---

# Design Decisions

| Decision | Benefit |
|-----------|---------|
| Streaming I/O | Constant memory usage |
| Temporary files | Avoids loading corpus into RAM |
| Multi-pass phrase mining | Detects hierarchical phrases |
| Periodic cleanup | Prevents dictionary explosion |
| Sequential processing | Scales to very large corpora |

---

# Key Takeaways

The streaming pipeline transforms raw Wikipedia text into a clean, phrase-aware training corpus while maintaining nearly constant memory usage.

By replacing in-memory processing with sequential streaming, the preprocessing stage scales from small experimental datasets to corpora containing hundreds of millions of tokens without exhausting system resources.