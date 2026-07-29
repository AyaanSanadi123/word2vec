# Matrix Initialization

> *Preparing the embedding matrices for efficient training.*

Before the C training engine can begin optimizing word embeddings, the model must initialize two learnable parameter matrices:

- **Target Matrix (`W_target`)**
- **Context Matrix (`W_context`)**

These matrices form the core of the Skip-Gram with Negative Sampling (SGNS) architecture. Every gradient update performed during training modifies one or both of these matrices.

---

# Why Two Embedding Matrices?

Unlike many neural networks that maintain a single representation for each word, Word2Vec learns **two independent embeddings**.

Every word in the vocabulary has:

- a **target embedding**, used when the word is the center of the context window.
- a **context embedding**, used when the word appears as a neighboring word.

```
                 "king"

        ┌──────────────────┐
        │ Target Embedding │
        └──────────────────┘

                 +

        ┌──────────────────┐
        │ Context Embedding│
        └──────────────────┘
```

During training, these matrices gradually learn complementary representations of the corpus.

---

# Breaking Symmetry

A neural network cannot begin learning if every parameter starts with the same value.

Suppose every weight were initialized to zero.

```
0 0 0 0 0 ...

↓

same forward pass

↓

same gradients

↓

same updates
```

Every dimension would evolve identically, preventing the model from learning distinct semantic features.

This problem is known as **symmetry breaking**.

To avoid it, the target matrix is initialized with small uniformly distributed random values.

```
-0.0031
 0.0017
-0.0022
 0.0048
...
```

Random initialization ensures that each embedding dimension begins with a unique value, allowing the optimization process to discover meaningful directions in the vector space. :contentReference[oaicite:0]{index=0}

---

# Memory Layout

Although the embedding matrices are conceptually two-dimensional,

```
Vocabulary Size × Embedding Dimension
```

they are stored internally as **contiguous NumPy arrays**.

For example,

```
Vocabulary = 100,000

Embedding Size = 300
```

The logical matrix

```
100000 × 300
```

is allocated as one continuous block of memory containing

```
30,000,000 float32 values
```

instead of a Python list containing thousands of nested objects.

---

# Why Contiguous Memory?

Python lists store references to objects rather than the objects themselves.

```
List

↓

Pointer

↓

Pointer

↓

Pointer
```

This layout introduces pointer indirection and scatters data throughout memory.

NumPy instead allocates one continuous memory block.

```
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
```

This provides several advantages:

- Better CPU cache locality
- Lower pointer overhead
- Faster sequential access
- Direct compatibility with C

Most importantly, contiguous memory allows Python and C to operate on the **same physical memory** without copying data. :contentReference[oaicite:1]{index=1}

---

# Flattening the Matrix

Although the embeddings behave like a two-dimensional matrix,

```
Vocabulary

↓

Word

↓

300-dimensional vector
```

the C engine accesses them as a one-dimensional array.

```
[target0][target1][target2]...

↓

float *
```

To retrieve the embedding for a particular word, the engine computes

```c
word_id * embedding_dimension
```

For example,

```
Embedding Size = 300

Word ID = 42

Starting Index

42 × 300 = 12600
```

The embedding vector therefore occupies

```
12600

↓

12899
```

within the flattened array.

This approach avoids the complexity of nested arrays while enabling extremely efficient pointer arithmetic.

---

# Zero-Copy Memory Sharing

One of the major advantages of NumPy is that its arrays expose their raw memory addresses.

Instead of copying the matrices into C,

Python simply passes the memory pointer.

```
NumPy Array

↓

Raw Memory Pointer

↓

C Engine

↓

Direct Read / Write
```

Because both languages reference the same memory block, every update performed inside the C engine immediately modifies the original NumPy array.

No serialization.

No copying.

No conversion.

This zero-copy design eliminates unnecessary memory overhead and allows the training engine to operate at native C performance.

---

# Key Takeaways

The matrix initialization stage prepares the model for efficient optimization by

- allocating two learnable embedding matrices,
- breaking symmetry through random initialization,
- storing embeddings in contiguous memory,
- flattening matrices into cache-friendly arrays,
- and enabling zero-copy memory sharing between Python and C.

These design decisions significantly simplify the C implementation while maximizing training performance.