# Python-to-C Integration

> *Building a zero-copy bridge between Python and the C training engine.*

Although the training algorithm is implemented in C for maximum performance, the rest of the project—including preprocessing, vocabulary construction, and evaluation—is written in Python.

To allow these two languages to work together efficiently, the project introduces a lightweight wrapper built using Python's `ctypes` library.

Rather than copying data between Python and C, the wrapper exposes the compiled training engine directly to Python while allowing both languages to operate on the same memory.

---

# Architecture Overview

```
Python

↓

NumPy Arrays

↓

ctypes Wrapper

↓

Shared Library

↓

C Training Engine
```

The wrapper has four primary responsibilities:

- Load the compiled shared library.
- Validate data types and memory layout.
- Define the C function signature.
- Expose a clean Python interface.

---

# Loading the Shared Library

The first step is loading the compiled C library.

```python
c_engine = ctypes.CDLL(lib_path)
```

Once loaded, every exported C function becomes callable directly from Python.

Conceptually,

```
maths_engine.dll

↓

ctypes

↓

Python Function
```

This allows the project to retain Python's ease of use while executing computationally intensive code at native speed. :contentReference[oaicite:2]{index=2}

---

# Memory Validation

C functions operate on raw memory addresses.

Passing fragmented or incorrectly typed arrays can lead to undefined behavior, including segmentation faults.

To prevent this, the wrapper enforces strict requirements using `numpy.ctypeslib.ndpointer`.

```python
np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=2,
    flags="C_CONTIGUOUS"
)
```

Each argument is validated before the C function is called.

The wrapper verifies:

- Data type (`int32`, `float32`)
- Number of dimensions
- Contiguous memory layout

This guarantees that the C engine always receives correctly formatted input. :contentReference[oaicite:3]{index=3}

---

# Function Signature Mapping

Unlike Python, C has no runtime type checking.

Without explicit declarations, `ctypes` assumes default argument and return types, which can easily result in corrupted memory or incorrect results.

The wrapper therefore mirrors the C function signature using

```python
.argtypes
```

and

```python
.restype
```

```
Python

↓

ctypes

↓

train_epoch(...)

↓

C
```

Every parameter—including corpus pointers, embedding matrices, vocabulary metadata, and learning-rate information—is mapped explicitly to its corresponding C type.

The return value is declared as a 64-bit integer to safely handle extremely large training counters without overflow. :contentReference[oaicite:4]{index=4}

---

# A High-Level Python API

Although the underlying C function accepts many low-level parameters, users never interact with it directly.

Instead, the wrapper exposes a simple helper function:

```python
run_c_epoch(...)
```

This function automatically

- determines array lengths,
- forwards all pointers to the C engine,
- executes one complete training epoch,
- returns the updated global progress counter.

From the rest of the project, training appears as a normal Python function call, even though all numerical computation is executed inside compiled C code. :contentReference[oaicite:5]{index=5}

---

# Why This Design?

Separating the wrapper from the training engine provides several advantages.

| Component | Responsibility |
|-----------|----------------|
| Python | Data preparation, orchestration, evaluation |
| NumPy | Efficient contiguous memory allocation |
| `ctypes` | Language interoperability |
| C Engine | High-performance numerical computation |

This separation keeps the training engine focused exclusively on computation while allowing the remainder of the project to benefit from Python's flexibility and extensive ecosystem.

---

# Key Takeaways

The C wrapper acts as the bridge between Python and the native training engine.

It enables:

- direct loading of the compiled library,
- zero-copy sharing of embedding matrices,
- strict validation of memory layout,
- explicit type-safe function signatures,
- and a clean Python API that hides the complexity of the underlying C implementation.

By combining Python's usability with C's performance, the wrapper allows the project to train large-scale Word2Vec models efficiently without sacrificing developer productivity.