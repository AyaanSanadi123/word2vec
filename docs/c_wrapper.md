**Python-to-C Integration & The C-Wrapper:**

1. Overview and Objectives
Writing high-performance math in C is useless if Python cannot communicate with it seamlessly. Because Python and C manage memory differently, we need a robust bridge layer.

The C-Wrapper (c_wrapper.py) handles loading the compiled shared library (maths_engine.dll), enforcing strict memory alignment rules, mapping data types, and providing a clean Python function interface (run_c_epoch) so the rest of the application can trigger training effortlessly.
2. Core Architectural Components
A. Loading the Shared Library (ctypes.CDLL)
To execute compiled C code from Python, we load the compiled dynamic-link library using Python's built-in ctypes module:

Python
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maths_engine.dll')
c_engine = ctypes.CDLL(lib_path)
This mounts the compiled functions directly into Python's runtime memory, allowing us to call C functions as if they were native Python methods.

B. Strict Memory Contiguity & NumPy Types (ndpointer)
C functions expect raw, continuous memory addresses. If a NumPy array is fragmented or non-contiguous, passing its pointer will cause a segmentation fault or memory corruption. We use NumPy's ctypes integration to enforce strict formatting:

Python
int_array_type = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
float_1d_type = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
float_2d_type = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
dtype Enforcement: Explicitly maps Python arrays to 32-bit integers (int32) or single-precision floats (float32), matching C's exact data types (int and float).

flags='C_CONTIGUOUS': Ensures the array data is laid out sequentially in RAM row-by-row with zero gaps.

3. Function Signature Mapping (argtypes and restype)
By default, ctypes assumes all C functions return a standard 32-bit integer (int) and accepts any arguments without validation. To prevent memory crashes or incorrect math, we explicitly declare the input arguments and return type matching our C function:

Python
c_engine.train_epoch.argtypes = [
    int_array_type,     # int* corpus (1D)
    ctypes.c_int,       # int corpus_len
    float_2d_type,      # float* target_matrix (2D)  
    float_2d_type,      # float* context_matrix (2D)
    ctypes.c_int,       # int vocab_size
    ctypes.c_int,       # int embed_size
    int_array_type,     # int* unigram_table (1D)
    ctypes.c_int,       # int unigram_size
    ctypes.c_int,       # int window_size
    ctypes.c_int,       # int num_negatives
    ctypes.c_float,     # float initial_lr
    ctypes.c_longlong,  # long long total_expected_pairs
    ctypes.c_longlong,  # long long global_pairs_processed
    float_1d_type       # float* discard_probs (1D) 
]

# Crucial: Declare return type as a 64-bit integer to prevent counter overflow
c_engine.train_epoch.restype = ctypes.c_longlong
4. The Python Interface Function (run_c_epoch)
The wrapper wraps the low-level C function call inside a clean, high-level Python utility function. This handles automatic length extraction and returns the updated global pair count after the epoch completes:

Python
def run_c_epoch(corpus, target_matrix, context_matrix, vocab_size, embed_size, 
                unigram_table, window_size, num_negatives, 
                initial_lr, total_expected_pairs, global_pairs_processed, 
                discard_probs):
    
    corpus_len = len(corpus)
    unigram_size = len(unigram_table)
    
    # Trigger the compiled C function directly via ctypes pointer sharing
    final_pair_count = c_engine.train_epoch(
        corpus, corpus_len,
        target_matrix, context_matrix,
        vocab_size, embed_size,
        unigram_table, unigram_size,
        window_size, num_negatives,
        initial_lr, total_expected_pairs, global_pairs_processed, 
        discard_probs
    )

    return final_pair_count