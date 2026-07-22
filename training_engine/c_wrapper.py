import ctypes
import numpy as np
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maths_engine.so')
c_engine = ctypes.CDLL(lib_path)

int_array_type = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
float_array_type = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')

# 3. Map the argument types for the C function
c_engine.train_epoch.argtypes = [
    int_array_type,     # int* corpus
    ctypes.c_int,       # int corpus_len
    float_array_type,   # float* target_matrix
    float_array_type,   # float* context_matrix
    ctypes.c_int,       # int vocab_size
    ctypes.c_int,       # int embed_size
    int_array_type,     # int* unigram_table
    ctypes.c_int,       # int unigram_size
    ctypes.c_int,       # int window_size
    ctypes.c_int,       # int num_negatives
    
    # --- NEW TYPES ---
    ctypes.c_float,     # float initial_lr
    ctypes.c_int,       # int total_expected_pairs
    ctypes.c_int,       # int global_pairs_processed
    
    float_array_type    # float* discard_probs
]

c_engine.train_epoch.restype = ctypes.c_int

# 4. The Python Interface Function
def run_c_epoch(corpus, target_matrix, context_matrix, vocab_size, embed_size, 
                unigram_table, window_size, num_negatives, 
                initial_lr, total_expected_pairs, global_pairs_processed, 
                discard_probs):
    
    corpus_len = len(corpus)
    unigram_size = len(unigram_table)
    
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