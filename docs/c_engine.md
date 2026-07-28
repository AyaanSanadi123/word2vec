* **phase-3 : The C-Engine & Training Architecture**

* **Overview and Objectives:**
Once Python has preprocessed the text corpus, built the unigram tables, and initialized our memory-aligned weight matrices, execution is handed off to maths_engine.c.

This component serves as the computational core of the model. It handles streaming token traversal, dynamic sliding window generation, negative sampling, and stochastic gradient descent (SGD) updates at maximum CPU speed.

* Overall File Structure and loop hierarchy
To process 100+ million tokens efficiently, the train_epoch function relies on a nested loop structure. Let's break down how the loops are layered from the outside in:
1.The Parallel Arena (#pragma omp parallel): Spawns our worker threads, ensuring each gets its own private workspace.

2.The Corpus Loop (for (int i = 0; i < corpus_len; i++)): Distributed statically across the CPU threads via OpenMP. It steps through the flat integer corpus token by token.

3.The Direction Loop (for (int direction = -1; direction <= 1; direction += 2)): Evaluates both the left and right context of the center word.

4.The Window Step Loop (for (int step = 1; step <= dynamic_window; step++)): Dynamically scales out to the randomized window boundary.

5.The Negative Sampling Loop (for (int n = 0; n <= num_negatives; n++)): Executes $1$ positive sample and $K$ negative samples ($n=0$ is the true context word; $n > 0$ pulls random words from the unigram table).

6.The Embedding Dimension Loop (for (int d = 0; d < embed_size; d++)): Iterates across the vector dimensions (e.g., $300$) to calculate dot products, sigmoids, and vector updates.

* **Function Parameters & State Tracking:**
The train_epoch function accepts the flattened data pointers and tracking variables needed to execute a single pass over the dataset
Tracking Progress and Rate Decay (epoch_pairs_processed)
The Problem: Tracking every single word pair globally across multiple threads introduces heavy lock contention.

The Solution: We maintain a thread-local counter (local_word_count). Every 10,000 words, the thread uses a lightweight atomic update to sync with a shared epoch counter

* **Vector Updates & Private Gradient Buffers :**(local_target_update)
When updating weights during SGD, a center word interacts with multiple context and negative words within a single window step. If we modified the target matrix directly during the negative sampling loop, we would create mathematical conflicts.

The Private Workspace: At the start of each thread, we allocate a private gradient buffer:


float *local_target_update = (float *)malloc(embed_size * sizeof(float));
Accumulation: As we loop through the positive and negative samples, gradients are accumulated safely inside this private buffer:


local_target_update[d] += gradient * u_context[d];
The Commit: Once all negative samples for that word pair are processed, the accumulated gradients are written to the target vector all at once:


float *v_target = &target_matrix[word_id * embed_size];
for (int d = 0; d < embed_size; d++) {
    v_target[d] += local_target_update[d];
}

***Thread-Safe Entropy: The Seed Initialization Line:**
One of the most critical lines in the multi-threaded engine ensures that threads do not repeat identical random sequences across epochs:


unsigned long long local_random = (unsigned long long)omp_get_thread_num() * 25214903917ULL + 11 + (unsigned long long)starting_global_pairs;
omp_get_thread_num(): Guarantees that Thread 0, Thread 1, and Thread 7 start with distinct foundational seeds.

+ (unsigned long long)starting_global_pairs: Solves the "Epoch Groundhog Day" bug. Because starting_global_pairs increases every epoch, the starting seed shifts every time a new epoch begins. This forces the PRNG to generate a fresh, diverse stream of negative samples and dynamic window sizes instead of overfitting to the same random choices across the epochs.

