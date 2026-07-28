* ** Symmetry Breaking & Weight Scaling **
Before the C-engine can start updating vectors, we must initialize our two core weight matrices: the Target Matrix and the Context Matrix.

Every word in our vocabulary needs two representations: one when it acts as the center (target) word, and one when it acts as a neighbor (context) word.

If we initialized all weights to 0.0f, every neuron in the model would calculate the exact same gradients. The network would suffer from symmetry failure and fail to learn.

Therefore, we initialize the target matrix with small, uniformly distributed random values.

* ** The Implementation: Contiguous Memory & Python-to-C Handoff:**

In Python, storing matrices as lists of lists creates massive pointer overhead and scatters data across random memory addresses. To make this compatible with our C-engine, we must flatten everything into contiguous blocks of memory.
NumPy Flat Arrays: We initialize the matrices as 1D NumPy arrays of datatype float32. If your vocabulary size is $V$ and your embedding dimension is $D$ (e.g., $300$), the total array size is $V \times D$.

Zero-Copy Memory Sharing: Because NumPy arrays allocate data contiguously in RAM, we can extract their raw memory pointers using ctypes. When we pass target_matrix.ctypes.data into our compiled DLL, Python and C share the exact same physical memory block. C can directly read and write weights via pointer arithmetic (target_matrix[word_id * embed_size]) with zero data copying or performance overhead.

In the c engine, the way we update the vectors, 
because the 2D matrix is flattened into a 1D array,
we get the starting pointer of the vector = word_id * embed_size, and then we, update the slots for bw that and embed_size, so the net length of the vector is embed size.
this allows the C engine to work at a super fast peace, avoiding the complex 2D arrays that are much harder to work with. 