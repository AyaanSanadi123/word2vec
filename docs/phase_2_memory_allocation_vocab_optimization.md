* **Memory Allocation & Vocabulary Optimization**
* 1. Overview and Objectives 
Before the training can begin, we need to ignore statistical noise, via subsampling,
building the probability tables for negative sampling and allocating float32 weight to the matrices 

* 2. Core Architectural Components
* ** Vocabulary Manager(vocabulary.py) **
Here we pull text line-by-line from the phrased.txt file and begin, to work on 3 tasks
. get total count of all the words using the Counter()
. prune words that are below mincount -> if any words appear less than 5 times in a corpus of 100 million words, they simply lack the vaolume to build any meaningful semantic relationships so, we dont include them 

. we build word_to_id and id_to_word, its a python dict that is gradually increatmented 

* **Frequent Word Subsampling**: 
Implements Mikolov’s subsampling formula. Extremely frequent words (like "the", "a", "is") provide very little semantic value. The manager calculates a discard_prob for every word in the vocabulary. During training, frequent words are probabilistically dropped from the context window, artificially increasing the effective context size and drastically speeding up training.

this is stored in a numpu array of datatype float32, 
this makes it very easy for our c-engine to use, as a numpy array is a contigeous allocation of memory, and its point can be passed to the c engine to work on....


