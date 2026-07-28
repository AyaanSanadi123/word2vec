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

* **The Unigram Table & Negative Sampling:**
For negative sampling, we need to find k negative words, 
and taking random words (each word with equal chance) leads to obscure pairs, that are not relevent 

And taking words just on pure frequency alone, we almost always will train on words like "the","and"... 
thus mikolov provides a formula 
$$P(w_i) = \frac{f(w_i)^{0.75}}{\sum_{j=0}^{n} f(w_j)^{0.75}}$$

Calculating probabilities on the fly for every single word in C would severely bottleneck our training speed. Instead, we pre-compute the probabilities in Python (vocabulary.py) and build a massive, static 1D array (the Unigram Table) to pass into C.
And the way it works is, our unigram table is 10 million slots long, 
if the word "animal" has id 3 and its p(w_i) is say 3%, we just fill 300,000 slots in this array with value 3...
later during training,  we generate a random number in the c engine and extract the word_id 


