# phase-1 : High-Throughput Streaming Data Pipeline & Phraser

## 1. Overview and Objectives 
when training large embeding models such as this, loading large datasets such as `enwik9` (1GB+ raw text), into the RAM is not feasible and will lead the system to crash 

**Phase 1** we try and solve this problem by implementing a memory efficient, disk streaming architecture that processes text sequentially in chunks, 
this standardizes the corpus and extracts multi-word expressions (phrases)
and also builds a pruned vocabulary
---
## 2. Core Architectural Components

### A. Text Tokenizer & Cleaner (`tokenizer.py`)
* **Standardization:** Strips out unwanted HTML tags, punctuation and special artifacts common in raw Wikipedia dumps.
* **Lowercasing & Normalization:** Converts the text to lowercase and tokenizes sentences uniformly to ensure consistent vocabulary mapping.

### B. Streaming Phrase Miner (`phraser.py`)
instead of relying only on unigrams (that represent one word), we aim to identify frequenty occuring words and merge them into a new word (bigram)
* **Multi-Pass Thresholding:** Implements Mikolov’s statistical scoring formula to calculate bigram association scores based on unigram and bigram counts.
* **File-Swapping I/O Pipeline:** We implemented temporary intermediate files (`temp_A.txt`, `temp_B.txt`) to stream data across multiple passes without loading the corpus into RAM.

---
## 3. Engineering Challenges & Solutions

### The Bigram Memory Bottleneck (`MemoryError`)
* **The Problem:** Running phrase mining on 100+ million words generates tens of millions of unique, one-off bigram combinations. Storing all unique tuples `(w1, w2)` in a standard Python dictionary causes RAM consumption to balloon past 10GB, resulting in a `MemoryError`.
* **The Solution:** Implemented a **Periodic Memory Manager** inside `get_stream_counts()` with a strict capacity threshold (capped safely at 15 million entries). Everytime the bigram reaches this limit, the stream evaluates dictionary size and aggressively purges 1-count bigrams (statistical noise that will never meet the phrase extraction threshold), keeping RAM usage completely flat and stable.

---
## 4. Pipeline Execution Flow
1. **Raw Stream Input:** Reads raw text chunk-by-chunk.
2. **Pass 1 (Cleaning):** Outputs a standardized clean corpus (`_clean.txt`).
3. **Pass 2 & 3 (Mining):** Applies multi-pass thresholding to detect multi-word collocations, outputting the final phased corpus (`_phrased.txt`).
4. **Teardown & Cleanup:** Automatically purges temporary scratch files to leave the workspace pristine.

---

## 5. Engineering Journal: Scaling from Local to Big Data
this is just an informal section where i want to talk about the challanges we faced and how we overcame them

alright so in the previous versions of this pipeline, we treated the corpus as a class object 
the entire text was converted into a list of lists and stored in the ram

this worked fine for smaller datasets, where the word count was a few million words 
there were about 11-12 thousand unique words 
plus the word_to_id dictinoary and the unigram and bigram counts 

the larger datasets have about 150 milllion + words and about 2-3 million unique words 
this itself would take significant amount of RAM 
plus the unigram of this would take a little more RAM but the real bottle neck comes while trying to store the bigram 
150 million words would create around 50 million bigrams 
this would eat up ram and lead to out-of-memory errors 


thus we need to use big data techniques to stream the data in bits and pieces.
also another change in the approach will be to stop treating the ram as a storage place and begin to treat it as a place where data is processed and discarded right away

 data -> ram (cleaning) -> new_file

* how to stream data line by line 
instead of using in-built python functions such as read(),readlines()
we use a simple for loop to get line, clean them break them into words and stream them back into a new file 

now, because we are streaming data line by line, we run into another problem 
* **the line to sentence issue**
now when we use to just dump the data into the RAM, our sentence_pattern regex use 
to find all the sentences identifying things like '.?!'
Now that we are streaming data line by line we run into a problem

one line != one sentence 
so how can we process one sentence(the distinction of a sentence is required for semantic meaning,
if we loose track of what a sentence is, its just becomes a large collection of words and the model behaves weird)
from streaming the data line by line??


how the dataset is structured and what are our options ->
the dataset we are going to use is enwik9,
each line of this dataset contains about 5-10 complete sentences 
thus when we run sentence_pattern on a line, we get those many sentences 

* the edge case,
now assume a sentence does not end in the line
we hit a \n and the remaining sentence finishes in the next line 

ideally to handle this case, we need to build a custom buffer, that stores the data of the two lines
runs the sentence_pattern on them both, glues the sentence and returns the value 

but for the simplicity of this version, we are going to just simply ignore the edge cases,
this feature will be added in future versions 
the goal of this version is to get the basic data streaming fundamentals 

so lets try and understand what type of impact it has on the data 
lets say we have something like
"The dog died. So\nthe owner got a new puppy!"
now the line "The dog died. So" will first get streamed 
the sentence_pattern will return two lists and the clean_text will convert it to all small 
[['the', 'dog', 'died'], ['so']]

the next line will be 
[['the', 'owner', 'got', 'a', 'new', 'puppy']]
and finally when we stream it into our new file we get 

the dog died
so
the owner got a new puppy

thus no word is deleted, but we do loose the semantic relation between so-the
this effects our bigram formation as "so" and "the" never really get to be a pair, But most such words never actually do become valid pairs so, the net effect of this edge case is minimal 

* **BiGrams and their challanges**
the bigram dilemma 
now as we stated before, the bigram count can reach 50 million if left unchecked, 
this can cause the RAM to run out of memory 

now the way bigrams are merged is by using a formula 
a scoring function, at the end of the unigram and bigram buildup
we run this formula score(a,b) = (count(a,b) - delta) * total_words/count(a) * count(b)

the count(a,b) is stored in the bigram and the count(a),count(b) in the unigram 

if this score value is above a certain thershold, we merge the words into a new word,
eventually the bigram hash map is discarded and only the new unigram hasmap exists with the count of the new formed words 

now again, the ideal case for this will be, we get the bigram and unigram loaded into the ram
so we know the count of every count(a,b) possible.
but again we are constrained by memory 
so we use a trick, now this data set will roughly produce around 50 million bigrams 
so everytime the bigram reaches 25 million slot capacity (this was later reduced to 15 million because my laptop crashed)
we run a small cleaning job on it, now its pretty evident that most bigrams wont make any sense 
things like 'so-the' will never be words
things like new-york and san-fransico will be,
so the bet is, at this mark of 25 million, we have new-york come atleast 2-4 times,
we clear out all the bigrams the have count = 1, this cleans a massive amount of space and also hopefully is enough 
context length to let words that actually need to be merged appear enough amount of times 

Now, we do this merging phase multiple times, for words such as new_york_times to form, we need atleast two passes
1. new york times -> new_york times
2. new_york times -> new_york_times

because the word 'new' and 'york' appear more frequently in text than new_york_times 
a lower threshold value is required for the second merge ,
thus we just send a list of threshold values such as 
[100,50,25], into our scoring function 


