# this class needs to handle 3 things
# 1. vocab mapper 
# 2. sub-sampling filter 
# 3.  unigram distribution noise generator (finding good quality words for negative sampling)


'''
the reason these three have been clubbed together is because,
all of these require count(wi) -> which basically means the count of word 'i'
and this can be calculated just once, and reduce the computational overhead 
'''
import numpy as np
from collections import Counter
import math 
class VocabManager:
    def __init__(self,min_count : int = 5,subsample_t:float = 1e-5):

        # basic allocations
        self.min_count = min_count
        self.t = subsample_t
        self.total_words = 0

        # the vocab mapper 
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = {} # dic to see how many times each word is used in a corpus

        # the sub-sampling filter 
        self.discard_probs = None
        # unigram noise distribution 
        self.unigram_table = None


    def build_vocab(self, filepath: str):
                 
        '''
         our tasks are 
         1. get total count of all the wors 
         2. prune words below min_count 
         3. build word_to_id and id_to_word         
        '''
        raw_counts = Counter()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip().split()
                if not sentence:
                    continue
                raw_counts.update(sentence)


        current_id = 0
        self.total_words = 0


        for word,count in raw_counts.items():
            if count >= self.min_count:

                # us building our own, pruned count 
                self.word_counts[word] = count
                self.total_words += count

                # Build the bidirectional bridge
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word

                
                current_id+=1
        print(f"✅ Vocab built! Unique words retained: {len(self.word_to_id):,}")
        print(f"✅ Total tokens in corpus after pruning: {self.total_words:,}")

    def calculate_subsampling(self):
        self.discard_probs = np.zeros(len(self.word_to_id), dtype=np.float32)

        for word,count in self.word_counts.items():
            word_id = self.word_to_id[word]

            f_w = count/self.total_words

            discard_prob = 1 - math.sqrt(self.t / f_w)

            self.discard_probs[word_id] = max(0.0,discard_prob) # if the f_w is smaller than t, discard prob can be 1 - (>1) and can be negitive, so basically rounded of to 0, saying no chance of dropping this rare word 

    def build_unigram_table(self,table_size:int=10_000_000):
        power = 0.75
        total_adjusted = sum(math.pow(count, power) for count in self.word_counts.values()) #word_counts is {'dog' : 100}, .values is basically just the int value, so math.pow(count,power) 100^0.75 and math.sum adds the entire value of the tokens 

        # redeclaring table 

        temp_table = []

        for word,count in self.word_counts.items():
            word_id = self.word_to_id[word]

            adjusted_fraction = math.pow(count,power)/total_adjusted

            num_slots = int(round(adjusted_fraction * table_size))
            temp_table.extend([word_id] * num_slots)
        self.unigram_table = np.array(temp_table, dtype=np.int32)
        print(f"✅ Unigram table generated with {len(self.unigram_table):,} slots.")