
from collections import defaultdict

class PhraseMiner:
    def __init__(self,thresholds: list = [100,50],delta : int = 5):
        self.thresholds = thresholds
        self.delta = delta 

    def _get_counts(self,tokens : list):
        # this is where we build two hash-maps that help us in the count functions
        unigram_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        for i in range(len(tokens)-1):
            w1 = tokens[i]
            w2= tokens[i+1]
            unigram_counts[w1] += 1
            bigram_counts[(w1,w2)] += 1

        if len(tokens) > 0:
            unigram_counts[tokens[-1]] += 1 

        return unigram_counts, bigram_counts  

    def _score_bigram(count_a:int,count_b:int,count_ab:int) -> float:
        pass
        
    def fit_transform(self, tokens: list) -> list:
        """
        Main Engine: 
        Loops through the thresholds. For each threshold, it gets the counts, 
        evaluates every adjacent pair, and builds a new list with merged phrases.
        """
        pass