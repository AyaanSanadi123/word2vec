
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

    def _score_bigram(self, count_a: int, count_b: int, count_ab: int, total_words: int) -> float:
        
        if count_ab <= self.delta:
            return 0.0

       
        numerator = (count_ab - self.delta) * total_words
        denominator = count_a * count_b
        
       
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def fit_transform(self, tokens: list) -> list:
        current_tokens = tokens 
        for threshold in self.thresholds: # this is the main-decay loop

            # get the latest counts of the tokens, this keeps changing after each iteration 
            unigram_counts,bigram_counts = self._get_counts(current_tokens)
            total_words = len(current_tokens)
            new_tokens = [] # list to save new tokens 
            i = 0 # this is a manual pointer, and its used to adjust the w(i+1) case 


            # process the current tokens 
            while i < total_words:
                if i == total_words - 1: # if we are on the last word, we cannot try and access i+1, else the code will break
                    # we just add to the new tokens and break out of the loop 
                    new_tokens.appned(current_tokens[i])
                    break

                
                w1 = current_tokens[i]
                w2 = current_tokens[i+1]

                count_a = unigram_counts[w1]
                count_b = unigram_counts[w2]
                count_ab = bigram_counts[(w1,w2)]

                score = self._score_bigram(count_a,count_b,count_ab,total_words)

                # if the score is more than the threshold
                if score >= threshold:
                    merged_word = f"{w1}_{w2}"
                    new_tokens.append(merged_word)
                    i+=2 # as we merged two words into one, we skip past them

                else:
                    new_tokens.append(w1)
                    i+=1

            current_tokens = new_tokens
        return current_tokens


