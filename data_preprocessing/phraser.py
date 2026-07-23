
from collections import defaultdict

class PhraseMiner:
    def __init__(self,thresholds: list = [100,50],delta : int = 5):
        self.thresholds = thresholds
        self.delta = delta 

    def _get_counts(self,corpus : list):
        # this is where we build two hash-maps that help us in the count functions
        unigram_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        total_words = 0
        
        for sentence in corpus:
            total_words += len(sentence)
            for i in range(len(sentence)-1):
                w1 = sentence[i]
                w2 = sentence[i+1]
                unigram_counts[w1] += 1
                bigram_counts[(w1,w2)] += 1
            if len(sentence) > 0:
                unigram_counts[sentence[-1]] += 1


        return unigram_counts, bigram_counts,total_words  

    def _score_bigram(self, count_a: int, count_b: int, count_ab: int, total_words: int) -> float:
        
        if count_ab <= self.delta:
            return 0.0

       
        numerator = (count_ab - self.delta) * total_words
        denominator = count_a * count_b
        
       
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def fit_transform(self, corpus: list) -> list:
        current_corpus = corpus 
        
        for threshold in self.thresholds:
            # Get counts across the whole corpus
            unigram_counts, bigram_counts, total_words = self._get_counts(current_corpus)
            new_corpus = [] 

            # Process each sentence independently
            for sentence in current_corpus:
                new_sentence = []
                i = 0 
                sent_len = len(sentence)

                while i < sent_len:
                    if i == sent_len - 1:
                        new_sentence.append(sentence[i]) 
                        break
                    
                    w1 = sentence[i]
                    w2 = sentence[i+1]
                    count_a = unigram_counts[w1]
                    count_b = unigram_counts[w2]
                    count_ab = bigram_counts[(w1, w2)]

                    score = self._score_bigram(count_a, count_b, count_ab, total_words)

                    if score >= threshold:
                        merged_word = f"{w1}_{w2}"
                        new_sentence.append(merged_word)
                        i += 2
                    else:
                        new_sentence.append(w1)
                        i += 1
                        
                new_corpus.append(new_sentence)
                
            current_corpus = new_corpus
            
        return current_corpus


