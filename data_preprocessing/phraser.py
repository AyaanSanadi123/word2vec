import os
from utils.streaming import get_stream_counts
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





class StreamingPhraseMiner:
    def __init__(self,thresholds:list = [100,50],delta:int=5):
        self.thresholds = thresholds
        self.delta = delta

    def _score_bigram(self, count_a: int, count_b: int, count_ab: int, total_words: int) -> float:
        
        if count_ab <= self.delta:
            return 0.0

       
        numerator = (count_ab - self.delta) * total_words
        denominator = count_a * count_b
        
       
        if denominator == 0:
            return 0.0
            
        return numerator / denominator 
    
    def fit_transform(self,base_clean_filepath:str,final_filepath: str):
        temp_A = 'temp_A.txt'
        temp_B = 'temp_B.txt'

        # the first step is to read the cleaned source file,
        # run the threshold loop once and save the merged words in file A

        current_input = base_clean_filepath
        current_output = temp_A

        for threshold in self.thresholds:
            print(f"\n--- Running Threshold: {threshold} ---")

            unigram_counts,bigram_counts = get_stream_counts(current_input)
            '''
            at this moment, the execution pauses,
            the get stream function calls the clean file of sentences,
            and line by line begins to build the hash maps, when this process is complete
            it is returned back to this with the values of unigram and bigrams ready 

            '''

            # get total unique words 
            total_words = sum(unigram_counts.values())

            self._process_stream(current_input,current_output,unigram_counts,bigram_counts,total_words,threshold)

            # the file swapping logic 
            current_input = current_output
            if current_output == temp_A:
                current_output = temp_B
            else:
                current_output = temp_A

            # Rename the final successful output file to your desired filename
        if os.path.exists(final_filepath):
            os.remove(final_filepath)
        os.rename(current_input, final_filepath)
        
        # Destroy leftover temp files to reclaim SSD space
        if os.path.exists(temp_A): os.remove(temp_A)
        if os.path.exists(temp_B): os.remove(temp_B)
        
        print(f"\n✅ Streaming Phrase Mining Complete! Output saved to: {final_filepath}")


    def _process_stream(self,input_file:str,output_file:str,unigram_counts:dict,bigram_counts:dict,total_words:int,threshold:int):
        # this takes sentences line by line and runs the scoring function 
        # if the score is greater than the threshold, we merge them
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            # everytime we open a file we run this  standard cleaning process 
            for line in infile:
                sentence = line.strip().split()
                if not sentence:
                    continue

                new_sentence = []
                i = 0
                sent_len = len(sentence)
                while i < sent_len:
                    if i == sent_len - 1:
                        new_sentence.append(sentence[i])
                        break
                    
                    w1 = sentence[i]
                    w2 = sentence[i+1]
                    
                    # Fetch counts with a default of 0 to prevent KeyError crashes
                    count_a = unigram_counts.get(w1, 0)
                    count_b = unigram_counts.get(w2, 0)
                    count_ab = bigram_counts.get((w1, w2), 0)

                    # Score the bigram
                    score = self._score_bigram(count_a, count_b, count_ab, total_words)

                    if score >= threshold:
                        merged_word = f"{w1}_{w2}"
                        new_sentence.append(merged_word)
                        i += 2
                    else:
                        new_sentence.append(w1)
                        i += 1

                outfile.write(" ".join(new_sentence) + "\n")

            
