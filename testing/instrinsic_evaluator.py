'''
this script has 2 major tasks 
1. check the cosine similarity of the model -> input : dog and get the top 5 outputs 
2. vector analogies -> king-man+women= queen 
'''

import numpy as np 
class Word2VecEvaluator:
    # target matrix -> V(center word matrix),vocab -> vocab manager(contains all the word_to_id and discard probabilities)
    def __init__(self,target_matrix:np.ndarray,vocab):
        self.vocab = vocab 
        # 1. Calculate the length (magnitude) of every vector in the matrix
        # keepdims=True ensures it stays a 2D column so we can divide properly
        vector_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)
        vector_norms[vector_norms == 0] = 1e-10
        # 2. Divide the matrix by the norms. 
        # Now, every vector has a length of exactly 1.0.
        self.normalized_matrix = target_matrix / vector_norms

    def get_vector(self,word:str)->np.ndarray:
        if word not in self.vocab.word_to_id:
            raise ValueError(f"'{word}' is not in the vocabulary.")
        word_id = self.vocab.word_to_id[word]
        return self.normalized_matrix[word_id]
    
    def most_similar(self,word:str,top_n : int = 5) -> list :
        query_vector = self.get_vector(word)
        return self._find_closest(query_vector,ignore_words={word},top_n=top_n)
    def _find_closest(self,query_vector:np.ndarray,ignore_words:set,top_n : int) -> list :

        # get the dot product with our vocabulary 
        similarities = np.dot(self.normalized_matrix,query_vector)
        # this gives u some cosine values of the words that might me closest, to our vector 

        # we never get exact match thats why we cant jus lookup query_vector 
        # we need to find the closest match in our vocab
        best_ids = np.argsort(similarities)[::-1]
        results = []
        for index in best_ids:
            match_word = self.vocab.id_to_word[index]

            if match_word not in ignore_words:
                results.append((match_word,similarities[index])) # add the word and its confidance score

                if len(results) == top_n:
                    break
            

        return results 
    
    def analogy(self,word_a:str,word_b:str,word_c:str,top_n: int = 1):
        # A is to B as C is to ? (B-A+C)
        try:
            vec_a = self.get_vector(word_a)
            vec_b = self.get_vector(word_b)
            vec_c = self.get_vector(word_c)
        except ValueError as e:
            return str(e)
        
        # now perform this b-a+c and get a vector 
        query_vector = vec_b - vec_a + vec_c
        # normalize this vector to unit length 1
        query_vector = query_vector / np.linalg.norm(query_vector)

        # pass to the helper function and add vec a,b,c to the ignore list 
        return self._find_closest(
            query_vector,
            ignore_words={word_a,word_b,word_c},
            top_n=top_n
        )
    def evaluate_benchmark(self,filepath:str) -> float:
        correct = 0
        total_evaluated = 0
        total_skipped = 0

        print(f"\nScanning benchmark file: {filepath}...")
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(':'):
                    continue
                # 2. Parse the line into the 4 words
                words = line.strip().lower().split()
                if len(words) != 4:
                    continue
                word_a, word_b, word_c, expected_word = words
                # If any of the 4 words were dropped by your VocabManager, 
                # we mathematically cannot test this analogy. Skip it.
                if not all(w in self.vocab.word_to_id for w in [word_a, word_b, word_c, expected_word]):
                    total_skipped += 1
                    continue
                total_evaluated += 1
                # run the analogy 
                result = self.analogy(word_a, word_b, word_c, top_n=1)
                if isinstance(result, list) and len(result) > 0:
                    predicted_word = result[0][0]
                    if predicted_word == expected_word:
                        correct += 1

            if total_evaluated == 0:
                print("Warning: All benchmark words were Out-Of-Vocabulary. Score: 0.0%")
                return 0.0
            
            accuracy = (correct / total_evaluated) * 100
        
            print(f"--- Benchmark Results ---")
            print(f"Accuracy:  {accuracy:.2f}% ({correct}/{total_evaluated})")
            print(f"Skipped:   {total_skipped} questions (OOV)")
            print(f"-------------------------\n")
        
            return accuracy
