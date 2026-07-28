import numpy as np
import os
import json 

class CustomWord2Vec:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models")
        matrix_path = os.path.join(model_dir, "best_target_matrix.npy")
        word_to_id_path = os.path.join(model_dir, "word_to_id.json")
        id_to_word_path = os.path.join(model_dir, "id_to_word.json")

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Could not find model weights at {matrix_path}.")
        if not os.path.exists(word_to_id_path) or not os.path.exists(id_to_word_path):
            raise FileNotFoundError(f"Could not find vocabulary JSON files in {model_dir}.")

        print("Loading embedding matrix...")
        self.vectors = np.load(matrix_path)
        
        print("Loading vocabulary dictionaries...")
        with open(word_to_id_path, "r", encoding="utf-8") as f:
            self.word_to_id = json.load(f)
            
        with open(id_to_word_path, "r", encoding="utf-8") as f:
            # JSON keys are saved as strings, so we convert them back to integers for id_to_word
            raw_id_to_word = json.load(f)
            self.id_to_word = {int(k): v for k, v in raw_id_to_word.items()}

        self.vocab_size = len(self.word_to_id)
        self.embed_dim = self.vectors.shape[1] if len(self.vectors.shape) > 1 else self.vectors.shape[0]

        if len(self.vectors.shape) == 1:
            self.vectors = self.vectors.reshape(self.vocab_size, self.embed_dim)

        print("Normalizing embedding vectors for fast similarity search...")
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_vectors = self.vectors / norms

    def __getitem__(self, word):
        
        if word not in self.word_to_id:
            raise KeyError(f"Word '{word}' not found in vocabulary.")
        idx = self.word_to_id[word]
        return self.vectors[idx]

    def get_vector(self, word):
        
        return self.__getitem__(word)

    def similarity(self, word1, word2):
        
        if word1 not in self.word_to_id or word2 not in self.word_to_id:
            raise KeyError("One or both words not found in vocabulary.")
        
        id1 = self.word_to_id[word1]
        id2 = self.word_to_id[word2]
        
        v1 = self.normalized_vectors[id1]
        v2 = self.normalized_vectors[id2]
        
        
        return float(np.dot(v1, v2))

    def most_similar(self, positive=[], negative=[], topn=10):
        
        query_vector = np.zeros(self.embed_dim, dtype=np.float32)

        for word in positive:
            if word in self.word_to_id:
                query_vector += self.normalized_vectors[self.word_to_id[word]]
            else:
                raise KeyError(f"Positive word '{word}' not found in vocabulary.")

        for word in negative:
            if word in self.word_to_id:
                query_vector -= self.normalized_vectors[self.word_to_id[word]]
            else:
                raise KeyError(f"Negative word '{word}' not found in vocabulary.")

        # Normalize the resulting query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm

        # Compute cosine similarity against all words via dot product
        similarities = np.dot(self.normalized_vectors, query_vector)

        # Exclude exact query input words from results
        exclude_ids = {self.word_to_id[w] for w in positive + negative if w in self.word_to_id}
        
        # Sort indices by highest similarity score
        best_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in best_indices:
            if idx in exclude_ids:
                continue
            word = self.id_to_word[idx]
            results.append((word, float(similarities[idx])))
            if len(results) == topn:
                break

        return results
    def doesnt_match(self, words):

        valid_words = [w for w in words if w in self.word_to_id]
        if not valid_words:
            raise ValueError("None of the provided words are in the vocabulary.")

        # Extract vectors and compute the mean vector (centroid) of the group
        word_vectors = np.array([self.normalized_vectors[self.word_to_id[w]] for w in valid_words])
        centroid = np.mean(word_vectors, axis=0)
        
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid /= centroid_norm

        # Find the word with the lowest similarity to the group centroid
        lowest_sim = float('inf')
        odd_one_out = None

        for w in valid_words:
            sim = np.dot(self.normalized_vectors[self.word_to_id[w]], centroid)
            if sim < lowest_sim:
                lowest_sim = sim
                odd_one_out = w

        return odd_one_out