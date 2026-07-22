import numpy as np 

class Word2VecMatrices:
    def __init__(self,vocab_size: int,embed_dim : int = 300):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim



        init_bound = 0.5/self.embed_dim
        # Matrix V -> this is where each word is the target word
        self.target_matrix = np.random.uniform(
            low = -init_bound,
            high = init_bound,
            size =(self.vocab_size,self.embed_dim)
        ).astype(np.float32)
        # this is U
        self.context_matrix = np.zeros((self.vocab_size,self.embed_dim),dtype=np.float32)

    def get_target_vector(self,target_id:int) -> np.ndarray:
        return self.target_matrix[target_id]
    def get_context_vector(self,context_id:int) -> np.ndarray:
        return self.context_matrix[context_id]
    
