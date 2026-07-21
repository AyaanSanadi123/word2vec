import numpy as np 

# this converts corpus into a flat array with their values being the index ids

def encode_corpus(mined_corpus:list,vocab) -> np.ndarray:
    print("Encoding 2D list into a 1D array of index values")
    flat_corpus = [] # this will store the values (12,1,-1,11,16,-1)

    for sentence in mined_corpus:
        sentence_ids = []

        for word in sentence:
            if word in vocab.word_to_id:
                sentence_ids.append(vocab.word_to_id[word])

        
        # check is the sentence has atleast 2 words 
        if len(sentence_ids) > 1:
            flat_corpus.extend(sentence_ids)
            flat_corpus.append(-1) # add the end of sentence 
    
    # we need to pack this flat_corpus into a contiguous block for c 
    c_ready_corpus = np.array(flat_corpus,dtype=np.int32)
    return c_ready_corpus


# writing a function to convert discard_prob_dict to flat array 
def dict_to_c_array(discard_prob_dict,word_to_id):
    vocab_size = len(word_to_id)
    c_array = np.zeros(vocab_size,dtype=np.float32)

    for word,prob in discard_prob_dict.items():
        c_array[word_to_id[word]] = prob

    return c_array