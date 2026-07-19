import random 

def generate_training_pairs(corpus:list,vocab,max_window_size : int = 5):
    
    
    for sentence in corpus:
        filtered_word_ids = []

        for word in sentence: # apply sub-sampling 
            # we checkf if the word is in word_to_id and if not we skip it
            if word not in vocab.word_to_id:
                continue

            #else 
            discard_prob = vocab.discard_probs[word] # if the word does exist, get its dis_prob
            if random.random() < discard_prob:
                # drop the word 
                continue

            # the word survives 
            word_id = vocab.word_to_id[word]

            #add it to the filtered word ids list 
            filtered_word_ids.append(word_id) 

        
        # so at this point, we have one sentence worth of id's in our list 
        # now the goal is to, get the centre word
        # get a random size bw 1 - c
        # return pairs 
        for i,target_id in enumerate(filtered_word_ids):
            dynamic_window = random.randint(1,max_window_size)

            start_index = max(0,i-dynamic_window)
            end_index = min(len(filtered_word_ids), i + dynamic_window + 1)
            for j in range(start_index, end_index):
                # We don't want to pair the target word with itself
                if i != j:
                    context_id = filtered_word_ids[j]
                    # skip-gram model, we need to pair j context words with one target word
                    #if u are gonna make a cbow model, reverse this logic 
           
                    yield (target_id, context_id)