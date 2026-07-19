# import layer-1 
from data_preprocessing.tokenizer import TextTokenizer
from data_preprocessing.phraser import PhraseMiner
from data_preprocessing.vocabulary import VocabManager

# import layer-2 
from data_loader.sliding_window import generate_training_pairs
from embedding_layer.matrices import Word2VecMatrices


# import layer - 3
from training_engine.sigmoid import sigmoid
from training_engine.sampler import get_negative_samples
from training_engine.graident import train_step


"""
LAYER-1 : the goal here is to input raw text and output a mathematically optimised series of data-
structures that represent the cleaned and pruned data.

1. TextTokenizer : reads raw text and spits out a list of sentences 
2. PhraseMiner : mergers commonly occuring tokens into new tokens 
3. VocabManager : this builds the dictionarys, also finds the probability
of dropping frequent words, builds the 10,000,000 slot array 

"""
def run_data_pipeline(raw_text : str):
    print("--- Phase 1: Data Pipeline ---")\
    
    # tokenize the text 
    print("1. Tokenizing text...")
    tokenizer = TextTokenizer()
    raw_corpus = tokenizer.tokenize(raw_text)

    # building new words (mining phase)
    print("2. Mining phrases...")
    pharser = PhraseMiner(thresholds=[100, 50], delta=5)
    mined_corpus = pharser.fit_transform(raw_corpus)

    # delete the raw corpus
    del raw_corpus

    # 3. Build the Vocabulary and Math Tables
    print("3. Building Vocabulary and Probability Tables...")
    vocab = VocabManager(min_count=5,subsample_t = 1e-5)
    
    # Pass the newly mined corpus into the vocab manager
    vocab.build_vocab(mined_corpus)
    vocab.calculate_subsampling()
    vocab.build_unigram_table()
    
    print(f"Phase 1 Complete. Pruned Vocabulary Size: {len(vocab.word_to_id)}")
    
    return mined_corpus, vocab


# phase-2 & 3 combined 
'''
phase-3, this phase has 4 major responsibilities 
1. Epochs (keeping track of how many trainig cycles to complete)
2. data streaming (using the generator to stream the data in pairs)
3. calling the K negative samples 
4. running the forward and backward passes 

we have 3 loops 
1. the outer loop (epochs)
2. the middle loop -> the generator looks at a sentence and picks a centre word, 
3. the inner loop -> for every valid pair in that sentence, we get negative samples and train the network

'''


def execute_training(
        mined_corpused:list,
        vocab:VocabManager,
        embed_dim : int = 300,
        epochs : int = 5,
        learning_rate : float = 0.025,
        negative_k:int = 5,
        window_size: int = 5
):
    print("\n--- Phase 2: Memory Allocation ---")
    vocab_size = len(vocab.word_to_id)
    matrices = Word2VecMatrices(vocab_size,embed_dim)

    print("\n--- Phase 3: The Training Engine ---")

    # loop - 1 (epochs)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} starting...")

        # pair generator 
        pair_generator = generate_training_pairs(mined_corpused,vocab,window_size)

        pair_count = 0
        for target_id,context_id in pair_generator:

            # get the negative sample 
            negative_samples= get_negative_samples(vocab.unigram_table,negative_k)

            # start the maths 
            train_step(target_id,context_id,negative_samples,matrices,learning_rate)

            pair_count += 1

    print("\nTraining Complete! 🚀")
    return matrices 



if __name__ == "__main__":

    mined_corpus, vocab = run_data_pipeline(
        raw_text=,
        phraser_thresholds=[100, 50],
        vocab_min_count=5
    )
    
    # 3. Run Phase 2 & 3
    trained_matrices = execute_training(
        mined_corpus=mined_corpus,
        vocab=vocab,
        embed_dim=300,
        epochs=5,
        learning_rate=0.025
    )


