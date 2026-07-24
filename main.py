import numpy as np
import json
import os
from datetime import datetime
import time
# import layer-1
from data_preprocessing.tokenizer import TextTokenizer
from data_preprocessing.phraser import PhraseMiner,StreamingPhraseMiner
from data_preprocessing.vocabulary import VocabManager
from utils.streaming import build_clean_corpus
# import layer-2 
from data_loader.sliding_window import generate_training_pairs
from embedding_layer.matrices import Word2VecMatrices


# import layer - 3
from training_engine.sigmoid import sigmoid
from training_engine.sampler import get_negative_samples
from training_engine.graident import train_step
from training_engine.schedular import LearningRateScheduler

# testing layer 
from testing.instrinsic_evaluator import Word2VecEvaluator



# layer-1 for c
from data_preprocessing.c_helper import encode_corpus

# layer 3 for c
from training_engine.c_wrapper import run_c_epoch



"""
LAYER-1 : the goal here is to input raw text and output a mathematically optimised series of data-
structures that represent the cleaned and pruned data.

1. TextTokenizer : reads raw text and spits out a list of sentences 
2. PhraseMiner : mergers commonly occuring tokens into new tokens 
3. VocabManager : this builds the dictionarys, also finds the probability
of dropping frequent words, builds the 10,000,000 slot array 

"""
def run_data_pipeline(raw_filepath:str,clean_filepath:str,phrased_filepath:str):
    print("--- Phase 1: Streaming Data Pipeline ---")

    # clean and tokenizer
    if not os.path.exists(clean_filepath):
        print("1. Cleaning and Tokenizing text stream...")
        tokenizer = TextTokenizer()
        build_clean_corpus(raw_filepath,clean_filepath,tokenizer)
    else:
        print(f"⏩ {clean_filepath} found. Skipping tokenizer phase.")

    # mining phase 
    if not os.path.exists(phrased_filepath):
        print("2. Mining phrases via disk-streaming...")
        miner = StreamingPhraseMiner()
        miner.fit_transform(clean_filepath,phrased_filepath)
    else:
        print(f"⏩ {phrased_filepath} found. Skipping phrase mining.")

    # building Vocabulary and unigram table 
    print("\n3. Building Vocabulary and Probability Tables...")
    vocab = VocabManager(min_count=5, subsample_t=1e-5)
    
    vocab.build_vocab(phrased_filepath)
    vocab.calculate_subsampling()
    vocab.build_unigram_table()

    # encoding file into c array 
    c_ready_corpus = encode_corpus(phrased_filepath,vocab)
    print(f"✅ Phase 1 Complete. Pruned Vocabulary Size: {len(vocab.word_to_id):,}")
    return c_ready_corpus,vocab

    
    
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

def log_experiment(hyperparameters: dict, best_accuracy: float,  total_training_time: float,filepath: str = "experiments.json"):
    """Appends the results of a training run to a JSON ledger."""
    
    # 1. Build the data dictionary for this specific run
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": hyperparameters,
        "metrics": {
            "best_accuracy_percent": best_accuracy,
            "total_training_time_seconds": round(total_training_time, 2),
            
        },
        "artifact_path": "models/best_target_matrix.npy"
    }

    # 2. Safely load existing history, or create a new list if the file doesn't exist
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = [] # In case the file is empty or corrupted
    else:
        history = []

    # 3. Append the new run and save it back to the disk
    history.append(entry)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)


def execute_training(
        encoded_corpus: np.ndarray,
        vocab : VocabManager,
        embed_dim : int = 300,
        epochs:int = 5,
        learning_rate : float = 0.025,
        negative_k : int = 5,
        window_size : int = 5
):
    print("\n--- Phase 2: Memory Allocation ---")
    vocab_size = len(vocab.word_to_id)
    matrices = Word2VecMatrices(vocab_size, embed_dim)

    # calculate the estimated total pairs and set a counter for the processed pairs 
    total_words = len(encoded_corpus)
    estimated_total_pairs = total_words * window_size * epochs
    global_pairs_processed = 0

    print("\n--- Phase 3: The C-Engine Autopilot ---")
    best_accuracy = -1.0
    patience_counter = 0
    patience_limit = 3
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs} starting in C...")
        global_pairs_processed = run_c_epoch(
            corpus=encoded_corpus,
            target_matrix=matrices.target_matrix,
            context_matrix=matrices.context_matrix,
            vocab_size=vocab_size,
            embed_size=embed_dim,
            unigram_table=vocab.unigram_table,
            window_size=window_size,
            num_negatives=negative_k,
            initial_lr=learning_rate,
            total_expected_pairs=estimated_total_pairs,
            global_pairs_processed=global_pairs_processed,
            discard_probs=vocab.discard_probs
        )

        print(f"✅ Epoch {epoch + 1} Math Completed.")
        print("Running Benchmark Evaluation...")
        evaluator = Word2VecEvaluator(matrices.target_matrix, vocab) 
        epoch_accuracy = evaluator.evaluate_benchmark("testing/questions-words.txt")
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            patience_counter = 0
            
            os.makedirs("models", exist_ok=True)
            np.save("models/best_target_matrix.npy", matrices.target_matrix)
            print(f"🌟 New best model saved! Accuracy: {best_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience_limit}")
            
            if patience_counter >= patience_limit:
                print(f"\n🛑 Early stopping triggered! Model hasn't improved in {patience_limit} epochs.")
                print(f"Keeping the best matrix with {best_accuracy:.2f}% accuracy.")
                break 

    total_training_time = time.time() - start_time       
    print("\n📝 Logging experiment results to experiments.json...")
    run_configs = {
        "embed_dim": embed_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "negative_k": negative_k,
        "window_size": window_size
    }
    log_experiment(run_configs, best_accuracy,total_training_time)               
    print("\nTraining Complete! 🚀")
    
    return matrices




if __name__ == "__main__":
    RAW_FILE = "data/text8.txt"
    CLEAN_FILE = "data/text8_clean.txt"
    PHRASED_FILE = "data/text8_phrased.txt"

    encoded_corpus,vocab = run_data_pipeline(RAW_FILE,CLEAN_FILE,PHRASED_FILE)
    
    # 2. Run Phase 2 & 3
    trained_matrices = execute_training(
        encoded_corpus=encoded_corpus,
        vocab=vocab,
        embed_dim=300,
        epochs=5,
        learning_rate=0.025
    )