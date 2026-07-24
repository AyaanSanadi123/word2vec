import os
import numpy as np 
import json
from data_preprocessing.tokenizer import TextTokenizer
from data_preprocessing.phraser import StreamingPhraseMiner
from data_preprocessing.vocabulary import VocabManager
from data_preprocessing.c_helper import encode_corpus
from utils.streaming import build_clean_corpus

def test_pipeline(raw_filepath:str,clean_filepath:str,phrased_filepath:str):
    try:

        print("\n[TEST 1] Testing TextTokenizer and Clean Stream...")
        tokenizer = TextTokenizer()
        build_clean_corpus(raw_filepath,clean_filepath,tokenizer)

        with open(clean_filepath,'r') as f:
            # what is the point of assignin the data to this object?
            clean_content = f.read()

        # test the miner 
        miner = StreamingPhraseMiner(thresholds=[15,5],delta=2)
        miner.fit_transform(clean_filepath,phrased_filepath)
        assert os.path.exists(phrased_filepath), "Phrased file failed to generate."
        assert not os.path.exists("temp_A.txt"), "Temp file A leaked."
        assert not os.path.exists("temp_B.txt"), "Temp file B leaked."

        with open(phrased_filepath,'r') as f:
            # again point-less assignment, our test volume is to small
            # the must be to see if the files work and if the two file swapping logic works well...
            phrased_content = f.read()

        print("\n[TEST 3] Testing VocabManager & Array Memory Alignment...")
        vocab = VocabManager(min_count=2,subsample_t=1e-4)
        vocab.build_vocab(phrased_filepath)
        vocab.calculate_subsampling()
        vocab.build_unigram_table()

        vocab_json_path = os.path.join("test", "word_to_id.json")
        with open(vocab_json_path, 'w', encoding='utf-8') as f:
            json.dump(vocab.word_to_id, f, indent=4)
        print(f"✅ Vocabulary dictionary successfully saved to {vocab_json_path}")

        print("\n[TEST 4] Testing Corpus Encoder...")
        c_ready_corpus = encode_corpus(phrased_filepath, vocab)

        assert c_ready_corpus.dtype == np.int32, "Corpus is not contiguous int32."
        assert -1 in c_ready_corpus, "End-of-sentence marker (-1) is missing from encoded corpus."

        print("\n🎉 ALL MODULAR TESTS PASSED! The Phase 1 streaming pipeline is structurally sound.")
    finally:
        print("\n[TEARDOWN] Removing dummy files to leave workspace clean...")
        


if __name__ == "__main__":
    raw_filepath = 'test/rawfile.txt'
    clean_filepath = 'test/cleanfile.txt'
    phrased_filepath = 'test/phrasedfile.txt'
    test_pipeline(raw_filepath,clean_filepath,phrased_filepath)
