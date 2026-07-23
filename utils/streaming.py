'''
we get a list of lists,
the goal is to 
1. get unigram and bigram counts to be stored in the RAM
2. each list in the lists, is one complete sentence, place it in the new file with \n to establish one line = one sentence 
'''


from collections import defaultdict
from data_preprocessing.tokenizer import TextTokenizer


# its only utility is to take data from the raw file, clean it and put it into a temp file
def build_clean_corpus(raw_filepath:str,clean_filepath:str,tokenizer:TextTokenizer):
    print("Pass-1, cleaning and counting")
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)

    # bigram cleaning cutoff 
    capacity_limit = 25_000_000

    print(f"Streaming {raw_filepath} to {clean_filepath}...")
    with open(raw_filepath,'r',encoding='utf-8') as raw_file,\
        open(clean_filepath,'w',encoding='utf-8') as clean_file:
            for line in raw_file:
                  sentences = tokenizer.tokenize(line)
                  for sentence in sentences:
                        if sentence:
                              clean_file.write(" ".join(sentence) + "\n")

    print("✅ Base corpus completely cleaned and standardized!")


def get_stream_counts(clean_filepath:str):
    print(f"--- Scanning counts for {clean_filepath} ---")
    
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    capacity_limit = 25_000_000

    with open(clean_filepath,'r',encoding='utf-8') as clean_file:
          for line in clean_file:
                sentence = line.strip().split()

                if not sentence:
                      continue
                
                for w in sentence:
                      unigram_counts[w] +=1

                for i in range(len(sentence) - 1):
                        w1 = sentence[i]
                        w2 = sentence[i+1]
                        bigram_counts[(w1, w2)] += 1


                if len(bigram_counts) >= capacity_limit:
                    print(f"⚠️ Bigram capacity hit. Triggering purge...")
                    garbage_keys = [k for k, v in bigram_counts.items() if v == 1]
                    for k in garbage_keys:
                        del bigram_counts[k]
                    print(f"✅ Purge complete. Dictionary shrank to {len(bigram_counts)} pairs.")
                    
    return unigram_counts,bigram_counts