import random
def get_negative_samples(unigram_table : list,num_samples : int = 5) -> list:

    negative_samples = []
    table_size = len(unigram_table) # this is about 10 million bits
    for _ in range(num_samples):
        random_index = random.randint(0,table_size-1)

        negative_samples.append(unigram_table[random_index])

    return negative_samples