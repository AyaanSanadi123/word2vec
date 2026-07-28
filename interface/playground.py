from custom_word2vec import CustomWord2Vec

def main():
    print("Initializing Custom Word2Vec Playground...")
    
    # 1. Load the model interface (automatically looks in the 'models/' folder)
    try:
        model = CustomWord2Vec()
    except FileNotFoundError as e:
        print(f"\n[Error]: {e}")
        print("Please ensure 'best_target_matrix.npy', 'word_to_id.json', and 'id_to_word.json' are placed in the 'models/' directory.")
        return

    print(f"Model loaded successfully! Vocabulary size: {model.vocab_size:,} words | Embedding dimensions: {model.embed_dim}\n")

    # 2. Test Vector Retrieval
    test_word = "anarchism" 
    if test_word in model.word_to_id:
        vector = model[test_word]
        print(f"--- 1. Vector Retrieval Test ---")
        print(f"Word: '{test_word}'")
        print(f"Vector shape: {vector.shape}")
        print(f"First 5 vector values: {vector[:5]}\n")
    
    # 3. Test Word Similarity
    print(f"--- 2. Cosine Similarity Test ---")
    word1, word2 = "anarchism", "state"
    try:
        sim = model.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {sim:.4f}\n")
    except KeyError as e:
        print(f"Skipped similarity test: {e}\n")

    # 4. Test Vector Arithmetic (Analogies)
    print(f"--- 3. Analogy Test (Vector Arithmetic) ---")
    print("Query: positive=['king', 'woman'], negative=['man']")
    try:
        results = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
        print("Top 5 results:")
        for word, score in results:
            print(f"  {word}: {score:.4f}")
    except KeyError as e:
        print(f"Analogy test skipped due to missing vocabulary words: {e}")
    print()

    # 5. Test Odd-One-Out
    print(f"--- 4. Odd-One-Out Test ---")
    sample_group = ['breakfast', 'cereal', 'dinner', 'car']
    print(f"Group: {sample_group}")
    try:
        odd_one = model.doesnt_match(sample_group)
        print(f"The odd one out is: '{odd_one}'")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()