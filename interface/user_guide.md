Setup & Installation
1. Prerequisites
Ensure you have Python installed along with NumPy:

Bash
pip install numpy
2. Download Model Artifacts
Because the embedding matrix file is large (~900 MB), it is hosted externally:

Download the required model files (best_target_matrix.npy, word_to_id.json, and id_to_word.json) from the link below:
👉 [Download Models Folder (Google Drive)](https://drive.google.com/drive/folders/1k5CVdTa7c3qbOQj4c2ggqbKXmaFKqmkG?usp=sharing)

Create a folder named models/ inside your interface/ directory and drop the downloaded files into it.

Usage Guide (playground.py)
You can interact with your model programmatically just like a standard machine learning library by running:

Bash
python playground.py
Example Code Snippet
Python
from custom_word2vec import CustomWord2Vec

# Load the inference model
model = CustomWord2Vec()

# 1. Retrieve a raw embedding vector
vector = model['anarchism']
print("Vector shape:", vector.shape)

# 2. Compute Cosine Similarity between two words
sim = model.similarity('anarchism', 'state')
print(f"Similarity: {sim:.4f}")

# 3. Vector Arithmetic / Analogies (e.g., king - man + woman = ?)
analogies = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("Analogies result:", analogies)

# 4. Find the Odd-One-Out
odd_one = model.doesnt_match(['breakfast', 'cereal', 'dinner', 'car'])
print("Odd one out:", odd_one)
API Reference (CustomWord2Vec)
__init__(model_dir=None): Automatically locates the models/ folder, loads the NumPy matrix and JSON dictionaries, reshapes flat arrays, and L2-normalizes vectors for instant dot-product cosine similarity searches.

__getitem__(word) / get_vector(word): Returns the 1D NumPy embedding vector for a specified word.

similarity(word1, word2): Computes the cosine similarity score between two words.

most_similar(positive=[], negative=[], topn=10): Performs vector arithmetic to resolve analogies and ranks nearest neighbors.

doesnt_match(words): Identifies the outlier word in a list by evaluating distance from the group's semantic centroid.
"""