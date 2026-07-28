Word2Vec from Scratch (Skip-Gram with Negative Sampling)A complete, end-to-end implementation of the landmark 2013 Word2Vec paper ("Distributed Representations of Words and Phrases and their Compositionality" by Mikolov et al.) built completely from scratch.This repository replicates the core mechanics of the original C-based Word2Vec engine, pairing a high-performance C training backend with a lightweight Python inference pipeline that emulates gensim.Project Overview & BackgroundThe original 2013 paper revolutionized Natural Language Processing (NLP) by introducing efficient methods for learning high-quality vector representations of words from massive unannotated corpora. Instead of treating words as discrete, sparse IDs (one-hot encoding), Word2Vec maps words into a continuous geometric space where semantic meaning is captured by spatial proximity.This project recreates that architecture from the ground up:The Data Pipeline: Custom text preprocessing, vocabulary construction, sub-sampling of frequent words, and dataset generation tailored for context-target window mapping.The Training Engine: Implements the Skip-gram architecture with Negative Sampling (SGNS) optimized for performance and low memory overhead.The Inference Engine: A modular Python class that loads raw embedding matrices and vocabulary dictionaries, providing blazing-fast dot-product cosine similarity searches, vector arithmetic (analogies), and centroid-based outlier detection.Repository StructurePlaintextword2vec/
├── c_engine/                # Core C training source code and build files
├── data/                    # Dataset processing scripts and corpus loaders
├── interface/               # Python inference API & playground
│   ├── models/              # External model weights (.npy & JSONs) - Git-ignored
│   ├── custom_word2vec.py   # Gensim-emulating inference class
│   └── playground.py        # Interactive terminal testing script
└── README.md                # Project documentation
What to ExpectHigh-Performance C Training: The model trains directly on raw text corpora using optimized C routines, yielding dense float matrices that capture complex linguistic relationships (e.g., semantic clustering, syntactic analogies).Vector Space Magic: Once trained, you can perform vector arithmetic matching the famous 2013 paper results:$$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$Plug-and-Play Inference: You don't need heavy C dependencies to use the trained model. The interface/ module provides a drop-in, lightweight Python wrapper requiring only numpy.Quick Start & Setup1. Clone the RepositoryBashgit clone https://github.com/your-username/word2vec-from-scratch.git
cd word2vec-from-scratch/interface
2. Install DependenciesBashpip install numpy
3. Download Model ArtifactsBecause the raw embedding matrix weights are large (~900 MB), they are hosted externally rather than tracked in Git:Download the model files (best_target_matrix.npy, word_to_id.json, and id_to_word.json) from the Google Drive link below:👉 Download Models Folder (Google Drive)Create a folder named models/ inside your interface/ directory and place the downloaded files there.4. Run the PlaygroundTest out vector lookups, similarities, analogies, and odd-one-out detection:Bashpython playground.py
Python API Usage (CustomWord2Vec)You can easily integrate the model into your own scripts just like a standard machine learning package:Pythonfrom custom_word2vec import CustomWord2Vec

# Load the inference engine
model = CustomWord2Vec()

# 1. Retrieve a raw embedding vector
vector = model['anarchism']
print("Vector shape:", vector.shape)

# 2. Compute Cosine Similarity between two words
sim = model.similarity('anarchism', 'state')
print(f"Similarity: {sim:.4f}")

# 3. Vector Arithmetic / Analogies
analogies = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("Analogies result:", analogies)

# 4. Find the Odd-One-Out
odd_one = model.doesnt_match(['breakfast', 'cereal', 'dinner', 'car'])
print("Odd one out:", odd_one)