#include <stdio.h>
#include <math.h>
#include <stdlib.h>

float fast_sigmoid(float x)
{
    if (x <= -6.0f)
        return 0.0f;
    if (x >= 6.0f)
        return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

int train_epoch(
    int *corpus, // this is the flat 1D array that has the index values
    int corpus_len,
    float *context_matrix, float *target_matrix,
    int vocab_size, int embed_size,
    int *unigram_table,                 // this is the 10 million slots array, each containing the index value of the word it represents
    int unigram_size,                   // this is prolly 10 million
    int window_size, int num_negatives, // this is prolly the number of negative samples usually 5? the K value in the equation
    float initial_lr, long long total_expected_pairs, long long global_pairs_processed,
    float *discard_probs)
{

    // the first course of action is to create a target graident variable
    // its the same logic of target_update variable in the python implementation
    // we need some place to hold the graident values so we can do a cumilative update on the target vector
    float *target_update = (float *)malloc(embed_size * sizeof(float));
    float curr_lr = initial_lr;
    static unsigned long long next_random = 1;
    // loop through the corpus
    for (int i = 0; i < corpus_len; i++)
    {
        // get the word id
        int word_id = corpus[i];

        // see if its the end of a sentence
        if (word_id == -1)
            continue;

        // check its discard_prob and run its luck
        next_random = next_random * (unsigned long long)25214903917 + 11;
        float rand_val = (float)(next_random & 0xFFFF) / 65536.0f;

        // if the number is lower than the discard_prob, then drop it
        if (rand_val < discard_probs[word_id])
            continue;

        // now if the word survived till here, we can start to generate the sliding window pairs
        // get a number bw 1 and window size
        next_random = next_random * (unsigned long long)25214903917 + 11;
        int dynamic_window = (next_random % window_size) + 1;

        for (int direction = -1; direction <= 1; direction += 2)
        {
            for (int step = 1; step <= dynamic_window; step++)
            {
                int j = i + (direction * step); // i is the center words index, and j is the current context word
                if (j < 0 || j >= corpus_len)
                    break;
                int context_id = corpus[j];
                if (context_id == -1)
                    break; // check if we hit the end

                // at this point we found a valid pair,
                // update the lr
                global_pairs_processed++;
                if (global_pairs_processed % 10000 == 0) // update every 10,000 words
                {
                    float progress = (float)global_pairs_processed / (float)total_expected_pairs;
                    curr_lr = initial_lr * (1.0f - progress);

                    if (curr_lr < initial_lr * 0.0001f)
                    {
                        curr_lr = initial_lr * 0.0001f;
                    }
                }
                // skip gram negative sampling

                // clear the entire target_update for a fresh batch of values
                for (int d = 0; d < embed_size; d++)
                    target_update[d] = 0.0f;

                // now run a loop, 1 positive pair and K negative pairs
                for (int n = 0; n <= num_negatives; n++)
                {
                    int current_context_id;
                    int label;

                    if (n == 0)
                    {
                        // this is the positive pair
                        current_context_id = context_id;
                        label = 1;
                    }
                    else
                    {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        current_context_id = unigram_table[(next_random >> 16) % unigram_size];
                        // get a random index from the unigram table

                        if (current_context_id == word_id)
                            continue; // make sure the negative word is not the target word itself
                        label = 0;
                    }

                    // now time to get the vectors from the matrices
                    // because its not a 2D array anymore,
                    // its a 1D flat array, so to find the vector in the matrix do, row_id * embed_size
                    // this gives us the starting point of the vector
                    // store that in a pointer
                    float *v_target = &target_matrix[word_id * embed_size];
                    float *u_context = &context_matrix[current_context_id * embed_size];
                    // now uk where to start updating the values
                    // forward pass (dot product)
                    float dot_product = 0.0f;
                    for (int d = 0; d < embed_size; d++)
                    {
                        dot_product += v_target[d] * u_context[d];
                    }
                    // apply sigmoid
                    float z = fast_sigmoid(dot_product);
                    // get the graident value
                    float gradient = (label - z) * curr_lr;

                    for (int d = 0; d < embed_size; d++)
                    {
                        target_update[d] += gradient * u_context[d]; // accumilate the values
                        // update the context vector right away
                        u_context[d] += gradient * v_target[d];
                    }
                }
                // update the target vector
                // we are redeclaring because the previous one was local to the for loop
                float *v_target = &target_matrix[word_id * embed_size];
                for (int d = 0; d < embed_size; d++)
                {
                    v_target[d] += target_update[d];
                }
            }
        }
    }
    free(target_update);
    return global_pairs_processed;
}