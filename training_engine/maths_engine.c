#include<stdio.h>
#include<math.h>
#include<stdlib.h>

float fast_sigmoid(float x){
    if (x <= -6.0f) return 0.0f;
    if(x >= 6.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

void train_epoch(
int* corpus, // this is the flat 1D array that has the index values 
int corpus_len,
float* context_matrix,float* target_matrix,
int vocab_size,int embed_size,
int* unigram_table, // this is the 10 million slots array, each containing the index value of the word it represents
int unigram_size, // this is prolly 10 million
int window_size,int num_negatives, // this is prolly the number of negative samples usually 5? the K value in the equation
float current_lr,float* discard_probs
){
    // the first course of action is to create a target graident variable 
    // its the same logic of target_update variable in the python implementation 
    // we need some place to hold the graident values so we can do a cumilative update on the target vector 
    float * target_update = (float*) malloc(embed_size * sizeof(float));
    
    // loop through the corpus 
    for (int i = 0; i < corpus_len; i++)
    {
       // get the word id 
       int word_id = corpus[i];

       // see if its the end of a sentence 
       if (word_id == -1) continue;

       // check its discard_prob and run its luck 
       float rand_val = (float)rand() / (float)RAND_MAX; // get a number bw 0-1

       // if the number is lower than the discard_prob, then drop it
       if (rand_val < discard_probs[word_id]) continue;

       // now if the word survived till here, we can start to generate the sliding window pairs
       // get a number bw 1 and window size 
       int dynamic_window = (rand() % window_size) + 1;

       for(int direction = -1;direction<=1;direction+=2){
            for(int step = 1; step<= dynamic_window; step++){
                int j = i + (direction * step); // i is the center words index, and j is the current context word 
                if (j < 0 || j >= corpus_len) break;
                int context_id = corpus[j];
                if (context_id == -1) break; // check if we hit the end 

                // skip gram negative sampling 
                
                // clear the entire target_update for a fresh batch of values
                for(int d=0;d<embed_size;d++)target_update[d] = 0.0f;
                
                // now run a loop, 1 positive pair and K negative pairs 
                for(int n = 0; n<= num_negatives;n++){
                    int current_context_id;
                    int label;

                    if(n == 0){
                        // this is the positive pair 
                        current_context_id = context_id;
                        label = 1;
                    }else{
                        int rand_id = rand() % unigram_size;
                        // get a random index from the unigram table
                        current_context_id = unigram_table[rand_id];

                        if (current_context_id == word_id) continue; // make sure the negative word is not the target word itself
                        label = 0;            
                    }

                    // now time to get the vectors from the matrices 
                    // because its not a 2D array anymore,
                    // its a 1D flat array, so to find the vector in the matrix do, row_id * embed_size 
                    //this gives us the starting point of the vector 
                    // store that in a pointer 
                    float * v_target = &target_matrix[word_id * embed_size];
                    float* u_context = &context_matrix[current_context_id * embed_size];
                    // now uk where to start updating the values 
                    // forward pass (dot product)
                    float dot_product = 0.0f;
                    for(int d = 0; d<embed_size;d++){
                        dot_product+= v_target[d]*u_context[d];
                    }
                    // apply sigmoid
                    float z = fast_sigmoid(dot_product);
                    // get the graident value
                    float gradient = (label - z) * current_lr;

                    for(int d = 0;d<embed_size;d++){
                        target_update[d] += gradient * u_context[d]; // accumilate the values 
                        // update the context vector right away 
                        u_context[d] +=  gradient * v_target[d];
                    }
                }
               // update the target vector 
               // we are redeclaring because the previous one was local to the for loop
               float* v_target = & target_matrix[word_id * embed_size];
               for (int d = 0; d < embed_size; d++) {
                    v_target[d] += target_update[d];
                }

                
            }
       }   
    }
    free(target_update);
}