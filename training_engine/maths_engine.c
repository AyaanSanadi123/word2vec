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
                

               
                
            }
       }


       
       
       
    }
    
}