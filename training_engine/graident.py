import numpy as np
from sigmoid import sigmoid
def train_step(target_id : int,context_id:int,negative_sample:list,matrices,learning_rate:float):
    # 1. pull the target and context vectors
    target_vector = matrices.get_target_vector(target_id)
    true_context_vector = matrices.get_context_vector(context_id)


    # we need to accumulate the updates of the target vector and all them at the end, for the math to stay stable 
    target_update = np.zeros_like(target_vector)
    # adding a loss variable to track the local loss
    loss = 0.0


    # positive sample 
    # forward pass
    pos_dot_product = np.dot(target_vector,true_context_vector)
    pos_prediction = sigmoid(pos_dot_product)

    # calculate the positive loss
    loss -= np.log(pos_prediction + 1e-10)


    # back-pass, calculate gradient 
    pos_gradient = (1.0 - pos_prediction) * learning_rate

    # now, update the target_update and adjust the context vector
    target_update += pos_gradient * true_context_vector
    true_context_vector += pos_gradient * target_vector

    # negative sampling 
    for neg_id in negative_sample:
        # get the vectors from their ids
        neg_context_vector = matrices.get_context_vector(neg_id)

        # forward pass, 
        neg_dot_product = np.dot(target_vector,neg_context_vector)
        neg_prediction = sigmoid(neg_dot_product)

        # calculate the negative loss 
        loss -= np.log((1.0 - neg_prediction) + 1e-10)
        # back pass 
        neg_gradient = (0.0 - neg_prediction) * learning_rate

        # Accumulate Target update, and instantly update Negative Context vector
        target_update += neg_gradient * neg_context_vector
        neg_context_vector += neg_gradient * target_vector

    target_vector += target_update
    return loss




