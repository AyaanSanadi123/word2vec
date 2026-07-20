import numpy as np
def sigmoid(x : np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x,-10,10)

    return 1.0/(1.0 + np.exp(-x_clipped))