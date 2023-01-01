import numpy as np
import tensorflow as tf

def set_numpy_seed(seed: int = 15):
    np.random.seed(seed)
    
def set_tensorflow_seed(seed: int = 15):
    tf.random.set_seed(seed)

def set_seeds(seed: int = 15):
    set_numpy_seed(seed)
    set_tensorflow_seed(seed)
