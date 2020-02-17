import tensorflow as tf

from helper_functions import *

load_data()

model = create_model()
get_batch_random(4)

#print(model.summary())