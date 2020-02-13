import tensorflow as tf

from helper_functions import *

model = create_model((128, 128, 3))

print(model.summary())