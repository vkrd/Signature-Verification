import tensorflow as tf
import keras
import numpy as np
import os
import random
import time

input_shape = (128, 128, 3)
train_users = []
test_users = []

def triplet_loss(y_actual, y_pred, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[1])), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[2])), axis=-1)

    loss = tf.reduce_sum(tf.maximum(pos_dist-neg_dist+alpha, 0.0))

    return loss

def create_model():
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None,
                                                input_shape=input_shape, pooling=None, classes=128)

    # Normalization Layer
    output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(model.output)

    return keras.Model(model.input, output)

def load_data():
    print("Loading data...")
    global train_users, test_users
    train_users = [f.path for f in os.scandir("./data/train") if (not f.path[-1] == "g")]
    test_users = [f.path for f in os.scandir("./data/test") if (not f.path[-1] == "g")]
    print(train_users)
    print("Loaded all data")


def get_batch_random(batch_size, set="train"):
    start = time.time()
    if (set == "train"):
        width, height, channels = input_shape

        retTriplet = [np.zeros((batch_size, width, height, channels)) for _ in range(3)]

        random_user = random.choice(train_users)

        pos = random.choices([f.path for f in os.scandir(random_user)], k=2)
        neg = random.choice([f.path for f in os.scandir(random_user + "_forg")])

        print("Anchor = " + pos[0] + " Pos = " + pos[1] + " Neg = " + neg)

    end = time.time()
    print(end - start)
    return retTriplet