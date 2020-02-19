import tensorflow as tf
import keras
import os
import random
import time
import numpy as np
from multiprocessing import Pool
from keras.preprocessing import image
from keras.layers import Layer, Input

input_shape = (256, 128, 3)
train_users = []
test_users = []

class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

        loss = tf.reduce_sum(tf.maximum(pos_dist-neg_dist+self.alpha, 0.0))

        return loss

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def create_model():
    # Set up input shapes
    inputs = [Input(input_shape) for _ in range(3)]


    # Import or create whatever model you want
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None,
                                                input_shape=input_shape, pooling=None, classes=128)

    # Normalization Layer
    output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(model.output)

    return keras.Model(model.input, output, outputs=loss)

def load_data():
    print("Mapping data...")
    global train_users, test_users
    train_users = [f.path for f in os.scandir("./data/train") if (not f.path[-1] == "g")]
    test_users = [f.path for f in os.scandir("./data/test") if (not f.path[-1] == "g")]
    print(train_users)
    print("Mapped all data")

def load_triplet(triplet):
    retTriplet = [image.img_to_array(image.load_img(triplet[i], target_size=(input_shape[0], input_shape[1]))) for i in range(3)]

    return retTriplet


def get_batch_random(batch_size, set="train", agents=None, chunksize=1):
    start = time.time()
    if (set == "train"):
        triplet_labels = []

        for _ in range(batch_size):
            random_user = random.choice(train_users)
            if (random.random() >= 0.5):
                pos = random.choices([f.path for f in os.scandir(random_user)], k=2)
                neg = random.choice([f.path for f in os.scandir(random_user + "_forg")])
            else:
                pos = random.choice([f.path for f in os.scandir(random_user + "_forg")], k=2)
                neg = random.choices([f.path for f in os.scandir(random_user)])

            triplet_labels.append([pos[0], pos[1], neg])

            print("Anchor = " + pos[0] + " Pos = " + pos[1] + " Neg = " + neg)

        with Pool(processes=agents) as pool:
            ret_triplet = pool.map(load_triplet, triplet_labels, chunksize=chunksize)

    end = time.time()
    print(end - start)
    return ret_triplet

def get_mixed_batch(sample_batch_size, hard_batch_size, normal_batch_size, model, s="train"):
    if (set == "train"):
        batch = get_batch_random(sample_batch_size)

        # Calculate embeddings
        anchor_embeddings = model.predict(batch[0])
        positive_embeddings = model.predict(batch[1])
        negative_embeddings = model.predict(batch[2])

        # Calculate how loss for each triplet
        sample_batch_losses = np.sum(np.square(anchor_embeddings - positive_embeddings), axis=1) - np.sum(np.square(anchor_embeddings - negative_embeddings), axis=1)

        sorted_batch = np.argsort(sample_batch_losses)[::-1]

        hard_triplets = sorted_batch[:hard_batch_size]
        random_triplets = random.choices(sorted_batch[hard_batch_size:], k=normal_batch_size)

        selection = np.append(hard_triplets, random_triplets)

        triplets = [batch[0][selection,:,:,:], batch[1][selection,:,:,:], batch[2][selection,:,:,:]]

        return triplets
