import tensorflow as tf
import keras
import os
import random
import time
from keras.optimizers import Adam
import numpy as np
from multiprocessing import Pool
from keras.preprocessing import image
from keras.layers import *
from keras import Model, Sequential

input_shape = (650, 275, 1)
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

def create_model(network):

    # Set up input shapes
    anchor_shape = Input(input_shape)
    positive_shape = Input(input_shape)
    negative_shape = Input(input_shape)

    anchor_embedding = network(anchor_shape)
    positive_embedding = network(positive_shape)
    negative_embedding = network(negative_shape)

    loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([anchor_embedding, positive_embedding, negative_embedding])

    return Model(inputs=[anchor_shape, positive_shape, negative_shape], outputs=loss_layer)


def create_network():
    # Import or create whatever model you want
    # model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None,
    #                                            input_shape=input_shape, pooling=None, classes=128)


    # Model below from keras documentation
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    network.add(Conv2D(32, (3, 3), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.25))

    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.25))

    network.add(Flatten())
    network.add(Dense(256, activation='relu'))
    network.add(Dropout(0.5))
    network.add(Dense(128, activation='softmax'))

    # Normalization Layer (need this for all models)
    network.add(Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1)))

    # Use this for non-sequential models
    # output = Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(model.output)

    return network

def load_data():
    print("Mapping data...")
    global train_users, test_users
    # Locate all train and test images
    train_users = [f.path for f in os.scandir("./data/train") if (f.is_dir() and not f.path[-1] == "g")]
    test_users = [f.path for f in os.scandir("./data/test") if (f.is_dir() and not f.path[-1] == "g")]
    print("Mapped all data")

def load_triplet(triplet):
    retTriplet = [image.img_to_array(image.load_img(triplet[i], color_mode="grayscale",
                                                    target_size=(input_shape[0], input_shape[1]))) for i in range(3)]

    return retTriplet


def get_batch_random(batch_size, dataset="train", agents=None, chunksize=1):
    triplet_labels = []

    for _ in range(batch_size):
        if dataset == "train":
            random_user = random.choice(train_users)
        else:
            random_user = random.choice(test_users)

        if random.random() >= 0.5:
            pos = random.choices([f.path for f in os.scandir(random_user)], k=2)
            neg = random.choice([f.path for f in os.scandir(random_user + "_forg")])
        else:
            pos = random.choices([f.path for f in os.scandir(random_user + "_forg")], k=2)
            neg = random.choice([f.path for f in os.scandir(random_user)])

        triplet_labels.append([pos[0], pos[1], neg])

    with Pool(processes=agents) as pool:
        ret_triplet = pool.map(load_triplet, triplet_labels, chunksize=chunksize)
    return ret_triplet

def get_mixed_batch(sample_batch_size, hard_batch_size, normal_batch_size, model, s="train"):
    batch = np.asarray(get_batch_random(sample_batch_size, dataset=s))

    # Calculate embeddings
    anchor_embeddings = model.predict(batch[:,0,:,:], batch_size=sample_batch_size)
    positive_embeddings = model.predict(batch[:,1,:,:], batch_size=sample_batch_size)
    negative_embeddings = model.predict(batch[:,2,:,:], batch_size=sample_batch_size)

    # Calculate loss for each triplet
    sample_batch_losses = np.sum(np.square(anchor_embeddings - positive_embeddings), axis=1) - np.sum(np.square(anchor_embeddings - negative_embeddings), axis=1)

    sorted_batch = np.argsort(sample_batch_losses)[::-1]

    # Pull hardest triplets
    hard_triplets = sorted_batch[:hard_batch_size]

    # Pull random triplets
    random_triplets = random.choices(sorted_batch[hard_batch_size:], k=normal_batch_size)

    selection = np.append(hard_triplets, random_triplets)

    triplets = [batch[selection,0,:,:], batch[selection,1,:,:], batch[selection,2,:,:]]

    return triplets

# Function to calculate ideal input shape
def calculate_average_shape():
    sumWidth = 0.0
    sumHeight = 0.0
    count = 0.0
    for subdir, dirs, files in os.walk("./data"):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            img = image.img_to_array(image.load_img(filepath, color_mode="grayscale"))
            sumWidth += img.shape[1]
            sumHeight += img.shape[0]
            count += 1
    return [sumWidth/count, sumHeight/count]
