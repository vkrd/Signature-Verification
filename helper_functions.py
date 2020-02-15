import tensorflow as tf
import keras

def triplet_loss(y_actual, y_pred, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[1])), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[2])), axis=-1)

    loss = tf.reduce_sum(tf.maximum(pos_dist-neg_dist+alpha, 0.0))

    return loss

def create_model(input_shape):
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None,
                                                input_shape=input_shape, pooling=None, classes=128)

    # Normalization Layer
    output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(model.output)

    return keras.Model(model.input, output)
