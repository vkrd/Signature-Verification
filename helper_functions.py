import tensorflow as tf

def triplet_loss(y_actual, y_pred, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[1])), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0], y_pred[2])), axis=-1)

    loss = tf.reduce_sum(tf.maximum(pos_dist-neg_dist+alpha, 0.0))

    return loss
