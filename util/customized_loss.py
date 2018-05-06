from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import array_ops
import random

cost = 1.75


def bias_binary_crossentropy(y_true, y_pred):
    raw_loss = K.binary_crossentropy(y_true, y_pred)
    cond = tf.logical_and(tf.equal(y_true, 0),
                          tf.greater_equal(y_pred, 0.5))
    tuned_loss = cost * raw_loss
    bias_raw_loss = array_ops.where(cond, tuned_loss, raw_loss)
    return K.mean(bias_raw_loss, axis=-1)


def bias_mean_square_error(y_true, y_pred):
    raw_loss = K.square(y_pred - y_true)
    cond = tf.logical_and(tf.equal(y_true, 0),
                          tf.greater_equal(y_pred, 0.5))
    tuned_loss = cost * raw_loss
    bias_raw_loss = array_ops.where(cond, tuned_loss, raw_loss)
    return K.mean(bias_raw_loss, axis=-1)


def bias_mean_abs_error(y_true, y_pred):
    raw_loss = K.abs(y_pred - y_true)
    cond = tf.logical_and(tf.equal(y_true, 0),
                          tf.greater_equal(y_pred, 0.5))
    tuned_loss = cost * raw_loss
    bias_raw_loss = array_ops.where(cond, tuned_loss, raw_loss)
    return K.mean(bias_raw_loss, axis=-1)


def binary_crossentropy(y_true, y_pred):
    # origin implementation of binary_crossentropy in keras
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == "__main__":
    x = tf.placeholder(tf.float16)
    y = tf.placeholder(tf.float16)

    session = tf.Session()

    x_value = [1, 1, 0, 0]
    y_value = [0.1, 0.9, 0.9, 0.01]
    cond = tf.logical_and(tf.equal(tf.convert_to_tensor(x_value), 0),
                          tf.greater_equal(tf.convert_to_tensor(y_value), 0.5))

    tmp0 = session.run(cond,
                       feed_dict={x: x_value, y: y_value})
    print(tmp0)

    tmp1 = session.run(mean_squared_error(x, y),
                       feed_dict={x: x_value, y: y_value})
    print(tmp1)
    tmp2 = session.run(bias_mean_square_error(x, y),
                       feed_dict={x: x_value, y: y_value})
    print(tmp2)
