import tensorflow as tf
import numpy as np


class TFModel:
    def __init__(self, active_func='sigmoid', hidden_nodes=3):
        self.XY = tf.placeholder(tf.float32, name='XY')
        self.F = tf.placeholder(tf.float32, name='F')
        self.active_func = active_func
        self.model = self.make_model(self.XY, [2, hidden_nodes, 1])

        self.cost = tf.reduce_mean(tf.square(self.model - self.F))
        self.train_op = tf.train.MomentumOptimizer(0.1, 0.8).minimize(self.cost)

    def make_model(self, x, layer_shape):
        atv_enum = {'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid}
        atv_fun = atv_enum[self.active_func]

        layer = tf.multiply(x, 1)
        for idx, size in enumerate(layer_shape[:-1]):
            w = tf.Variable(tf.random_normal([size, layer_shape[idx + 1]]))
            b = tf.Variable(tf.random_normal([layer_shape[idx + 1]]))
            layer = tf.add(tf.matmul(layer, w), b)
            if idx != len(layer_shape) - 2:
                layer = atv_fun(layer)

        return layer

    def train(self, train_xy, train_f, valid_xy, valid_f, n_epoch=500):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for step in range(n_epoch):
                train_error, _ = sess.run([self.cost, self.train_op], feed_dict={self.XY: train_xy, self.F: train_f})
                if step % 100 == 0:
                    print(step, train_error)
            train_error, _ = sess.run([self.cost, self.train_op], feed_dict={self.XY: train_xy, self.F: train_f})
            valid_error, _ = sess.run([self.cost, self.train_op], feed_dict={self.XY: valid_xy, self.F: valid_f})
            return train_error, valid_error
