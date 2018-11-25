import tflearn
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error


class DnnModel:
    def __init__(self, active_func='sigmoid', hidden_nodes=3):
        self.learn_rate = 0.1
        self.momentum_coefficient = 0.8
        self.hidden_nodes = hidden_nodes
        tnorm = tflearn.initializations.uniform_scaling()
        input_layer = tflearn.input_data(shape=[None, 2])
        hidden_layer = tflearn.fully_connected(input_layer, hidden_nodes, activation=active_func, weights_init=tnorm)
        output_layer = tflearn.fully_connected(hidden_layer, 1, activation='linear', weights_init=tnorm)  # output layer of size 1
        momentum = tflearn.Momentum(
                    learning_rate=self.learn_rate, momentum=self.momentum_coefficient, name='Momentum')
        network = tflearn.regression(output_layer, optimizer=momentum, loss='mean_square')

        model = tflearn.DNN(network, tensorboard_verbose=0)
        self.model = model

    def train(self, train_xy, train_f, valid_xy, valid_f, n_epoch=5):
        self.model.fit(train_xy, train_f,  n_epoch=n_epoch, validation_set=(valid_xy, valid_f), show_metric=True, run_id="dnn_model", snapshot_step=1)
        train_prediction = np.array(self.model.predict(train_xy))
        train_mse = mean_squared_error(train_f, train_prediction.tolist())
        print('MSE train: %f' % train_mse)
        valid_prediction = np.array(self.model.predict(valid_xy)).tolist()
        valid_mse = mean_squared_error(valid_f, valid_prediction)
        print('MSE valid: %f' % valid_mse)
        tf.reset_default_graph()
        return train_mse, valid_mse
