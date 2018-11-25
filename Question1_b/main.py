import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


class MNISTModel:
    def __init__(self, learning_rate=0.01, momentum=0.8):
        self.X, self.Y, self.test_x, self.test_y = mnist.load_data(one_hot=True)

        self.X = self.X.reshape([-1, 28, 28, 1])
        self.test_x = self.test_x.reshape([-1, 28, 28, 1])

        conv_net = input_data(shape=[None, 28, 28, 1], name='input')
        conv_net = conv_2d(conv_net, 20, 5, strides=1, padding='valid', activation='relu')
        conv_net = max_pool_2d(conv_net, 2, strides=2, padding='valid')

        conv_net = conv_2d(conv_net, 50, 5, strides=1, padding='valid', activation='relu')
        conv_net = max_pool_2d(conv_net, 2, strides=2, padding='valid')

        conv_net = fully_connected(conv_net, 500, activation='relu')
        #convnet = dropout(convnet, 0.8)

        conv_net = fully_connected(conv_net, 10, activation='softmax')
        momentum = tflearn.optimizers.Momentum(learning_rate=learning_rate, momentum=momentum)
        self.network = regression(conv_net, optimizer=momentum, loss='categorical_crossentropy', name='targets')
        self.model = None

    def train(self, n_epoch=10):
        self.model = tflearn.DNN(self.network, tensorboard_verbose=3)
        self.model.fit(self.X, self.Y, n_epoch=n_epoch, validation_set=(self.test_x, self.test_y), show_metric=True, run_id='convnet_mnist', snapshot_step=5)


if __name__ == "__main__":
    learning_rate = 0.1
    momentum = 0.8
    model = MNISTModel(learning_rate, momentum)
    epochs = 1
    model.train(epochs)
