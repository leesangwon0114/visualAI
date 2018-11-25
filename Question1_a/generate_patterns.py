import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GeneratePattern:
    def __init__(self, num=1000):
        self.data_num = num
        self.space_min_value = -1
        self.space_max_value = 1
        self.mu = 0
        self.sigma = 1
        self.x_data = []
        self.y_data = []
        self.f_data = []
        self.data = None
        self.filename = 'data.npy'
        self.func = lambda x, y: (math.sin(20*math.sqrt(pow(x, 2) + pow(y, 2))) / (20 * math.sqrt(pow(x, 2) + pow(y, 2)))) \
                                 + (math.cos(10*math.sqrt(pow(x, 2) + pow(y, 2))) / 5) \
                                 + (y/2) \
                                 - 0.3

    def generate_data(self):
        self.x_data = [random.uniform(self.space_min_value, self.space_max_value) for _ in range(self.data_num)]
        self.y_data = [random.uniform(self.space_min_value, self.space_max_value) for _ in range(self.data_num)]
        self.f_data = list(map(self.func, self.x_data, self.y_data))
        gaussian_noise = np.random.normal(self.mu, self.sigma, self.data_num)
        self.f_data = list(self.f_data + (0.1 * gaussian_noise))
        self._save_data()

    def _save_data(self):
        x_data = np.reshape(self.x_data, (self.data_num, 1))
        y_data = np.reshape(self.y_data, (self.data_num, 1))
        f_data = np.reshape(self.f_data, (self.data_num, 1))
        features = np.append(x_data, y_data, axis=1)
        self.data = np.append(features, f_data, axis=1)
        np.random.shuffle(self.data)
        np.save(self.filename, self.data)

    def load_data(self):
        if os.path.exists(self.filename):
            self.data = np.load(self.filename)
        else:
            self.generate_data()
        training, validation, test = self.data[:int(self.data_num*0.70), :], \
                                     self.data[int(self.data_num*0.70):int(self.data_num*0.70 + self.data_num*0.15), :], \
                                     self.data[int(self.data_num*0.70 + self.data_num*0.15):, :]
        return training, validation, test

    def plot_data(self):
        if self.data is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for x_val, y_val, f_val in self.data:
                ax.scatter(x_val, y_val, f_val, color="b")
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('F Label')
            plt.show()


if __name__ == "__main__":
    gp = GeneratePattern()
    train, valid, test = gp.load_data()
    gp.plot_data()


