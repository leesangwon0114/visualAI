import matplotlib.pyplot as plt
from Question1_a.generate_patterns import GeneratePattern
from Question1_a.dnn_model import DnnModel
from Question1_a.tf_model import TFModel
from sklearn.neural_network import MLPRegressor


if __name__ == "__main__":
    gp = GeneratePattern(10000)
    train, valid, test = gp.load_data()
    #gp.plot_data()

    train_xy = train[:, 0:2]
    train_f = train[:, 2:3]
    valid_xy = valid[:, 0:2]
    valid_f = valid[:, 2:3]
    test_xy = test[:, 0:2]
    test_f = test[:, 2:3]

    hidden_nodes = []
    trains_mse = []
    valids_mse = []

    for hidden_nodes_num in range(3, 20):
        model1 = DnnModel('tanh', hidden_nodes_num)
        #model1 = TFModel('relu', hidden_nodes_num)
        train_mse, valid_mse = model1.train(train_xy, train_f, valid_xy, valid_f)
        hidden_nodes.append(hidden_nodes_num)
        trains_mse.append(train_mse)
        valids_mse.append(valid_mse)

    y = [tuple(trains_mse), tuple(valids_mse)]
    x = hidden_nodes
    fig = plt.figure(figsize=(8, 6))

    plt.plot(x, tuple(trains_mse), label='train_error', lw=2, marker='o')
    plt.plot(x, tuple(valids_mse), label='valid_error', lw=2, marker='s')
    plt.xlabel('hidden_nodes')
    plt.ylabel('mse error')
    plt.grid()
    plt.legend(loc='upper right')
    plt.gcf().autofmt_xdate()
    plt.show()
