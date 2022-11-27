import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

def draw():
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)

    X, Y = np.meshgrid(x, y)
    Z1 = []
    Z2 = []
    print("1")
    for i in range(len(X)):
        z = []
        for j in range(len(X[i])):
            point = torch.tensor([float(X[i][j]), float(Y[i][j])])
            z.append(nnet(point).detach().numpy()[0])
        Z1.append(z)
    print("2")
    for i in range(len(X)):
        z = []
        for j in range(len(X[i])):
            point = torch.tensor([float(X[i][j]), float(Y[i][j])])
            z.append(nnet(point).detach().numpy()[1])
        Z2.append(z)
    """"
    fig = plt.figure("3d")
    ax = plt.axes(projection='3d')
    print("1")
    for i in range(len(X)):
        for j in range(len(X[i])):
            ax.plot_wireframe(X, Y, np.array(Z1), color='red', rstride=1, cstride=1)
    print("2")
    for i in range(len(X)):
        for j in range(len(X[i])):
            ax.plot_wireframe(X, Y, np.array(Z2), color='blue', rstride=1, cstride=1)
    """
    plt.figure("2d")
    x = np.linspace(-10., 10., 80)
    y = np.linspace(-10., 10., 80)
    for i in range(len(x)):
        for j in range(len(y)):
            data = [x[i], y[j]]
            print(nnet(data))
            if nnet(data)[0] > nnet(data)[1]:
                plt.plot(x[i], y[j], marker=".", markersize=2, markeredgecolor="red", markerfacecolor="red")
            else:
                plt.plot(x[i], y[j], marker=".", markersize=2, markeredgecolor="blue", markerfacecolor="blue")
    for i in range(len(point_xy)):
        data = [point_xy[i][0], point_xy[i][1]]
        print(nnet(data))
        if point_z[i][0] > point_z[i][1]:
            plt.plot(point_xy[i][0], point_xy[i][1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        else:
            plt.plot(point_xy[i][0], point_xy[i][1], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
    plt.show()


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_features=2, out_features=70)
        self.l2 = nn.Linear(in_features=70, out_features=70)
        self.l3 = nn.Linear(in_features=70, out_features=2)

    def forward(self, input_x):
        input_x = np.array(input_x).reshape(2)
        input_x = torch.tensor(input_x, dtype=torch.float32)
        input_x = self.relu(self.l1(input_x))
        input_x = self.relu(self.l2(input_x))
        input_x = self.l3(input_x)

        return input_x

nnet = NN()

input = np.array([1, 2])

print(nnet(input))

point_xy = [[0., 0.], [1., 1.], [0., 2.], [-3., -6.], [-4., -6.], [5., -2.], [9., -5.], [-7., 7.], [2, 2], [2., -2.], [2., 0.], [1., 0.], [2., 1.], [1., 2.], [0., 1.], [1., 6.], [5., 5.], [8., 2.], [1., -5], [2., -5], [3., -5], [1., -9], [2., -9], [3., -9], [0., -6.], [0., -7.], [0., -8.], [4., -6.], [4., -7.], [4., -8.], [1., -6.], [2., -6.], [3., -6.], [1., -8.], [2., -8.], [3., -8.], [1., -7.], [2., -7.], [3., -7.], [-3., 7.]]
point_z = [[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [1., 0.]]

# draw()

e = 1000
op = torch.optim.SGD(params=nnet.parameters(), lr=0.01)
for epoch in range(e):
    for i in range(len(point_xy)):
        loss = F.cross_entropy(nnet(point_xy[i]), torch.tensor(point_z[i]))
        print(str(epoch) + ": " + str(loss))
        if loss > 0.1:
            loss.backward()
            op.step()
            op.zero_grad()


draw()

