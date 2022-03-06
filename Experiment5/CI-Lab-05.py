import numpy as np
import random
import math
import matplotlib.pyplot as plt


class RBF:

    def __init__(self, n, inp, out):
        self.num = n
        self.input = inp
        self.expected = out
        self.u = np.full(self.num, 0.0001)
        for i in range(self.num):
            a = random.uniform(0.0001, 1.0)
            self.u[i] = a
        self.weight = np.random.uniform(0.0001, 1, self.num)
        self.net = np.full(self.num, 0.0001)
        self.output = np.full(len(self.expected), 0.0001)
        self.lr = 0.01

        self.dist = np.full((self.num, self.num), 0, dtype=float)
        for i in range(self.num):
            for j in range(self.num):
                self.dist[i][j] = self.distance(self.u[i], self.u[j])

        self.s = np.full(self.num, np.divide(self.dist.max(1), math.sqrt(2)*self.num), dtype=float)

        self.bias = random.uniform(0, 1)

    def gaussian(self, x, u, s):
        return (1/math.sqrt(2*math.pi*s*s))*math.exp(-(x-u)**2/(2*s*s))

    def distance(self, x, y):
        return math.sqrt((x-y)**2)

    def update(self):
        for j in range(200):
            for k in range(len(self.input)):

                for i in range(self.num):
                    self.net[i] = self.gaussian(self.input[k], self.u[i], self.s[i])

                self.output[k] = self.net.T.dot(self.weight) + self.bias
                self.loss = (self.output[k] - self.expected[k]) ** 2
                self.error = -(self.output[k] - self.expected[k])

                for i in range(self.num):
                    self.weight[i] += self.lr * self.net[i] * self.error
                self.bias += self.lr * self.error


samples = 100
x = np.random.uniform(0, 1, samples)
x = np.sort(x)
noise = np.random.uniform(-0.1, 0.1, samples)
y = [0 for i in range(samples)]
z = [0 for i in range(samples)]


for i in range(samples):
    y[i] = math.sin(2 * math.pi * x[i]) + noise[i]
    z[i] = math.sin(2 * math.pi * x[i])

t = RBF(3, x, y)
t.update()
for i in range(samples):
    print('desired output:', y[i], ' | predicted output:', t.output[i])

line0 = plt.plot(x, z, 'm')
line1 = plt.plot(x, y, 'y')
line2 = plt.plot(x, t.output, 'r')
plt.title('Comparison between real and predicted outputs')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Real Output', 'Noisy Output', 'Network Output'])
plt.show()




