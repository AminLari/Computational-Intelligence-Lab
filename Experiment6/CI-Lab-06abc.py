import numpy as np
import random


class Hopfield:
    def __init__(self, inp, num):
        self.input = inp
        self.num = num
        #self.size = len(self.input[0])
        self.size = 4
        self.weight = np.full((self.size, self.size), 0.0)

        self.output = np.full((self.num, self.size), 0.0)
        self.output_uni = np.full((self.num, self.size), 0.0)

    def uni_sgn(self, weight, v):
        a = np.dot(weight, v.T)
        for i in range(len(a)):
            if a[i] > 0:
                a[i] = 1
            else:
                a[i] = 0
        return a

    def sgn(self, weight, v):
        a = np.dot(weight, v.T)
        for i in range(len(a)):
            if a[i] > 0:
                a[i] = 1
            else:
                a[i] = -1
        return a

    def update(self):
        for z in range(self.num):
            for i in range(self.size):
                for j in range(self.size):
                    if i == j:
                        self.weight[i][j] += 0
                    else:
                        self.weight[i][j] += self.input[z][i]*self.input[z][j]

            self.output[z] = self.sgn(self.weight, self.input[z])
            self.output_uni[z] = self.uni_sgn(self.weight, self.input[z])


inp = np.array([[1, 1, 0, 0]])
inp2 = np.array([[-1, -1, 1, 1]])
inp3 = np.array([[-1, -1, 1, 1], [1, 1, -1, -1]])

s = Hopfield(inp, 1)
t = Hopfield(inp2, 1)
r = Hopfield(inp3, 2)

s.update()
print("Weight matrix:\n", s.weight)
print('-'*40, '\nInput:{0} | Output:{1}\n'.format(s.input, s.output_uni))
print('*'*40)

t.update()
print("Weight matrix:\n", t.weight)
print('-'*40, '\nInput:{0} | Output:{1}'.format(t.input, t.output))
print('*'*40)

r.update()
print("Weight matrix:\n", r.weight)
print('-'*40, '\nInput:{0} | Output:{1}'.format(r.input, r.output))


'''inp4 = np.random.uniform(-1, 1, 4)
w = np.array([[0, 1, 1, -1], [1, 0, 1, -1], [1, 1, 0, -1], [-1, -1, -1, 0]])

y_after = np.sign(np.dot(w, inp4))
y_before = inp4

while y_before.all() != y_after.all():
    y_after = np.sign(np.dot(w, y_before))
    y_before = y_after

print(y_after.T)'''
