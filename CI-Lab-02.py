import random
import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler

# Using load breast cancer dataset to train and test the learning algorithm
data = load_breast_cancer()

# test size
sample = 120

# Read the DataFrame, first using the feature data
df = pd.read_csv('breast_cancer.csv')

# Concatenating input data and column vector filled with 1 elements
arr1= data.data
arr2= np.full((569, 1), 1, dtype=int)
input = np.hstack((arr1, arr2))

# extracting output data from dataset
output = data.target

# Feature Scaling
sc = StandardScaler()
input = sc.fit_transform(input)


# test data in primary version
'''input = [[2.7810836, 2.550537003, 1],
        [1.465489372, 2.362125076, 1],
        [3.396561688, 4.400293529, 1],
        [1.38807019, 1.850220317, 1],
        [3.06407232, 3.005305973, 1],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1]]

output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]'''

weight = [0 for i in range(31)]
threshold = 0

# initializing weights and threshold
def initial():
    for i in range(31):
        x = random.randint(0, 1000)
        weight[i] = x/1000.0

    threshold = (random.randint(0, 1000))/1000.0


# calculating predicted output of algorithm
def predict(z, inp):
    temp = 0
    for i in range(31):
        temp += weight[i] * inp[z][i]

    if temp < threshold:
        p = 0
    else:
        p = 1

    return p


# updating weights to achieve a more accurate output using first #sample items
def learn(inp, out, learning_rate):
    for z in range(sample):
        if predict(z, inp) != out[z]:
            for j in range(100):
                for i in range(31):
                    weight[i] = weight[i] + learning_rate*((out[z]-predict(z, inp))*inp[z][i])
        else:
            continue


# testing algorithm on last 20 items
def test(num):
    counter = num*20
    for z in range(num):
        print('-' * 45)
        initial()
        learn(input, output, 0.1)
        print('test #', z + 1, ':\nTuned weights:\n', weight, '\n')
        for i in range(549, 569):
            print('desired output: ', output[i], ' | predicted output: ', predict(i, input))
            if output[i] != predict(i, input):
                counter -= 1

    accuracy = 100.0*(counter/(num*20.0))
    print('-'*45)
    print('Learning accuracy: ', accuracy, '%')


# calling test function to implement developed algorithm on breast cancer dataset
test(10)


