import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Read the DataFrame, first using the feature data
df1 = pd.read_csv('Dataset1.csv')
df2 = pd.read_csv('Dataset2.csv')


class Kmean:
    # function to calculate distance between two points
    def distance(self, x, y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    # initialize class parameters
    def __init__(self, cluster, dataset):
        self.cluster = cluster
        self.dataset = np.array(dataset.values)
        self.error = np.full((self.cluster, 1), 0, dtype=float)
        self.avg = 0.0

        self.mean = np.full((self.cluster, 2), 0, dtype=float)

        center = random.sample(range(0, len(self.dataset)), self.cluster)
        self.centers = np.full((self.cluster, 2), 0, dtype=float)

        for i in range(self.cluster):
            self.centers[i] = self.dataset[center[i]]

        self.dist = np.full((len(dataset), cluster), 0, dtype=float)

    # update center of clusters and assign classify data
    def update(self):
        for k in range(5):
        #while 1:
            for i in range(len(self.dataset)):
                for j in range(self.cluster):
                    self.dist[i][j] = self.distance(self.dataset[i], self.centers[j])

            self.output = np.argmin(self.dist, axis=1)

            for i in range(len(self.dataset)):
                cnt = 0.2
                for j in range(self.cluster):
                    if self.output[i] == j:
                        cnt += 1.0
                        self.mean[j][0] += self.dataset[i][0]
                        self.mean[j][1] += self.dataset[i][1]
                    else:
                        continue

                self.mean = np.divide(self.mean, cnt)

            for a in range(self.cluster):
                self.error[a] = self.distance(self.mean[a], self.centers[a])

            self.centers = self.mean

            if all(x < 1.74 for x in self.error):
                print('stopping train process...')
                break
            else:
                continue

        # plot center of clusters
        '''for c in range(self.cluster):
            plt.plot(self.centers[c][0], self.centers[c][1], 'm.')'''

        # plot cluster data with distinct colours
        for a in range(len(self.dataset)):
                if self.output[a] == 0:
                    plt.plot(self.dataset[a][0], self.dataset[a][1], 'r.')
                if self.output[a] == 1:
                    plt.plot(self.dataset[a][0], self.dataset[a][1], 'y.')
                if self.output[a] == 2:
                    plt.plot(self.dataset[a][0], self.dataset[a][1], 'g.')
                if self.output[a] == 3:
                    plt.plot(self.dataset[a][0], self.dataset[a][1], 'b.')

        self.avg = np.average(self.error)


# construct object of class K-mean
t = Kmean(4, df1)
t.update()

# draw output of network
plt.title('Output of K-mean Algorithm Network')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Draw elbow diagram
'''arr_x = np.arange(1, 17)
arr_y = np.full((16, 1), 0, dtype=float)
for i in range(15):
    t = Kmean(i+1, df1)
    t.update()
    arr_y[i+1] = t.avg

plt.plot(arr_x, arr_y)
plt.title('Elbow Diagram')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Error')
plt.show()'''

