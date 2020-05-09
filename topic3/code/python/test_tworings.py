import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# generate the data
# generate random angles between [0, 2pi]
n = 800
rangle = 2 * np.pi * np.random.rand(n, 1)

# generate random radius for the first circle
e = 0.2
rr = 1.9 + e * np.random.rand(n, 1)

rx = rr * np.sin(rangle)
ry = rr * np.cos(rangle)

x = rx
y = ry

# generate random radius for the second circle
rr2 = 1.2 + e * np.random.rand(n, 1)

rx2 = rr2 * np.sin(rangle)
ry2 = rr2 * np.cos(rangle)

x = np.concatenate((x, rx2))
y = np.concatenate((y, ry2))

rx3 = 1.4 + (1.9 - 1.4) * np.random.rand(10, 1)
ry3 = e * np.random.rand(10, 1)

# uncomment this to comment the two rings;
x = np.concatenate((x, rx3))
y = np.concatenate((y, ry3))

data = np.concatenate((x, y), axis=1)

plt.scatter(data[:, 0], data[:, 1], c='black')
plt.title('original data')
plt.show()

# run kmeans on the original coordinates
K = 2
kmeans = KMeans(n_clusters=K).fit(data)
idx = kmeans.labels_

plt.scatter([x for i, x in enumerate(data[:, 0]) if idx[i] == 0],
            [y for i, y in enumerate(data[:, 1]) if idx[i] == 0],
            c='r')

plt.scatter([x for i, x in enumerate(data[:, 0]) if idx[i] == 1],
            [y for i, y in enumerate(data[:, 1]) if idx[i] == 1],
            c='b')

plt.title('K-means')
plt.show()

distmat = pairwise_distances(data) * pairwise_distances(data)

A = (distmat < 0.1).astype(np.int)

plt.spy(A)
plt.title('Adjacency Matrix')
plt.show()

D = np.diag(np.sum(A, axis=1))
L = D - A

s, v = np.linalg.eig(L)
K = 2
v = v[:, 0:K].real
kmeans = KMeans(n_clusters=K).fit(v)
idx = kmeans.labels_

plt.scatter([x for i, x in enumerate(data[:, 0]) if idx[i] == 0],
            [y for i, y in enumerate(data[:, 1]) if idx[i] == 0],
            c='r')

plt.scatter([x for i, x in enumerate(data[:, 0]) if idx[i] == 1],
            [y for i, y in enumerate(data[:, 1]) if idx[i] == 1],
            c='b')

plt.title('Spectral Clustering')
plt.show()
