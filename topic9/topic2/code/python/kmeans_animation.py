######################################################################
# The codes are based on Python3.
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
######################################################################
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, find
from itertools import combinations
import copy
# k-means clustering;

# change the value to 0 to run just kmeans without comparing to brute force
# method;
iscomparebruteforce = 0

#####################################################################
# k-means algorithm;
# Greedy algorithm trying to minimize the objective function;
#
dim = 2  # dimension of the data points
# change to larger number of data points and cno = 3 after comparison with brute force
# method
if iscomparebruteforce == 1:
    m = 6  # number of data points
    # fix the seed of the random number generator, so each time generate
    # the same random points;
    np.random.seed(1)
    x = np.concatenate((np.random.randn(dim, m) + np.tile(np.array([4, 1]), (m, 1)).T,
                        np.random.randn(dim, m) + np.tile(np.array([4, 4]), (m, 1)).T), axis=1)
    x = np.concatenate((x, np.random.randn(dim, m) +
                        np.tile(np.array([1, 2]), (m, 1)).T), axis=1)

    # number of clusters
    cno = 2
else:
    m = 100  # number of data points
    # fix the seed of the random number generator, so each time generate
    # the same random points;
    np.random.seed(1)
    x = np.concatenate((np.random.randn(dim, m) + np.tile(np.array([4, 1]), (m, 1)).T,
                        np.random.randn(dim, m) + np.tile(np.array([4, 4]), (m, 1)).T), axis=1)
    x = np.concatenate((x, np.random.randn(dim, m) +
                        np.tile(np.array([1, 2]), (m, 1)).T), axis=1)

    x = np.concatenate((x, np.tile(
        np.array([-7, -7]), (10, 1)).T + 0.1 * np.random.randn(dim, 10)), axis=1)
    # number of clusters
    cno = 6
m = x.shape[1]

##
# randomly initialize the cluster center; since the seed for function rand
# is not fixed, every time it is a different matrix
np.random.seed(int(time.time()))
c = 6 * np.random.rand(dim, cno)
c_old = copy.deepcopy(c) + 10

plt.ion()
plt.figure()
i = 1
# check whether the cluster centers still change
tic = time.time()
while np.linalg.norm(c - c_old, ord='fro') > 1e-6:
    print("--iteration %d \n" % i)

    # record previous c;
    c_old = copy.deepcopy(c)

    # Assign data points to current cluster;
    # Squared norm of cluster centers.
    cnorm2 = np.sum(np.power(c, 2), axis=0)
    tmpdiff = 2 * np.dot(x.T, c) - cnorm2
    labels = np.argmax(tmpdiff, axis=1)

    # Update data assignment matrix;
    # The assignment matrix is a sparse matrix,
    # with size m x cno. Only one 1 per row.
    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))

    # adjust the cluster centers according to current assignment;
    cstr = ['r.', 'b.', 'g.', 'r+', 'b+', 'g+']
    obj = 0
    for k in range(0, cno):
        idx = find(P[:, k])[0]
        nopoints = idx.shape[0]
        if nopoints == 0:
            # a center has never been assigned a data point;
            # re-initialize the center;
            c[:, k] = np.random.rand(dim, 1)[:, 0]
        else:
            # equivalent to sum(x(:,idx), 2) ./ nopoints;
            c[:, k] = ((P[:, k].T.dot(x.T)).T / float(nopoints))[:, 0]
            plt.plot(x[0, idx], x[1, idx], cstr[k])
            # plt.hold(True)
        obj = obj + \
            np.sum(
                np.sum(np.power(x[:, idx] - np.tile(c[:, k], (nopoints, 1)).T, 2)))

    plt.plot(c[0, :], c[1, :], 'yo')
    # plt.hold(False)
    plt.draw()
    plt.pause(1)

    i = i + 1

toc = time.time()
# k-means will be much faster than brute force enumeration, even after we
# have the additional pause 1 second and visualization in the codes;

# run it several times and you will see that objective function is
# different;
print('Elapsed time is %f seconds \n' % float(toc - tic))
print('obj =', obj)

#####################################################################
# enumerating all possibilities is computational intensive;
# we can only work with a small number data points;
#
if iscomparebruteforce == 1:

    # enumerate all possibility;
    best_obj = 1e7
    #result = []
    tic = time.time()
    for i in range(1, m):
        print('-- case %d \n' % i)
        partition1 = combinations(np.arange(0, m, 1), i)
        for j in partition1:
            obj = 0
            group1_idx = np.array(j)
            center1 = np.sum(x[:, group1_idx], axis=1) / \
                float(group1_idx.shape[0])
            obj = obj + \
                np.sum(np.sum(
                    np.power(x[:, group1_idx] - np.tile(center1, (group1_idx.shape[0], 1)).T, 2)))
            # equivalent:
            # center1.shape=(2,1)
            #center1 = np.transpose(center1)
            # for l in range(0, group1_idx.shape[0]):
            #    obj = obj + np.sum(  np.power( x[:,group1_idx[l]] - center1 , 2), axis=1  )

            group2_idx = np.setdiff1d(np.arange(0, m, 1), np.array(j))
            center2 = np.sum(x[:, group2_idx], axis=1) / \
                float(group2_idx.shape[0])
            center2.shape = (2, 1)
            center2 = np.transpose(center2)
            for l in range(0, group2_idx.shape[0]):
                obj = obj + \
                    np.sum(np.power(x[:, group2_idx[l]] - center2, 2), axis=1)

            if obj < best_obj:
                result = group1_idx
                best_obj = obj

    toc = time.time()
    # look at the objective function; the objective function of brute force
    # enumerate is smaller! kmeans only find a local minimum in this case;
    print('Elapsed time is %f seconds \n' % float(toc - tic))
    print('best_obj =', best_obj)

##
ra = np.random.randn(m, 1)
idx1 = np.where(ra > 0)[0]
idx2 = np.setdiff1d(np.arange(0, m, 1), idx1)

center1 = np.mean(x[:, idx1], axis=1)
center2 = np.mean(x[:, idx2], axis=1)

newobj = np.sum(np.sum(np.power(x[:, idx1] - np.tile(center1, (idx1.shape[0], 1)).T, 2)))  \
    + np.sum(np.sum(np.power(x[:, idx2] -
                             np.tile(center2, (idx2.shape[0], 1)).T, 2)))

print('newobj =', newobj)
plt.show(block=True)
