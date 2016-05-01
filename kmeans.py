import random
import collections
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pylab


def read_file(filename):
    x = []
    with open(filename) as f:
        lines = f.readlines()
        for l in lines:
            nums = l.split()
            x.append([float(n) for n in nums])
    return np.asarray(x)


def plot(x):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.scatter(x[:, 0], x[:, 1], s=10, c='b', marker="s", label='1')
    plt.show()


class KMeans(object):
    def __init__(self, clusters, runs=1):
        super(KMeans, self).__init__()
        self.clusters = clusters
        self.runs = runs

    @staticmethod
    def _preprocess(X):
        new_X = []
        for x in X:
            new_X.append(x if isinstance(x, collections.Iterable) else [x])
        return new_X

    def _dist(self, x, c):
        return np.linalg.norm(x - c)

    def _closest_centroid(self, x, centroids):
        closest_c = None
        for ci, c in enumerate(centroids):
            d = self._dist(x, c)
            if closest_c is None or d < closest_c[1]:
                closest_c = (ci, d)
        return closest_c[0]

    def _run(self, X):
        X = self._preprocess(X)
        centroids = [None] * self.clusters
        for i in xrange(0, len(centroids)):
            rand_index = random.randint(0, len(X)-1)
            centroids[i] = X[rand_index]

        clusters = None
        clusters_x_ids = None

        for _ in xrange(0, 100):
            clusters = [[] for _ in xrange(0, self.clusters)]
            clusters_x_ids = [[] for _ in xrange(0, self.clusters)]
            for xi, x in enumerate(X):
                ci = self._closest_centroid(x, centroids)
                clusters[ci].append(x)
                clusters_x_ids[ci].append(xi)

            for ci, c in enumerate(clusters):
                centroids[ci] = np.average(c, axis=0)

        J = 0.0
        X_labels = [None] * len(X)
        for ci, c in enumerate(clusters_x_ids):
            for xi in c:
                J += np.linalg.norm(X[xi] - centroids[ci]) ** 2
                X_labels[xi] = ci

        J /= len(X)
        return X_labels, J, np.asarray(centroids)

    def run(self, X):
        print 'K = %d' % self.clusters
        results = None
        for r in xrange(0, self.runs):
            print 'Run %d/%d' % (r+1, self.runs)
            labels, J, centroids = self._run(X)
            if results is None or J < results[1]:
                results = (labels, J, centroids)

        return results


def analyse_errors(X, runs=2):
    res = []
    for k in xrange(1, 11):
        kmeans = KMeans(k, runs=runs)
        labels, J, centroids = kmeans.run(X)
        res.append([labels, J])

    res = np.asarray(res)
    pyplot.plot(xrange(1, 11), res[:, 1])
    pyplot.show()


def clusterize(X, clusters=2, runs=2):
    kmeans = KMeans(clusters, runs=runs)
    labels, J, centroids = kmeans.run(X)
    print 'Error = %.4f' % J
    pyplot.scatter(X[:, 0], X[:, 1], c=labels)
    pyplot.scatter(centroids[:, 0], centroids[:, 1], c='green', s=90)
    # for c in centroids:
    #     pyplot.sc
    pyplot.show()


def main():
    X = read_file('datasets/kmeans.txt')
    # plot(X)

    clusters = 2
    runs = 2
    # analyse_errors(X, runs)
    clusterize(X, clusters, runs)



if __name__ == "__main__":
    main()