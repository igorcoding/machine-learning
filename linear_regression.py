import collections
import math
from util import LabeledDataSet


class LinearRegression:
    _training_dataset = None
    _max_iterations = None
    _alpha = None
    _theta = None

    def __init__(self, max_iterations=200, alpha=0.1):
        self._training_dataset = LabeledDataSet()
        self._max_iterations = max_iterations
        self._alpha = alpha

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        """
        Change max iterations count
        :param value: int
        """
        self._max_iterations = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Change a learning rate alpha
        :param value: float
        """
        self._alpha = value

    def _multiply_theta(self, x):
        n = len(self._theta)
        h = 0.0
        for i in xrange(0, n):
            h += self._theta[i] * x[i]
        return h

    def _compute_h(self, x):
        return self._multiply_theta(x)

    def train(self, training_dataset):
        self._training_dataset.load_dataset(training_dataset)
        self._theta = []
        for i in xrange(0, self._training_dataset.feature_size):
            self._theta.append(0.0)
        print "Begin training. Training set size: %d" % len(self._training_dataset)

        n = len(self._theta)
        m = len(self._training_dataset)
        for k in xrange(0, self.max_iterations):
            print "=== Iteration %d / %d ===" % (k + 1, self.max_iterations)

            error = 0.0
            for pair in self._training_dataset:
                h = self._compute_h(pair.x)
                error += math.pow(h - pair.y, 2)
            error /= 2 * m

            print "Current error = %f" % error

            error_derivs = []
            for j in xrange(0, n):
                e = 0.0
                for pair in self._training_dataset:
                    h = self._compute_h(pair.x)
                    e += (h - pair.y) * pair.x[j]
                e /= m
                error_derivs.append(e)

            for j in xrange(0, n):
                self._theta[j] -= self._alpha * error_derivs[j]

            print "Theta = ", self._theta

        print "Training finished"

    def predict(self, x):
        return self._compute_h(LabeledDataSet.make_tuple(x))


def main():
    lr = LinearRegression(max_iterations=10000, alpha=0.01)
    lr.train([((0, 0), 0),
              ((-1, 1), 1),
              ((1, 1), 1),
              ((2, 4), 4),
              ((-2, 4), 4),
              ((3, 9), 9),
              ((-3, 9), 9),
              ((4, 16), 16),
              ((-4, 16), 16)])

    test = (15.0, 225.0)
    print "Predict for ", test, " is %f" % (lr.predict(test),)

if __name__ == "__main__":
    main()