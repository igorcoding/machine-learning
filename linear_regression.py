import collections
import math


class LabeledPair:
    _x = None
    _y = None

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class LabeledRegressionDataSet:
    _dataset = None
    _feature_size = None

    def __init__(self, dataset=None):
        if dataset is not None:
            self.load_dataset(dataset)

    @staticmethod
    def make_tuple(x):
        if isinstance(x, collections.Iterable):
            return (1, ) + tuple(x)
        else:
            return 1, x

    def load_dataset(self, dataset):
        if isinstance(dataset, list) or isinstance(dataset, tuple):
            self._dataset = []
            for x, y in dataset:
                x = self.make_tuple(x)
                feature_size = len(x)
                if self._feature_size is None:
                    self._feature_size = feature_size
                elif feature_size != self._feature_size:
                    raise AttributeError("All features' sizes have to be equal")
                self._dataset.append(LabeledPair(x, y))
        else:
            raise AttributeError("Dataset should be either list or tuple")

    @property
    def feature_size(self):
        return self._feature_size

    def __getitem__(self, index):
        """

        :param index: int
        :return: LabeledPair
        """
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._dataset)


class LinearRegression:
    _training_dataset = None
    _max_iterations = None
    _alpha = None
    _theta = None

    def __init__(self, max_iterations=200, alpha=0.1):
        self._training_dataset = LabeledRegressionDataSet()
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
        return self._compute_h(LabeledRegressionDataSet.make_tuple(x))


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