import math
from linear_regression import LinearRegression


class LogisticRegression(LinearRegression):
    def _compute_h(self, x):
        h = 1.0 / (1.0 + math.pow(math.e, -1.0 * self._multiply_theta(x)))
        if h >= 0.5:
            return 1
        else:
            return 0


def main():
    lc = LogisticRegression(max_iterations=1000, alpha=0.001)
    lc.train([
        ((1, 1), 1),
        ((4, 2), 1),
        ((2, 5), 1),
        ((7, 14), 1),
        ((5, 3), 1),
        ((6, -2), 1),
        ((4, -2), 1),
        ((6, -5), 1),
        ((2, -1), 1),
        ((-1, 5), 1),
        ((-2, 7), 1),
        ((-1, -6), 0),
        ((-2, -4), 0),
        ((-5, -9), 0),
        ((-3, 1), 0),
        ((-2, 5), 0),
        ((-4, 9), 0),
        ((-5, 4), 0),
        ((2, -6), 0),
        ((1, -10), 0)
    ])

    test = (-1, 6)
    print "Predict for ", test, " is %f" % (lc.predict(test),)


if __name__ == "__main__":
    main()