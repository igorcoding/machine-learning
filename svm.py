from numpy import *
import random


class SVM:
    epsilon = 1e-3

    def __init__(self):
        self.alphas = None
        self.b = None
        self.dataset = None
        self.C = None
        self.K = None

    def train(self, max_iterations, c, kernel_function, dataset):
        self.C = c
        self.K = kernel_function
        self.dataset = dataset

        m = len(dataset)  # Data size
        n = dataset.feature_size  # First example's first component which is a feature vector X

        self.b = 0

        self.alphas = zeros(m, 1)

        t = 0
        while t < max_iterations:

            alphas_changes_count = 0
            for i in xrange(0, m):
                xi = dataset[i]
                ui = self.predict(xi.x)
                Ei = ui - xi.y

                j = self._pick_second(i)
                xj = dataset[j]
                uj = self.predict(xj.x)
                Ej = uj - xj.y

                # some condition goes here

                eta = 1.0 / (self.K(xi.x, xi.x) + self.K(xj.x, xj.x) - 2.0 * self.K(xi.x, xj.x))

                old_alpha_i = self.alphas[i]
                old_alpha_j = self.alphas[j]

                self.alphas[j] += xj.y * (Ei - Ej) / eta

                if xi.y == xj.y:
                    L = max(0, self.alphas[i] + self.alphas[j] + self.C)
                    H = min(self.C, self.alphas[i] + self.alphas[j])
                else:
                    L = max(0, self.alphas[j] - self.alphas[i])
                    H = min(self.C, self.C + self.alphas[j] - self.alphas[i])

                if self.alphas[j] >= H:
                    self.alphas[j] = H
                elif self.alphas[j] <= L:
                    self.alphas[j] = L

                s = xi.y * xj.y
                self.alphas[i] += s * (old_alpha_j - self.alphas[j])

                b1 = Ei + xi.y * (self.alphas[i] - old_alpha_i) * self.K(xi.x, xi.x) \
                        + xj.y * (self.alphas[j] - old_alpha_j) * self.K(xi.x, xj.x) + self.b

                b2 = Ej + xi.y * (self.alphas[i] - old_alpha_i) * self.K(xi.x, xj.x) \
                        + xj.y * (self.alphas[j] - old_alpha_j) * self.K(xj.x, xj.x) + self.b

                self.b = (b1 + b2) / 2.0

                alphas_changes_count += 1

            if alphas_changes_count == 0:
                t += 1
            else:
                t = 0

    def predict(self, x):
        s = 0
        for j in xrange(0, len(self.dataset)):
            data_point = self.dataset[j]
            s += data_point.y * self.alphas[j] * self.K(data_point.x, x)
        return s

    def _pick_second(self, i):
        def _pick_internal():
            return random.randint(0, len(self.dataset) - 1)

        j = _pick_internal()
        while j == i:
            j = _pick_internal()

        return j


def main():
    svm = SVM()


if __name__ == '__main__':
    main()