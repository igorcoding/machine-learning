import collections


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


class LabeledDataSet:
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