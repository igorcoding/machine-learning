# coding=utf-8
import math
import numpy as np


class Node:
    def __init__(self):
        self.attr_id = None
        self.values = None
        self.children = {}
        self.leaf = False
        self.label = None

    def add_child(self, attr_value, node):
        assert attr_value not in self.children
        self.children[attr_value] = node
        return self.children[attr_value]

    def set(self, attr_id, values):
        self.attr_id = attr_id
        self.values = values

    def add_leaf(self, attr_value, label):
        leaf = Node()
        leaf.label = label
        self.children[attr_value] = leaf
        return self.children[attr_value]

    def is_leaf(self):
        return self.leaf

    @staticmethod
    def create_leaf(label):
        n = Node()
        n.leaf = True
        n.label = label
        return n


class DecisionTree:
    def __init__(self, attributes_meta=None):
        self.root = None
        self.attributes_meta = attributes_meta

    def entropy(self, dataset):
        info = {}
        for d in dataset:
            class_label = d[-1]
            if class_label in info:
                info[class_label] += 1
            else:
                info[class_label] = 1

        res = 0.0
        s = len(dataset)
        max_label = None
        max_label_count = None
        for label, count in info.iteritems():
            p = float(count) / float(s)
            res += -p * math.log(p, 2.0)

            if (max_label_count is None and max_label is None) or count > max_label_count:
                max_label_count = count
                max_label = label
        return res, max_label

    def best_attribute(self, dataset):
        attrs_count = len(dataset[0]) - 1

        dataset_entropy, max_label = self.entropy(dataset)
        if dataset_entropy == 0:
            return None, max_label

        max_gain = None
        max_gained_attr_id = None
        max_gained_splits = None
        for attr_id in xrange(attrs_count):
            splitted = self.split_dataset(dataset, attr_id)

            subsets_entropy = 0.0
            for t in splitted.values():
                t_p = float(len(t)) / float(len(dataset))
                t_entropy = self.entropy(t)[0]
                subsets_entropy += -t_p * t_entropy

            gain = dataset_entropy + subsets_entropy

            if (max_gain is None and max_gained_attr_id is None and max_gained_splits is None) or gain > max_gain:
                max_gain = gain
                max_gained_attr_id = attr_id
                max_gained_splits = splitted

        return max_gained_attr_id, max_gained_splits

    @staticmethod
    def split_dataset(dataset, attr_id):
        """
        :param dataset:
        :param attr_id:
        :return: dict where keys are attribute values and values are subsets
        :rtype: dict
        """

        res = {}

        for d in dataset:
            attr_value = d[attr_id]
            if attr_value in res:
                res[attr_value].append(d)
            else:
                res[attr_value] = [d]

        return res

    def prepare_dataset(self, dataset):
        k = 3
        new_dataset = []
        for attr_id, attr_meta in enumerate(self.attributes_meta):
            attr_name = attr_meta[0]
            attr_type = attr_meta[1]

            if attr_type == 'n':
                attr_max = None
                attr_min = None

                for d in dataset:
                    if attr_max is None or d[attr_id] > attr_max:
                        attr_max = d[attr_id]
                    if attr_min is None or d[attr_id] < attr_min:
                        attr_min = d[attr_id]

                width = float(attr_max - attr_min) / float(k)

                # attr_very_min = attr_min
                # attr_very_max = attr_max
                attr_very_min = attr_min + width
                attr_very_max = attr_min + (k - 1) * width

                for d in dataset:
                    d = list(d)
                    old_value = d[attr_id]
                    if old_value < attr_very_min:
                        d[attr_id] = '{}<' + str(attr_very_min)
                    elif old_value > attr_very_max:
                        d[attr_id] = '{}>' + str(attr_very_max)
                    else:
                        f = math.ceil(float(old_value - attr_min) / width) - 1
                        if f < 0:
                            f = 0
                        interval_min = attr_min + f * width
                        interval_max = attr_min + (f + 1) * width
                        d[attr_id] = str(interval_min) + '<={}<' + str(interval_max)

                    new_dataset.append(tuple(d))
        return new_dataset

    def train(self, dataset):
        if self.attributes_meta:
            dataset = self.prepare_dataset(dataset)
        self.root = self._build_subtree(dataset)

    def _build_subtree(self, dataset):
        attr_id, subsets = self.best_attribute(dataset)
        if attr_id is None:
            label = subsets
            node = Node.create_leaf(label)
        else:
            node = Node()
            node.attr_id = attr_id

            for attr_val, subset in subsets.iteritems():
                n = self._build_subtree(subset)
                node.add_child(attr_val, n)

        return node

    def predict(self, entry):
        return self._predict(entry, self.root)

    def _predict(self, entry, node):
        if node.is_leaf():
            return node.label
        attr_id = node.attr_id
        for attr_value in node.children:
            if self._attr_equal(attr_id, entry[attr_id], attr_value):
                next_node = node.children[attr_value]
                return self._predict(entry, next_node)

    def _attr_equal(self, attr_id, entry_attr, tree_attr):
        if self.attributes_meta and self.attributes_meta[attr_id][1] == 'n':
            expr = tree_attr.format(entry_attr)
            return eval(expr)

        return entry_attr == tree_attr



def main():
    dataset1 = [
        ('Rainy', 'Hot', 'High', False, 'No'),
        ('Rainy', 'Hot', 'High', True, 'No'),
        ('Overcast', 'Hot', 'High', False, 'Yes'),
        ('Sunny', 'Mild', 'High', False, 'Yes'),
        ('Sunny', 'Cool', 'Normal', False, 'Yes'),
        ('Sunny', 'Cool', 'Normal', True, 'No'),
        ('Overcast', 'Cool', 'Normal', True, 'Yes'),
        ('Rainy', 'Mild', 'High', False, 'No'),
        ('Rainy', 'Cool', 'Normal', False, 'Yes'),
        ('Sunny', 'Mild', 'Normal', False, 'Yes'),
        ('Rainy', 'Mild', 'Normal', True, 'Yes'),
        ('Overcast', 'Mild', 'High', True, 'Yes'),
        ('Overcast', 'Hot', 'Normal', False, 'Yes'),
        ('Sunny', 'Mild', 'High', True, 'No'),
    ]

    attributes = ((u'Температура', 'n'), (u'Боль в горле', 'c'), (u'Насморк', 'c'), (u'Хрип в горле', 'c'))
    labels = (u'Здоров', u'Грипп', u'ОРЗ', u'Воспаление легких')
    medicin = (None, u'Аспирин', u'Нафтизин', u'Антибиотик', u'Молоко с медом')
    #          0       1           2            3              4

    dataset2 = [
        (36.6, False, False, False, 0),
        (36.6, False, False, True, 4),
        (36.6, False, True, False, 4),
        (36.6, False, True, True, 4),
        (36.6, True, False, False, 1),
        (37.2, True, False, True, 1),
        (37.6, True, True, False, 1),
        (37.7, True, True, True, 3),

        (37, False, True, False, 2),
        (37, False, True, True, 1),
    ]

    dtree = DecisionTree(attributes)
    dtree.train(dataset2)

    pred = dtree.predict((50, True, True, True))
    print medicin[pred]
    pass

if __name__ == '__main__':
    main()