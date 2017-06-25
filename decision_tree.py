# coding=utf-8
import math
from collections import OrderedDict
from pprint import pprint
import random
import uuid
import graphviz as gv
import numpy as np

import functools

import pandas as pd

graph = functools.partial(gv.Graph, format='png')
digraph = functools.partial(gv.Digraph, format='png')


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


class Node:
    def __init__(self):
        self.id = uuid.uuid4()
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

    def __str__(self):
        if self.attr_id is None:
            return '{label = ' + unicode(self.label) + '},'

        s = '{attr_id = ' + str(self.attr_id) + ', children = {'
        for ch in self.children:
            s += unicode(ch) + ':' + unicode(self.children[ch])
        s += '}}'
        return s

    def graph(self, attributes_meta=None, labels_meta=None):
        g = graph()
        if self.attr_id is not None:
            if attributes_meta:
                label = attributes_meta[self.attr_id][0]
            else:
                label = str(self.attr_id)
            g.node(str(self.id), label=unicode(label))
        elif self.attr_id is None and self.label:
            if labels_meta:
                label = self.label
                color = labels_meta[self.label]['color']
            else:
                label = self.label
                color = 'yellow'
            g.node(str(self.id), label=unicode(label), shape='hexagon', style='filled', fillcolor=color)

        for ch in self.children:
            g.subgraph(self.children[ch].graph(attributes_meta, labels_meta))

            if attributes_meta and attributes_meta[self.attr_id][1] == 'n':
                edge_label = ch.format('x')
            else:
                edge_label = ch
            g.edge(str(self.id), str(self.children[ch].id), label=unicode(edge_label))

        return g


class DecisionTree:
    def __init__(self, attributes_meta=None, labels_meta=None):
        self.root = None
        self.attributes_meta = attributes_meta
        self.labels_meta = labels_meta

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

        all_gains = OrderedDict()

        for attr_id in xrange(attrs_count):
            splitted = self.split_dataset(dataset, attr_id)

            subsets_entropy = 0.0
            for t in splitted.values():
                t_p = float(len(t)) / float(len(dataset))
                t_entropy = self.entropy(t)[0]
                subsets_entropy += -t_p * t_entropy

            gain = dataset_entropy + subsets_entropy
            all_gains[self.attributes_meta[attr_id][0]] = gain

            if (max_gain is None and max_gained_attr_id is None and max_gained_splits is None) or gain > max_gain:
                max_gain = gain
                max_gained_attr_id = attr_id
                max_gained_splits = splitted

        # print(pd.DataFrame(pd.Series(all_gains), columns=['Gain']).to_html())

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
                        d[attr_id] = '{}>=' + str(attr_very_max)
                    else:
                        f = math.ceil(float(old_value - attr_min) / width) - 1
                        if f < 0:
                            f = 0
                        interval_min = attr_min + f * width
                        interval_max = attr_min + (f + 1) * width
                        d[attr_id] = str(interval_min) + '<={}<' + str(interval_max)

                    new_dataset.append(tuple(d))
        if len(new_dataset) == 0:
            return dataset
        return new_dataset

    def train(self, dataset):
        if self.attributes_meta:
            dataset = self.prepare_dataset(dataset)
        self.root = self._build_subtree(dataset)

    def _build_subtree(self, dataset, i=1):
        attr_id, subsets = self.best_attribute(dataset)

        if attr_id is None:
            label = subsets
            node = Node.create_leaf(label)
        elif len(subsets) == 1:
            arr = subsets.values()[0]
            label = arr[random.randint(0, len(arr) - 1)][-1]
            node = Node.create_leaf(label)
        else:
            node = Node()
            node.attr_id = attr_id

            for attr_val, subset in subsets.iteritems():
                # print('-' * i)
                # print(u"{}: {}".format(self.attributes_meta[attr_id][0], attr_val))
                # print(pd.DataFrame(subset, columns=map(lambda a: a[0], self.attributes_meta) + [u'Доступность']).to_html())
                # print('-' * i)
                n = self._build_subtree(subset, i + 1)
                node.add_child(attr_val, n)

        return node

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if len(X.shape) > 1:
                return np.asarray([self._predict(x, self.root) for x in X])
        elif isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], np.collections.Iterable):
                return [self._predict(x, self.root) for x in X]
        self._predict(X, self.root)

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

    def __str__(self):
        return str(self.root)

    def graph(self):
        g = self.root.graph(self.attributes_meta, self.labels_meta)
        return g


def main():
    attributes = ((u'Температура', 'n'), (u'Боль в горле', 'c'), (u'Насморк', 'c'), (u'Хрип в горле', 'c'))
    labels = (u'Здоров', u'Грипп', u'ОРЗ', u'Воспаление легких')
    medicine = (u'Аспирин', u'Нафтизин', u'Антибиотик', u'Молоко с медом')
    medicine = {
        u'None': {
            'color': 'grey'
        },
        u'Аспирин': {
            'color': 'cyan'
        },
        u'Нафтизин': {
            'color': 'red'
        },
        u'Антибиотик': {
            'color': '#87D958'
        },
        u'Молоко с медом': {
            'color': 'yellow'
        },

    }

    dataset2 = [
        (36.6, False, False, False, u'Молоко с медом'),
        (36.6, False, False, True, u'Молоко с медом'),
        (36.6, False, True, False, u'Молоко с медом'),
        (36.6, False, True, True, u'Молоко с медом'),
        (36.7, True, False, False, u'Аспирин'),
        (36.8, True, False, True, u'Аспирин'),
        (36.9, True, True, False, u'Аспирин'),
        (36.7, True, True, True, u'Антибиотик'),

        (37, False, False, False, u'Молоко с медом'),
        (37.3, False, False, True, u'Аспирин'),
        (37.2, False, True, False, u'Нафтизин'),
        (37.5, False, True, True, u'Аспирин'),
        (37.9, True, False, False, u'Аспирин'),
        (37.9, True, False, True, u'Аспирин'),
        (37.4, True, True, False, u'Молоко с медом'),
        (37.8, True, True, True, u'Аспирин'),

        (38, False, False, False, u'Аспирин'),
        (38.2, False, False, True, u'Аспирин'),
        (38.4, False, True, False, u'Аспирин'),
        (38.5, False, True, True, u'Аспирин'),
        (38.9, True, False, False, u'Аспирин'),
        (38.3, True, False, True, u'Антибиотик'),
        (38.6, True, True, False, u'Антибиотик'),
        (38.9, True, True, True, u'Антибиотик'),

        (39, False, False, False, u'Антибиотик'),
        (39.2, False, False, True, u'Антибиотик'),
        (39.6, False, True, False, u'Антибиотик'),
        (39.3, False, True, True, u'Антибиотик'),
        (39.5, True, False, False, u'Антибиотик'),
        (39.7, True, False, True, u'Антибиотик'),
        (39.8, True, True, False, u'Антибиотик'),
        (39.9, True, True, True, u'Антибиотик'),
    ]

    dtree = DecisionTree(attributes, medicine)
    dtree.train(dataset2)

    pred = dtree.predict((38, True, True, False))
    pprint(unicode(dtree.root))

    g = dtree.graph()
    g.render('img/g')
    pass

if __name__ == '__main__':
    main()
