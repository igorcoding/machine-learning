# !/usr/bin/python
# coding=utf-8

# P(c | d) = P(c) * П [P(tk | c)];
# P(c | d) = log (P(c)) + Sum [ log(P(tk | c)) ]
# P(tk | c) = (Tkc + 1) / (Sum(Tkc) + B)

import math
from pprint import pprint
import itertools


class MultinomialNaiveBayes(object):
    def __init__(self):
        self.V = None
        self.preprocessed = {}
        self.apriory = {}
        self.prob_of_term = {}

    def clean(self):
        self.V = None
        self.preprocessed = {}
        self.apriory = {}
        self.prob_of_term = {}

    def train(self, documents):
        # documents = [(doc, class), ...]
        print "Begin training..."

        self.clean()
        self._preprocess(documents)
        classes = self.preprocessed.keys()

        N = len(documents)
        self.V = self._get_vocabulary(documents)

        for c in classes:
            print "Working on class:", c

            Nc = float(len(self.preprocessed[c]))
            self.apriory[c] = Nc / N

            tokens_c = self._merge_docs_of(c)

            for t in self.V:
                Tkc = float(tokens_c.count(t))
                prob = (Tkc + 1) / float((len(tokens_c) + len(self.V)))
                self._add_prob(t, c, prob)

        print "Finished training\n"

    def _preprocess(self, documents):
        for doc in documents:
            c = doc[1]
            d = doc[0]
            if c not in self.preprocessed:
                self.preprocessed[c] = []
            self.preprocessed[c].append(d.split(' '))

    def _merge_docs_of(self, c):
        docs = self.preprocessed[c]
        merged = itertools.chain(*docs)
        return self._check_tokens(list(merged))

    def _add_prob(self, t, c, prob):
        if t not in self.prob_of_term:
            self.prob_of_term[t] = {}
        self.prob_of_term[t][c] = prob

    @staticmethod
    def _get_vocabulary(documents):
        V = set()
        for doc in documents:
            tokens = doc[0].split(' ')
            for token in tokens:
                V.add(token)
        return V

    def _check_tokens(self, tokens):
        return filter(lambda token: token in self.V, tokens)

    def test(self, document):
        tokens = self._get_tokens(document)
        probabilities = {}

        classes = self.preprocessed.keys()
        for c in classes:
            apriory = math.log(self.apriory[c])
            if c in probabilities:
                probabilities[c] += apriory
            else:
                probabilities[c] = apriory

            for t in tokens:
                probabilities[c] += math.log(self.prob_of_term[t][c])

        m_key, m = self.find_max(probabilities)
        return m_key

    def _get_tokens(self, document):
        doc_tokens = document.split(' ')
        return self._check_tokens(doc_tokens)

    @staticmethod
    def find_max(d):
        m = None
        m_key = None

        for key in d:
            if m is None and m_key is None:
                m = d[key]
                m_key = key
                continue
            if d[key] > m:
                m = d[key]
                m_key = key

        return m_key, m


def main():
    data = [
        ('computer coding c++', 'coding'),
        ('javascript candle tree compiler', 'coding'),
        ('tree plant flower', 'nature'),
        ('nature sun moon mountain', 'nature'),
        ('sun java c++', 'coding')
    ]

    nb = MultinomialNaiveBayes()

    print "Train data: "
    pprint(data)

    nb.train(data)

    test_data = 'sun moon'
    c = nb.test(test_data)
    print "Test data:", test_data
    print "Test data predicted class:", c


if __name__ == '__main__':
    main()