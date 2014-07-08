# !/usr/bin/python

import math
from pprint import pprint


class MultinomialNaiveBayes(object):
    def __init__(self):
        self.V = None
        self.classes = None
        self.apriory = {}
        self.prob_of_term = {}

    def clean(self):
        self.V = None
        self.classes = None
        self.apriory = {}
        self.prob_of_term = {}

    def train(self, documents):
        # documents = [(doc, class), ...]
        print "Begin training..."

        self.clean()
        self.classes = self._find_classes(documents)

        N = len(documents)
        self.V = self._get_vocabulary(documents)

        for c in self.classes:
            print "Working on class:", c

            Nc = float(self._count_docs_of(c, documents))
            self.apriory[c] = Nc / N

            tokens_c = self._merge_of_class(c, documents)

            for t in self.V:
                Tkc = float(tokens_c.count(t))
                prob = (Tkc + 1) / float((len(tokens_c) + len(self.V)))
                self._add_prob(t, c, prob)

        print "Finished training\n"

    def _add_prob(self, t, c, prob):
        if t not in self.prob_of_term:
            self.prob_of_term[t] = {}
        self.prob_of_term[t][c] = prob

    def _get_vocabulary(self, documents):
        V = set()
        for doc in documents:
            tokens = doc[0].split(' ')
            for token in tokens:
                V.add(token)

        return V

    def _merge_of_class(self, c, documents):
        text_c = ''
        for doc in documents:
            if doc[1] == c:
                text_c += doc[0] + ' '

        return self._split_into_tokens(text_c)

    def _split_into_tokens(self, text):
        tokens = []
        text_tokens = text.split(' ')
        for t in text_tokens:
            if t in self.V:
                tokens.append(t)
        return tokens

    def _count_docs_of(self, c, documents):
        n = 0
        for doc in documents:
            if doc[1] == c:
                n += 1
        return n

    def _find_classes(self, documents):
        classes = set()
        for pair in documents:
            classes.add(pair[1])
        return classes

    def test(self, document):
        tokens = self._get_tokens(document)
        probabilities = {}
        for c in self.classes:
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
        tokens = []
        doc_tokens = document.split(' ')
        for t in doc_tokens:
            if t in self.V:
                tokens.append(t)
        return tokens

    def find_max(self, d):
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