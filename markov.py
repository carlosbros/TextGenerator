############################################################
# Markov Models
############################################################

import string 
import random 
import math 

def tokenize(text):
    tokens = [token.strip() for token in text.split()]
    final_list = []
    for tok in tokens:
        next_token = ""
        prev = len(final_list)
        for c in tok:
            if c in string.punctuation:
                if next_token != "":
                    final_list.append(next_token)
                final_list.append(c)
                next_token = ""
            else:
                next_token += c
        if prev == len(final_list):
            final_list.append(next_token)
    return final_list


def ngrams(n, tokens):
    inp = list(tokens)
    for _ in xrange(n - 1):
        inp.insert(0, '<START>')
    inp.append('<END>')
    ngrams = [(tuple([inp[i - (n - 1) + x] for x in xrange(n - 1)]),
               inp[i]) for i in xrange(len(inp)) if i >= n - 1]
    return ngrams


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.prev_counts = {}
        pass

    def update(self, sentence):
        tokens = tokenize(sentence)
        n_grams = ngrams(self.n, tokens)
        for prev, w in n_grams:
            context = prev, w
            self.counts[context] = self.counts.get(context, 0) + 1
            self.prev_counts[prev] = self.prev_counts.get(prev, 0) + 1

    def prob(self, context, token):
        return (self.counts.get((context, token), 0) /
                float(self.prev_counts.get(context, 0)))

    def random_token(self, context):
        words = [w for prev, w in self.counts if prev == context]
        words.sort()
        r = random.random()
        left_prob_sum = 0
        right_prob_sum = 0
        for i in xrange(len(words) - 1):
            if i != 0:
                left_prob_sum += self.prob(context, words[i - 1])
            right_prob_sum += self.prob(context, words[i])
            if left_prob_sum <= r and r < right_prob_sum:
                return words[i]
        return words[len(words) - 1]

    def reset_context(self):
        return ["<START>" for _ in xrange(self.n - 1)]

    def update_context(self, context, next_token):
        for i in xrange(len(context) - 1):
            context[i] = context[i + 1]
        if len(context) > 0:
            context[len(context) - 1] = next_token

    def random_text(self, token_count):
        text = ""
        context = self.reset_context()
        while token_count > 0:
            next_token = self.random_token(tuple(context))
            text += next_token + " "
            if self.n > 1:
                if next_token == '<END>':
                    context = self.reset_context()
                else:
                    self.update_context(context, next_token)
            token_count -= 1
        return text.strip()

    def perplexity(self, sentence):
        tokens = tokenize(sentence)
        n_grams = ngrams(self.n, tokens)
        log_prob = 0
        for prev, w in n_grams:
            log_prob += math.log(self.prob(prev, w))
        prob = math.exp(log_prob)
        return (1 / prob) ** (len(n_grams) ** -1)


def create_ngram_model(n, path):
    m = NgramModel(n)
    with open(path) as f:
        t = f.readlines()
    for line in t:
        m.update(line)
    return m
