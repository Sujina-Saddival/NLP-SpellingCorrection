import math
from decimal import Decimal

eps = 0.0001

class UnsmoothedUnigramLM:
    def __init__(self, fname):
        self.freqs = {}
        self.uniwords = {}
        self.uniwords = set(self.uniwords)
        for line in open(fname):
            tokens = line.split()
            for i, t in enumerate(tokens):
                self.freqs[t] = self.freqs.get(t, 0) + 1
                # self.uniwords.add(t[1])

        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def log_prob(self, word):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        if word in self.freqs:
            return math.log(self.freqs[word]) - math.log(self.num_tokens)
        else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            return float("-inf")

    def log_prob_for_unigram(self, word):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        # if word in self.freqs:
        #     return math.log(self.freqs[word])/math.log(self.num_tokens)
        # else:
        #     # This is a bit of a hack to get a float with the value of
        #     # minus infinity for words that have probability 0
        #     return float("-inf")

        if word in self.freqs:
            return Decimal((self.freqs[word] + 1)/(self.num_tokens + self.num_types))
        else:
            return float("-inf")

    def log_prob_for_bigram(self, first_word, second_word):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        if first_word in self.freqs and second_word in self.freqs:
            return Decimal((self.freqs[word] + 1)/(self.num_tokens + self.num_types))
        else:
            return float("-inf")

    def in_vocab(self, word):
        return word in self.freqs

    def check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        log_prob_value = float("InF")
        total_prob_value = Decimal()
        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            log_prob_value = self.log_prob_for_unigram(w)
            total_prob_value = total_prob_value + log_prob_value
            assert 0 - eps < log_prob_value < 1 + eps

        # Make sure that the sum of probabilities for all words is 1
        # assert 1 - eps < \
        #     sum([math.exp(log_prob_value) for w in self.freqs]) < \
        #     1 + eps
        assert 1 - eps < \
           total_prob_value < \
            1 + eps

def delete_edits(word):
    # Return the set of strings that can be formed by applying one
    # delete operation to word w
    # result = set()
    # for i in range(len(w)):
    #     result.add(w[:i] + w[i+1:])
    # return result

    # "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    charecter     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletion    = [L + R[1:]               for L, R in charecter if R]
    insertion    = [L + c + R               for L, R in charecter for c in letters]
    transposition = [L + R[1] + R[0] + R[2:] for L, R in charecter if len(R)>1]
    substitution   = [L + c + R[1:]           for L, R in charecter if R for c in letters]

    return set(deletion + transposition + substitution + insertion)
    
class UnsmoothedBigramLM:
    def __init__(self, fname):
        self.freqs = {}
        self.bifreqs = {}
        for line in open(fname):
            tokens = line.split()
            self.each_document_linestokens = tokens
            for i, t in enumerate(tokens):
                if(i + 1 < len(tokens)):
                    bigram_key = t + " " + tokens[i+1]
                    self.bifreqs[bigram_key] = self.bifreqs.get(bigram_key, 0) + 1

    def check_probs(self):
        log_prob_value = float("InF")
        total_prob_value = Decimal()
        # Make sure the probability for each word is between 0 and 1
        for w in self.bifreqs:
            log_prob_value = self.log_prob_for_bigram(w)
            total_prob_value = total_prob_value + log_prob_value
            assert 0 - eps < math.exp(log_prob_value) < 1 + eps

        # Make sure that the sum of probabilities for all words is 1
        # assert 1 - eps < \
        #     sum([math.exp(log_prob_value) for w in self.freqs]) < \
        #     1 + eps
        assert 1 - eps < \
           total_prob_value < \
            1 + eps

if __name__ == '__main__':
    import sys

    # Look for the training corpus in the current directory
    train_corpus = 'corpus.txt' 

    # n will be '1', '2' or 'interp' (but this starter code ignores
    # this)
    n = sys.argv[1]
    # n = 1

    # The collection of sentences to make predictions for
    predict_corpus = sys.argv[2]
    # predict_corpus = "dev.txt"

    # Train the language model
    if (n == 1):
       lm = UnsmoothedUnigramLM(train_corpus)
    elif (n == 2):
        lm = UnsmoothedBigramLM(train_corpus)
    elif (n == "interp"):
        lm = UnsmoothedBigramLM(train_corpus)
    else:
        lm = UnsmoothedUnigramLM(train_corpus)

    # You can comment this out to run faster...
    lm.check_probs()
    

    for line in open(predict_corpus):
        # Split the line on a tab; get the target word to correct and
        # the sentence it's in
        target_index,sentence = line.split('\t')
        target_index = int(target_index)
        sentence = sentence.split()
        target_word = sentence[target_index]

        # Get the in-vocabulary candidates (this starter code only
        # considers deletions)
        candidates = delete_edits(target_word)
        iv_candidates = [c for c in candidates if lm.in_vocab(c)]
        
        # Find the candidate correction with the highest probability;
        # if no candidate has non-zero probability, or there are no
        # candidates, give up and output the original target word as
        # the correction.
        best_prob = float("-inf")
        best_correction = target_word
        for ivc in iv_candidates:
            if n == 1:
                ivc_log_prob = lm.log_prob_for_unigram(ivc)
            elif n == 2:
                ivc_log_prob = lm.log_prob_for_bigram(ivc)
            elif n == "interp":
                ivc_log_prob = lm.log_prob(ivc)
            else:
                ivc_log_prob = lm.log_prob(ivc)
            # ivc_log_prob = lm.log_prob(ivc)
            if ivc_log_prob > best_prob:
                best_prob = ivc_log_prob
                best_correction = ivc

        print(best_correction)
