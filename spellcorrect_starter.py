from decimal import Decimal

eps = 0.0001

def delete_edits(w):
    # Return the set of strings that can be formed by applying one
    # delete operation to word w
    # "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    charecter     = [(w[:i], w[i:])    for i in range(len(w) + 1)]
    deletion    = [left_key + right_key[1:]               for left_key, right_key in charecter if right_key]
    insertion    = [left_key + c + right_key               for left_key, right_key in charecter for c in letters]
    transposition = [left_key + right_key[1] + right_key[0] + right_key[2:] for left_key, right_key in charecter if len(right_key)>1]
    substitution   = [left_key + c + right_key[1:]           for left_key, right_key in charecter if right_key for c in letters]

    return set(deletion + transposition + substitution + insertion)

class UnsmoothedUnigramLM:
    def __init__(self, fname):
        self.freqs = {}
        self.uniwords = {}
        self.uniwords = set(self.uniwords)
        for line in open(fname):
            tokens = line.split()
            for i, t in enumerate(tokens):
                self.freqs[t] = self.freqs.get(t, 0) + 1

        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def log_prob_for_unigram(self, word):

        if word in self.freqs:
            return Decimal((self.freqs[word] + 1)/(self.num_tokens + self.num_types))
        else:
            return float("-inf")

    def in_vocab(self, word):
        return word in self.freqs

    def check_probs(self):

        log_prob_value = float("InF")
        total_prob_value = Decimal()
        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            log_prob_value = self.log_prob_for_unigram(w)
            total_prob_value = total_prob_value + log_prob_value
            assert 0 - eps < log_prob_value < 1 + eps

        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
           total_prob_value < \
            1 + eps
    
class UnsmoothedNgramLM:
    def __init__(self, fname):
        self.freqs = {}
        self.bifreqs = {}

        for line in open(fname):
            tokens = line.split()
        
            for i, t in enumerate(tokens):
                self.freqs[t] = self.freqs.get(t, 0) + 1
                
                if i < len(tokens) - 1:

                    if (tokens[i], tokens[i+1]) in self.bifreqs:
                        self.bifreqs[(tokens[i], tokens[i + 1])] += 1
                    else:
                        self.bifreqs[(tokens[i], tokens[i + 1])] = 1

        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def check_probs(self):
        log_prob_value = float("InF")
        total_prob_value = Decimal()
        # Make sure the probability for each word is between 0 and 1
        for w in self.bifreqs:
            log_prob_value = self.log_prob_for_bigram(w, None, None)
            total_prob_value = total_prob_value + log_prob_value
            assert 0 - eps < log_prob_value < 1 + eps

        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
           Decimal(total_prob_value) < \
            1 + eps

    def log_prob_for_bigram(self, word, next_word, current_word):
        if next_word and current_word:
            current_word_total = self.freqs[current_word] if current_word in self.freqs else 0
            next_word_total = self.bifreqs[next_word] if next_word in self.bifreqs else 0
            previous_word_total = self.bifreqs[word] if word in self.bifreqs else 0

            return Decimal(((previous_word_total + 1)/(current_word_total + self.num_types)) * ((next_word_total + 1)/(current_word_total + self.num_types)))
        else:
            word_total = self.bifreqs[word] if word in self.bifreqs else 0
            current_word_total = self.freqs[word[1]] if word[1] in self.freqs else 0
            
            return Decimal(word_total + 1)/(current_word_total + self.num_types)

    def in_vocab(self, word):
        return word in self.freqs

    
    def log_prob_for_interp(self, word, next_word, current_word):
        if next_word and current_word:
            current_word_total = self.freqs[current_word] if current_word in self.freqs else 0
            next_word_total = self.bifreqs[next_word] if next_word in self.bifreqs else 0
            previous_word_total = self.bifreqs[word] if word in self.bifreqs else 0

            lamda = 0.2

            bigram_prob_1 = (previous_word_total/current_word_total)
            bigram_prob_2 = (next_word_total/current_word_total)

            next_single_word_total = self.freqs[next_word[1]] if next_word[1] in self.freqs else 0

            unigram_prod_1 = (current_word_total + 1)/(self.num_tokens + self.num_types)
            unigram_prod_2 = (next_single_word_total + 1)/(self.num_tokens + self.num_types)

            return Decimal((((1 - lamda) * bigram_prob_1) + (lamda * unigram_prod_1)) * (((1 - lamda) * bigram_prob_2) + (lamda * unigram_prod_2)))
        else:
            word_total = self.bifreqs[word] if word in self.bifreqs else 0
            current_word_total = self.freqs[word[1]] if word[1] in self.freqs else 0
            
            return Decimal(word_total + 1)/(current_word_total + self.num_types)

if __name__ == '__main__':
    import sys

    # Look for the training corpus in the current directory
    train_corpus = 'corpus.txt' 

    # n will be '1', '2' or 'interp' (but this starter code ignores
    # this)
    n = str(sys.argv[1])
    # n = "interp"

    # The collection of sentences to make predictions for
    predict_corpus = sys.argv[2]
    # predict_corpus = "dev.txt"

    # Train the language model
    if (n == "1"):
       lm = UnsmoothedUnigramLM(train_corpus)
    elif (n == "2" or n == "interp"):
        lm = UnsmoothedNgramLM(train_corpus)
    else:
        print("Provide proper input")

    # You can comment this out to run faster...
    lm.check_probs()

    for line in open(predict_corpus):
        # Split the line on a tab; get the target word to correct and
        # the sentence it's in
        target_index,sentence = line.split('\t')
        target_index = int(target_index)
        sentence = sentence.split()
        target_word = sentence[target_index]
        previous_word = sentence[target_index - 1]
        next_word = sentence[target_index + 1]

        # Get the in-vocabulary candidates (this starter code only
        # considers deletions)
        candidates = delete_edits(target_word)
        iv_candidates = [c for c in candidates if lm.in_vocab(c)]
        
        best_prob = float("-inf")
        best_correction = target_word
        word = {}
        for ivc in iv_candidates:
            if n == "1":
                ivc_log_prob = lm.log_prob_for_unigram(ivc)
            elif n == "2":
                word1 = (previous_word, ivc)
                word2 = (ivc, next_word)
                ivc_log_prob = lm.log_prob_for_bigram(word1, word2, ivc)
            elif n == "interp":
                word1 = (previous_word, ivc)
                word2 = (ivc, next_word)
                ivc_log_prob = lm.log_prob_for_interp(word1, word2, ivc)
            else:
                print("Provide proper input")
            if ivc_log_prob > best_prob:
                best_prob = ivc_log_prob
                best_correction = ivc

        print(best_correction)
