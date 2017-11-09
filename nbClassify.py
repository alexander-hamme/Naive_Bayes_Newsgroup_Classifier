'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Alex Hamme

Date: 7.Nov.2017

'''
from __future__ import print_function
import pickle
import random
import time
import math
import sys
import os


class NaiveBayes():
    '''Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''

    def __init__(self, method, train=None, save=False):
        '''Create classifier using train, the name of an input
        training file.
        '''
        self.method = method                        # "raw", "mest", or "tfidf"
        self.vocab = []
        self.categoryPriorProbs = dict()            # Dict of prior probabilities for each category
        self.categorySizes = dict()
        self.final_dict = dict()
        self.filename = "word_frequencies_dict"     # filename to save to
        self.fullVocSize = 0
        self.uniqueVocSize = 0                      # unique number of words in total vocab
        self.numbCategories = 20.0
        if train:
            self.learn(train, save=save)            # loads train data, fills probability table

    def load_pickle_file(self, filename):
        with open(filename + '.pickle', 'rb') as handle:
            self.final_dict = pickle.load(handle)

        self.categoryPriorProbs = self.final_dict.pop("categoryPriorProbs", None)
        self.categorySizes = self.final_dict.pop("categorySizes", None)

        if not (self.categoryPriorProbs and self.categorySizes):
            raise SystemExit("Could not load probabilities or prior probabilities from serialized dictionary")

        print("loaded pickle file")

    def printTrain(self, prob_dict, numb_lines):

        print("############### TRAIN OUTPUT #########################")
        print("Total # words\t{}".format(self.fullVocSize))
        print("VocabSize\t{}".format(self.uniqueVocSize))
        print("-"*103)
        print("Category\t\t\tNDoc\t\t\t    NWords\t\t\tP(cat)")

        for cat, dct in prob_dict.items():
            print("{:<25} {:>10}\t\t\t{:>10}\t\t\t{:>10}".format(
                cat, numb_lines.get(cat), len(dct), self.categoryPriorProbs.get(cat))
            )

    def printTest(self, cat_stats, cat_lines):
        print("\n############### TEST OUTPUT #########################")
        print("Category\t\t\tNCorrect\t\t\t N\t\t\t%corr")
        for cat, numb in cat_stats.items():
            print("{:<25} {:>10}\t\t\t{:>10}\t\t\t{:>10}".format(
                cat, numb, cat_lines.get(cat), numb / float(cat_lines.get(cat))
            ))


    def load_text_file(self, filename):

        category_dicts = dict()     # divy words up into their categories. Of the form {"cat": [list of words], ...}

        full_vocab = []             # list of all words in document

        numb_lines = {}             # keep track of number of lines for each category, for prior probability calculation

        with open(filename, 'rb') as fd:

            document = fd.readlines()

            for i, line in enumerate(document):

                data = line.split()

                cat, words = data[0], data[1:]

                if cat in category_dicts.keys():
                    category_dicts[cat].extend(words)
                else:
                    category_dicts[cat] = words

                if cat in numb_lines:
                    numb_lines[cat] = numb_lines.get(cat) + 1
                else:
                    numb_lines[cat] = 1

                full_vocab.extend(words)

        total_lines = float(sum(numb_lines.values()))

        for cat, lst in category_dicts.items():
            self.categorySizes[cat] = len(lst)

        for cat, count in numb_lines.items():                           # calculate prior probabilities of categories
            self.categoryPriorProbs[cat] = count / total_lines          # P(Vj) = |docsj| / |Examples|

        self.fullVocSize = len(full_vocab)
        self.uniqueVocSize = len(set(full_vocab))

        self.printTrain(category_dicts, numb_lines)

        return category_dicts, full_vocab

    def learn(self, traindat, save=False):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''

        print("\nLearning...\n")

        category_dicts, full_vocab = self.load_text_file(traindat)

        assert len(category_dicts) > 1

        for i, cat in enumerate(category_dicts.keys()):

            all_cat_words = category_dicts.get(cat)        # type --> list of words

            unique_cat_words = list(set(all_cat_words))

            """
            
            Sort the lists, so that counting occurrences of each word can be done by
            one single iteration through the whole list, chunk by chunk, 
            instead of searching the entire list for every single word

            Also, converting lists to tuples before iteration gives a small boost in speed.
            """

            all_cat_words.sort()
            unique_cat_words.sort()

            all_cat_words = tuple(all_cat_words)
            unique_cat_words = tuple(unique_cat_words)

            counts = {}

            last_idx = 0
            for wrd in unique_cat_words:
                count = 0
                for w in all_cat_words[last_idx:]:          # continue from where iteration stopped at last word
                    if w == wrd:
                        count += 1
                        last_idx += 1
                    elif w > wrd:
                        break

                counts[wrd] = count

            print("Progress: {} of 20 categories processed".format(i+1))

            self.final_dict[cat] = counts    # dictionary of dictionaries, of form {"category1": {"word1": 3, ...}, ...}

        if save:        # optional, serialize dictionary to file

            self.final_dict["categoryPriorProbs"] = self.categoryPriorProbs        # save prior probabilities
            self.final_dict["categorySizes"] = self.categorySizes
            with open(self.filename + '.pickle', 'wb') as handle:
                pickle.dump(self.final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Serialized dictionary to file")

    def argmax(self, dct):
        max_prob = 0.0
        k_to_return = ""
        for k, v in dct.items():
            if v > max_prob:
                k_to_return = k
                max_prob = v

        return k_to_return

    def convert_to_probability(self, wrd_counts_dct):
        '''
        Converts each word's number of occurrences to its overall frequency in the category
        :param dictionary of <"string": int> pairs: E.g. {'word':5, 'automobile':7, 'dog':3}
        :return: new dictionary in the form of {'word':0.33, 'automobile':0.46, 'dog':0.2}
        '''

        numb_words = float(sum(wrd_counts_dct.values()))                 # total number of words in category

        if not(numb_words):
            return {}

        word_frequencies = dict()

        if self.method == "raw":                                         # calculate word frequencies
            for word, freq in wrd_counts_dct.items():
                word_frequencies[word] = freq / numb_words

        elif self.method == "mest":                                      # use m-estimate for tfidf as well
            # Formula:  P(wk | vj) = (nk + 1) / (n + |Vocab|)
            for wrd, frq in wrd_counts_dct.items():
                word_frequencies[wrd] = (frq + 1.0) / (numb_words + self.uniqueVocSize)

        elif self.method == "tfidf":
            # tf =  P(wk | vj) = (nk + 1) / (n + |Vocab|),  idf = log(total # categories / categories with wk)
            for wrd, frq in wrd_counts_dct.items():

                tf = (frq + 1.0) / (numb_words + self.uniqueVocSize)

                count = len([key for key in self.final_dict.keys() if wrd in self.final_dict.get(key)])

                if count == 0:              # if a word has never been seen, treat it as if it is completely average
                    count = self.numbCategories / 2    # (this gives slightly better results than ignoring it by making IDF = 1

                # Using a very tiny base for the logarithm made all the difference,
                # it went from 31% accuracy to 72%
                idf = math.log(self.numbCategories / (float(count)), 1.000000000000001)
                idf = 0.0 if idf < 0.0 else idf

                word_frequencies[wrd] = tf * idf

        return word_frequencies

    def guess_category(self, list_of_words, prob_dict):

        if not(len(self.categoryPriorProbs) and len(self.final_dict)):
            raise SystemExit("Classifier has not been trained yet")

        if not(len(list_of_words)):
            print("List of words or probability dictionary is empty")
            return ""

        assert len(prob_dict) > 1

        category_probs = {}                                   # calculate probability of each category being the answer

        for cat, dct in prob_dict.items():                    # e.g. "alt.atheism", {"word1":0.6, "word2":0.1,...}

            probability = self.categoryPriorProbs.get(cat)    # start out with prior probability of category

            for word in list_of_words:                        # multiply word probabilities

                word_prob = dct.get(word)                     # (Wk  |  Vj)

                if not word_prob:                             # handle the case of new unseen word

                    if self.method == "raw":
                        word_prob = 0.0
                    else:
                        # calculate m-estimate probability of new unseen word, and add it to dictionary for future check
                        n_words = float(self.categorySizes.get(cat))
                        word_prob = (0 + 1) / (n_words + self.uniqueVocSize)
                        dct[word] = word_prob

                probability *= word_prob

            category_probs[cat] = probability                 # assign calculated probability for each category

        k_to_return = self.argmax(category_probs)

        if not len(k_to_return):                              # if no guess calculated, make a random guess
            k_to_return = random.choice(self.categoryPriorProbs.keys())

        return k_to_return


def main():

    if len(sys.argv) != 4:
        print("Usage: {} trainfile testfile ['raw' | 'mest' | 'tfidf']".format(sys.argv[0]))
        sys.exit(-1)

    trainfile, testfile, method = sys.argv[1], sys.argv[2], sys.argv[3]

    """
    method = "tfidf"
    trainfile = "20ng-train-stemmed.txt"
    testfile = "20ng-test-stemmed.txt"
    """

    nbclassifier = NaiveBayes(method, trainfile, save=False)

    with open(testfile, "rb") as fd:
        document = fd.readlines()
        length = len(document)

    numb_wrong = 0
    numb_right = 0
    total_numb = 0

    probability_dict = dict()       # Construct dictionary of dictionaries, with word frequencies instead of occurrences
    for category, dct in nbclassifier.final_dict.items():
        probability_dict[category] = nbclassifier.convert_to_probability(dct)

    category_stats = {key: 0 for key in probability_dict.keys()}        # number guessed correctly per category
    category_lines = {key: 0 for key in probability_dict.keys()}        # total number of tests per category

    for i, line in enumerate(document):

        data = line.split()
        answer, words_to_classify = data[0], data[1:]

        guess = nbclassifier.guess_category(words_to_classify, probability_dict)

        if len(guess) > 0:
            if answer == guess:                                         # if guessed correctly
                numb_right += 1
                category_stats[answer] = category_stats.get(answer) + 1
            else:
                numb_wrong += 1

        category_lines[answer] = category_lines.get(answer) + 1

        total_numb += 1

        if not (i % 1000):  # Print update every 1000 lines
            print("progress: {} of {} lines classified".format(i, length))
    print("progress: {} of {} lines classified".format(i+1, length))

    nbclassifier.printTest(category_stats, category_lines)

    print("-"*102)
    print("Number correct: {}\nNumber incorrect: {}"
          "\nTotal trials: {}\nFinal accuracy: {:.3f}".format(numb_right, numb_wrong, total_numb,
                                                              float(numb_right) / total_numb * 100)
          )

if __name__ == "__main__":
    commencement = time.time()
    main()
    print("Total elapsed run time: {}m {}s".format(int((time.time()-commencement) / 60), (time.time()-commencement)%60))



