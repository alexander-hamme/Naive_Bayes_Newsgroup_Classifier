'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Alex Hamme

Date: 7.Nov.2017

'''
from __future__ import print_function
import pickle
import math
import time
import sys
import os


class NaiveBayes():
    '''Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''

    def __init__(self, train=None):
        '''Create classifier using train, the name of an input
        training file.
        '''
        self.vocab = []
        self.categories = dict()            # Dict of prior probabilities for each category
        self.final_dict = dict()
        self.filename = "word_dict"
        self.fullVocSize = 0
        self.uniqueVocSize = 0  # unique words in vocab
        if train:
            self.learn(train)               # loads train data, fills prob. table

    def load_pickle_file(self, filename):
        with open(filename + '.pickle', 'rb') as handle:
            self.final_dict = pickle.load(handle)

        self.categories = {key: 0.0 for key in self.final_dict.iterkeys()}

        print("loaded pickle file")
        print(self.categories)

    def printClasses(self):
        print(self.categories)


    def load_text_file(self, filename):

        categorySets = dict()
        vocab = []

        with open(filename,'rb') as fd:

            document = fd.readlines()
            length = len(document)

            for i, line in enumerate(document):

                # id, *word = line.split()
                data = line.split()

                id, words = data[0], data[1:]

                if id in categorySets.keys():
                    categorySets[id].extend(words)
                else:
                    categorySets[id] = words

                vocab.extend(words)

                if not(i % 1000):          # Print update every 1000 lines
                    print("Progress: {} of {} lines processed".format(i, length))

            print("Processed {} of {} lines".format(i + 1, length))

        self.fullVocSize += len(vocab)
        self.uniqueVocSize = len(set(vocab))


        print("{} total words categorized".format(self.fullVocSize))

        self.categories = {key: 0.0 for key in categorySets.iterkeys()}

        return categorySets, vocab


    def learn(self, traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''

        t = time.time()

        categorySets, vocab = self.load_text_file(traindat)

        assert len(categorySets) > 1

        for i, cat in enumerate(categorySets.iterkeys()):       # Python 3:  .items()

            all_words = categorySets.get(cat)

            self.categories[cat] = len(all_words) / float(self.fullVocSize)         # Calculate prior probability

            assert isinstance(all_words, list)

            unique_words = set(all_words)

            counts = {wrd: all_words.count(wrd) for wrd in unique_words}

            # for wrd in unique_words:
            #     counts[wrd] = all_words.count(wrd)

            print("Progress: {} of 20 labels processed".format(i+1))

            self.final_dict[cat] = counts

        print("Length of dict_count: ", len(self.final_dict))
        print("Total elapsed training time: {}m {}s".format(int((time.time()-t)/60), (time.time()-t) % 60))

        with open(self.filename + '.pickle', 'wb') as handle:
            pickle.dump(self.final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Serialized dictionary to file")

    def test(self, testdat):
        pass

    def convert_to_probability(self, word_dict):
        '''

        :param word_dict: E.g. {'word':5, 'automobile':7, 'dog':3}
        :return: new dictionary in the form of {'word':0.33, 'automobile':0.46, 'dog':0.2}
        '''
        assert isinstance(word_dict, dict)
        numb_words = float(sum(word_dict.itervalues()))

        if not(numb_words):
            return {}

        prob_dict = dict()

        for key, val in word_dict.iteritems():
            prob_dict[key] = val / numb_words

        return prob_dict
        # for word in word_dict:
        #     prob_dict[str(word.keys())] = map(lambda v: v/numb_words, word.values())
        # return prob_dict

    def guessCategory(self, list_of_words):

        if not(len(self.categories)) and not(len(self.final_dict)):
            raise SystemExit("Classifier has not been trained yet")
        elif not(len(list_of_words)):
            print("List of words is empty")
            return ""

        category_votes = {cat: 0 for cat in self.categories}

        probability_dict = dict()

        for category, dct in self.final_dict.iteritems():
            probability_dict[category] = self.convert_to_probability(dct)

        for word in list_of_words:
            most_likely_cat = ""
            max_prob = 0

            for cat, dct in probability_dict.iteritems():#self.final_dict.iteritems():

                probability = dct.get(word)

                if not(probability):    # Current newsgroup category does not contain current word
                    continue

                if (probability) > max_prob:
                    most_likely_cat = cat
                    max_prob = probability

                elif probability == max_prob:
                    most_likely_cat = cat  # add a vote to this category too, because it's equally likely

            # Handle error!!!

            if not len(most_likely_cat):
                continue

            # print("Getting {} from dict".format(most_likely_cat))

            category_votes[most_likely_cat] = category_votes.get(most_likely_cat) + 1

        # self.convert_to_probability(category_votes)

        total_votes = float(sum(category_votes.itervalues()))

        for cat, votes in category_votes.iteritems():
            probability = self.categories.get(cat) * votes/total_votes


        if not category_votes:
            return ""

        # print("Category votes = ", category_votes)

        max_votes = 0
        k_to_return = ""
        for k, v in category_votes.items():
            if v > max_votes:
                k_to_return = k
                max_votes = v

        return k_to_return


def argmax(lst):
    return lst.index(max(lst))

def main():

    verbose = True

    """

    IMPORTANT:

    Prior probability of each category is number of words in it DIVIDED by total number of words

    :return:
    """

    filename = "word_dict"

    if not os.path.exists(filename + '.pickle'):
        nbclassifier = NaiveBayes("20ng-train-stemmed.txt")
    else:
        nbclassifier = NaiveBayes()
        nbclassifier.load_pickle_file(filename)

    nbclassifier.printClasses()

    accuracyScores = []

    with open("20ng-test-stemmed.txt", "rb") as fd:

        document = fd.readlines()
        length = len(document)

    numb_wrong = 0
    numb_right = 0
    total_numb = 0

    start = time.time()

    for i, line in enumerate(document):
        # id, *word = line.split()
        data = line.split()
        answer, wordsToClassify = data[0], data[1:]

        guess = nbclassifier.guessCategory(wordsToClassify)

        if len(guess) > 0:
            if answer == guess:
                if verbose:
                    print("Guessed answer {} correctly".format(answer))
                accuracyScores.append(1)
                numb_right += 1
            else:
                if verbose:
                    print("Incorrectly guessed {}, correct answer was {}".format(guess, answer))
                accuracyScores.append(0)
                numb_wrong += 1

            total_numb += 1

        if not (i % 100):  # Print update every 100 lines
            print("Progress: {} of {} lines classified".format(i, length))

    print("Elapsed classification time: {}m {}s\nNumber correct: {}\nNumber incorrect: {}"
          "\nTotal trials: {}\nFinal accuracy: {}".format(
                                                          int((time.time() - start) / 60), (time.time() - start) % 60,
                                                          numb_right, numb_wrong, total_numb,
                                                          float(sum(accuracyScores)) / len(accuracyScores)
                                                          )
          )

    raise SystemExit()


    if len(sys.argv) != 3:
        print("Usage: %s trainfile testfile" % sys.argv[0])
        sys.exit(-1)

    nbclassifier = NaiveBayes(sys.argv[1])
    nbclassifier.printClasses()
    nbclassifier.runTest(sys.argv[2])

if __name__ == "__main__":
    main()



