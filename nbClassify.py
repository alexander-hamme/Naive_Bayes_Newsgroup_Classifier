'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Your name here

Date: XX.Oct.2017

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
        self.categories = dict()
        self.dict_count = dict()
        self.filename = "word_dict"
        if train:
            self.learn(train)               # loads train data, fills prob. table
        self.vocSize = len(set(self.vocab))  # unique words in vocab

    def printClasses(self):
        print(self.categories)

    def learn(self, traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''

        t = time.time()

        with open(traindat,'rb') as fd:

            document = fd.readlines()
            length = len(document)

            for i, line in enumerate(document):

                # id, *word = line.split()
                data = line.split()

                id, words = data[0], data[1:]

                if id in self.categories.keys():
                    self.categories[id].extend(words)
                else:
                    self.categories[id] = words
                # if id in self.categories.keys():
                #     for word in words:
                #         if word in self.categories[id].itervalues():
                #             self.categories[id][word]

                self.vocab.extend(words)

                # print(id, words)
                # TODO

                if not(i % 1000):          # Print update every 1000 lines
                    print("Progress: {} of {} lines processed".format(i+1, length))

        print("Processed {} of {} lines".format(i+1, length))


        for i, cat in enumerate(self.categories.iterkeys()):       # Python 3:  .items()
            all_words = self.categories.get(cat)
            assert isinstance(all_words, list)
            unique_words = set(all_words)

            counts = {}

            for wrd in unique_words:
                counts[wrd] = all_words.count(wrd)

            print("Progress: {} of 20 labels processed".format(i+1))

            self.dict_count[cat] = counts

        print(self.dict_count)
        print(len(self.dict_count))
        print("Total elapsed time: {}m {}s".format(int((time.time()-t)/60), (time.time()-t) % 60))

        with open(self.filename + '.pickle', 'wb') as handle:
            pickle.dump(self.dict_count, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def convert_to_probability(self, word_dict):
        '''

        :param word_dict: E.g. {'word':5, 'automobile':7, 'dog':3}
        :return: new dictionary in the form of {'word':0.33, 'automobile':0.46, 'dog':0.2}
        '''
        assert isinstance(word_dict, dict)
        numb_words = float(sum(word_dict.itervalues()))
        print("number of words = ", numb_words)
        prob_dict = dict()

        for key, val in word_dict.iteritems():
            prob_dict[key] = val / numb_words

        return prob_dict
        # for word in word_dict:
        #     prob_dict[str(word.keys())] = map(lambda v: v/numb_words, word.values())
        # return prob_dict

    """ GUESS category based on series of words:

	category_votes = {label: 0 for label in self.categories}
	
	
	for word in words:
		most_prob_cat = ""
		
		max_inversed_prob = 0
		
		for cat in prob_dict
			inversed_prob = 1.0 - prob_dict.get(cat).get(word) 
			if (inversed_prob) > max_inversed_prob:
				most_prob_cat = cat
				max_inversed_prob = inversed_prob
			elif (1 - prob_dict.get(cat).get(word)) == max_inversed_prob:
				most_prob_cat = cat	# add a vote to this category too, because it's equally likely
		
		category_votes[most_prob_cat] = category_votes.get(most_prob_cat) + 1

	return self.convert_to_probability(category_votes)
	
    """


def argmax(lst):
    return lst.index(max(lst))
    
def main():

    filename = "word_dict"

    if not os.path.exists(filename + '.pickle'):
        nbclassifier = NaiveBayes("20ng-test-stemmed.txt")
    else:
        nbclassifier = NaiveBayes()

        with open(filename + '.pickle', 'rb') as handle:
            nbclassifier.dict_count = pickle.load(handle)

        print("loaded pickle file")

    for category in nbclassifier.dict_count:
        print("Probability dict for {} is {}".format(
            category, nbclassifier.convert_to_probability(nbclassifier.dict_count.get(category)))
        )

    # for val in nbclassifier.dict_count.values():
    #     blah

    # nbclassifier.printClasses()

    raise SystemExit()


    if len(sys.argv) != 3:
        print("Usage: %s trainfile testfile" % sys.argv[0])
        sys.exit(-1)

    nbclassifier = NaiveBayes(sys.argv[1])
    nbclassifier.printClasses()
    nbclassifier.runTest(sys.argv[2])

if __name__ == "__main__":
    main()


    
