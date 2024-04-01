#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=====================================================================#
#
# Description:
# An example sript to fit word embeddings with political metadata.
# For more information, see www.github.com/lrheault/partyembed
#
# Usage:
# python3 partyembeddings_senate.py
#
# @author: L. Rheault
#
#=====================================================================#

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
assert gensim.models.doc2vec.FAST_VERSION > -1

class corpusIterator(object):

    def __init__(self, inpath, house, bigram=None, trigram=None):
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
        self.house = house
        self.inpath = inpath

    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        with open(self.inpath, 'r') as f:
            for line in f:
                ls = line.split('\t')
                congress = str(ls[0])
                chamber = ls[5]
                if congress == "114" and chamber == self.house: 
                    text = ls[10].replace('\n','')
                    party = ls[7]
                    senator = ls[3]
                    senator_party = senator + "_" + party
                    tokens = text.split()
                    if self.bigram and self.trigram:
                        self.words = self.trigram[self.bigram[tokens]]
                    elif self.bigram and not self.trigram:
                        self.words = self.bigram[tokens]
                    else:
                        self.words = tokens
                    self.tags = [senator_party]
                    yield self.speeches(self.words, self.tags)

class phraseIterator(object):

    def __init__(self, inpath, house):
        self.inpath = inpath
        self.house = house

    def __iter__(self):
        with open(self.inpath, 'r') as f:
            for line in f:
                ls = line.split('\t')
                congress = str(ls[0])
                chamber = ls[5]
                if congress == "114" and chamber == self.house: 
                    text = ls[10].replace('\n','')
                    yield text.split()

if __name__=='__main__':

    # Fill in the paths to desired location.
    # Corpus is expected to be in tab-separated format with column ordering specified in
    # reformat_congress.py, and clean text in column #10.

    inpath = '2_build/preprocessed_congress'
    savepath = '2_build/models/usa/'

    phrases = Phrases(phraseIterator(inpath, house='S'))
    bigram = Phraser(phrases)
    tphrases = Phrases(bigram[phraseIterator(inpath, house='S')])
    trigram = Phraser(tphrases)

    # To save phraser objects for future usage.
    # bigram.save('.../phraser_bigrams')
    # trigram.save('.../phraser_trigrams')

    model0 = Doc2Vec(vector_size=200, window=20, min_count=50, workers=8, epochs=20)
    model0.build_vocab(corpusIterator(inpath, house='S', bigram=bigram, trigram=trigram))
    model0.train(corpusIterator(inpath, house='S', bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)
    model0.save(savepath + 'senate114')
