import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
assert gensim.models.doc2vec.FAST_VERSION > -1

''' Script to train word embedding model for UK corpus'''

df  = pd.read_hdf('data/processed_uk.h5','s')    

class corpusIterator:
    def __init__(self, df, bigram=None, trigram=None):
        self.df = df
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        for row in self.df.iterrows():
            text = row[1]['cleaned_text'].replace('\n','')
            congress = str(row[1]['parliament'])
            party = row[1]['party']
            partytag = party + '_' + congress
            congresstag = 'CONGRESS_' + congress
            tokens = text.split()
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [partytag, congresstag]
            yield self.speeches(self.words, self.tags)
class phraseIterator():
    def __init__(self, df):
        self.df = df
    def __iter__(self):
        for row in self.df.iterrows():
            text = row[1]['cleaned_text'].replace('\n','')
            yield text.split()


phrases = Phrases(phraseIterator(df))
bigram = Phraser(phrases)
tphrases = Phrases(bigram[phraseIterator(df)])
trigram = Phraser(tphrases)

# To save phraser objects for future usage.
# bigram.save('outputs/phraser_bigrams')
# trigram.save('outputs/phraser_trigrams')

model0 = Doc2Vec(vector_size=200, window=20, min_count=50, workers=8, epochs=5)
model0.build_vocab(corpusIterator(df, bigram=bigram, trigram=trigram))
model0.train(corpusIterator(df, bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)
model0.save('outputs/uk_model')