import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import os
import pandas as pd
import csv
import sys

class CorpusIterator:
    def __init__(self, folder_path, bigram=None, trigram=None):
        self.folder_path = folder_path
        self.bigram = bigram
        self.trigram = trigram

    def __iter__(self):
        Speeches = namedtuple('speeches', 'words tags')
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f, delimiter=',')
                        for row in reader:
                            text = row['speechtext_preprocessed']
                            congress = str(row['speechdate'].split('-')[0])
                            party = row['speakerparty']
                            partytag = party + '_' + congress
                            congresstag = 'CONGRESS_' + congress
                            tokens = text.split()
                            if self.bigram and self.trigram:
                                words = self.trigram[self.bigram[tokens]]
                            elif self.bigram and not self.trigram:
                                words = self.bigram[tokens]
                            else:
                                words = tokens
                            tags = [partytag, congresstag]
                            yield Speeches(words, tags)

class PhraseIterator:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __iter__(self):
        # Increase the field size limit- I was running into issues with csv exceeding the default limit in csv packaga
        csv.field_size_limit(sys.maxsize)
        
        # instead of just taking in one file with all the text, this will iterate through every folder in my folder path
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f, delimiter=',')
                        for row in reader:
                            text = row['speechtext_preprocessed']
                            yield text.split()


if __name__=='__main__':
    folder_path = '1_raw/canada/lipad'
    savepath = '2_build/models/canada/updated_model'

    iterator = PhraseIterator(folder_path)

    phrases = Phrases(iterator)
    bigram = Phraser(phrases)
    tphrases = Phrases(bigram[iterator])
    trigram = Phraser(tphrases)

    model0 = Doc2Vec(vector_size=200, window=20, min_count=50, workers=8, epochs=5)
    model0.build_vocab(CorpusIterator(folder_path, bigram=bigram, trigram=trigram))
    model0.train(CorpusIterator(folder_path, bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)
    model0.save(savepath)
