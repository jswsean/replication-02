import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

class Interpret(object):

    def __init__(self, model, parties, dr, Z, labels, rev1=False, rev2=False, min_count=100, max_count = 1000000, max_features=10000):
        self.model = model
        self.parties = parties
        self.labels = labels
        self.P = len(self.parties)
        self.M = self.model.vector_size
        self.voc = self.sorted_vocab(min_count, max_count, max_features)
        self.V = len(self.voc)
        self.pca = dr
        self.max = Z.max(axis=0)
        self.min = Z.min(axis=0)
        self.sims = self.compute_sims()
        self.dim1 = rev1
        self.dim2 = rev2

    def sorted_vocab(self, min_count=100, max_count=10000, max_features=10000):
        wordlist=[]
        for word in self.model.wv.key_to_index:
            count = self.model.wv.get_vecattr(word, "count")
            wordlist.append((word, count))
        wordlist = sorted(wordlist, key=lambda tup: tup[1], reverse=True)
        return [w for w,c in wordlist if c>min_count and c<max_count and w.count('_')<3][0:max_features]

    def compute_sims(self):

        S = np.zeros((self.V, 2))
        for idx, w in enumerate(self.voc):
            S[idx, :] = self.pca.transform(self.model.wv[w].reshape(1,-1))
        sims_right = euclidean_distances(S, np.array([self.max[0],0]).reshape(1, -1))
        sims_left = euclidean_distances(S, np.array([self.min[0],0]).reshape(1, -1))
        sims_up = euclidean_distances(S, np.array([0,self.max[1]]).reshape(1, -1))
        sims_down = euclidean_distances(S, np.array([0,self.min[1]]).reshape(1, -1))
        temp = pd.DataFrame({'word': self.voc, 'right': sims_right[:,0], 'left': sims_left[:,0], 'up': sims_up[:,0], 'down': sims_down[:,0]})
        return temp

    def top_words_list(self, topn=20, savepath='3_docs/tables/table1.txt'):

        with open(savepath, 'w') as f:
            print("Table 1: Interpreting PCA Axes", file=f)
            if self.dim1:
                ordering = ['left','right']
            else:
                ordering = ['right', 'left']
            temp = self.sims.sort_values(by=ordering[0])
            print(80*"-", file=f)
            print("Words Associated with Positive Values (Right) on First Component:", file=f)
            print(80*"-", file=f)
            self.top_positive_dim1 = temp.word.tolist()[0:topn]
            self.top_positive_dim1 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim1])
            print(self.top_positive_dim1, file=f)
            temp = self.sims.sort_values(by=ordering[1])
            print(80*"-", file=f)
            print("Words Associated with Negative Values (Left) on First Component:", file=f)
            print(80*"-", file=f)
            self.top_negative_dim1 = temp.word.tolist()[0:topn]
            self.top_negative_dim1 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim1])
            print(self.top_negative_dim1, file=f)

            if self.dim2:
                ordering = ['down','up']
            else:
                ordering = ['up', 'down']
            temp = self.sims.sort_values(by=ordering[0])
            print(80*"-", file=f)
            print("Words Associated with Positive Values (North) on Second Component:", file=f)
            print(80*"-", file=f)
            self.top_positive_dim2 = temp.word.tolist()[0:topn]
            self.top_positive_dim2 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim2])
            print(self.top_positive_dim2, file=f)
            temp = self.sims.sort_values(by=ordering[1])
            print(80*"-", file=f)
            print("Words Associated with Negative Values (South) on Second Component:", file=f)
            print(80*"-", file=f)
            self.top_negative_dim2 = temp.word.tolist()[0:topn]
            self.top_negative_dim2 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim2])
            print(self.top_negative_dim2, file=f)
            print(80*"-", file=f)
