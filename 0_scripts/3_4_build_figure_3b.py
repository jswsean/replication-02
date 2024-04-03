import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import os
import re

import sys
sys.path.append('utils')
import labels
import plots 
import accuracy
from accuracy import pairwise_accuracy


# ============================================================================================= #
# Build Figure 3b: Placement of Canadian political parties
# ============================================================================================= #

# load in Canada model
canmodel = Doc2Vec.load('2_build/models/canada/updated_model')

can_dict = labels.party_labels('Canada')
cannames, canparties, cancols, canmkers = labels.party_tags(canmodel, 'Canada')

# preprocess the keys in canparties to have it match the inverted can_dict
#canparties_processed = [party.replace('_', ' ') for party in canparties]

# invert the dictionary to map from values to keys
can_dict_inverted = {value: key for key, value in can_dict.items()}

# changing this so that it only looks at ones that are actually in the dictionary
canlabs = [can_dict_inverted[p] for p in canparties if p in can_dict_inverted]
Mcan = canmodel.vector_size
#Pcan = len(canparties_processed)
Pcan = len(canparties)

zcan = np.zeros((Pcan, Mcan))
for i in range(Pcan):
    zcan[i,:] = canmodel.docvecs[canparties[i]]
pca_can = PCA(n_components = 2)
Zcan = pd.DataFrame(pca_can.fit_transform(zcan), columns = ['dim1', 'dim2'])
Zcan['label'] = canparties

# change all instances of 'New democratic party' to NDP
Zcan['label'] = Zcan['label'].apply(lambda x: re.sub(r'New Democratic Party', 'NDP', x))
Zcan['label'] = Zcan['label'].apply(lambda x: re.sub(r'Conservative', 'Cons', x))
Zcan['label'] = Zcan['label'].apply(lambda x: re.sub(r'Bloc Québécois', 'Bloc', x))
# had to do my own research to find out that reform party/canadian alliance merged with the progressive conservative party
Zcan['label'] = Zcan['label'].apply(lambda x: re.sub(r'Progressive Cons', 'RefAll', x))

# Remove underscores from labels
Zcan['label'] = Zcan['label'].str.replace('_', ' ')

# Re-orienting the first axis for substantive interpretation:
if Zcan[Zcan.label=='NDP_2015'].dim1.values[0] > Zcan[Zcan.label=='Cons_2015'].dim1.values[0]:
    Zcan['dim1'] = Zcan.dim1 * (-1)

# Export plots
plots.plot_3b(Zcan, canlabs, cancols, canmkers, savepath='3_docs/figures/figure3b.pdf')
print("Saved Figure 3b to file 3_docs/figures/figure3b.pdf")


