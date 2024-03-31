import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import sys
# sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\lib')
sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\original_lib')
import labels 
import plots 
import interpret


#=========================================================================================#
# Pre-plot processing
#=========================================================================================#

# Loading pretrained model and label names:
# model = Doc2Vec.load('2_build/models/usa/house')
model = Doc2Vec.load('C:\\Users\\Sean Hambali\\Desktop\\DATA\\dataverse_files\\dataverse_files\\models\\house200')
label_dict = labels.party_labels('USA')
fullnames, parties, cols, mkers = labels.party_tags(model, 'USA')
labs = [label_dict[p] for p in parties]
M = model.vector_size; P = len(parties)

# Fitting PCA dimensionality reduction model for plotting:
z = np.zeros((P,M))
for i in range(P):
    z[i,:] = model.docvecs[parties[i]]
pca = PCA(n_components = 2)
Z = pd.DataFrame(pca.fit_transform(z), columns = ['dim1', 'dim2'])
Z['label'] = labs

# Re-orienting the axes for substantive interpretation:
rev1 = False; rev2 = False;
if Z[Z.label=='Dem 2015'].dim1.values[0] > Z[Z.label=='Rep 2015'].dim1.values[0]:
    Z['dim1'] = Z.dim1 * (-1)
    rev1 = True
if Z[Z.label=='Dem 2015'].dim2.values[0] < Z[Z.label=='Rep 2015'].dim2.values[0]:
    Z['dim2'] = Z.dim2 * (-1)
    rev2 = True


#=========================================================================================#
# Table 1: Interpreting PCA Axes
#=========================================================================================#
interpret.Interpret(model, parties, pca, Z, labs, rev1=rev1, rev2=rev2, min_count=100, max_count = 1000000, max_features = 50000).top_words_list(20)
print("Saved Table 1 to file 3_docs/tables/table1.txt")


