import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import sys
sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\lib')
import labels 
import plots 

# import utils.labels as labels
# import utils.plots as plots
# from utils.interpret import Interpret
# from utils.accuracy import pairwise_accuracy

#=========================================================================================#
# Figure 2: Party Placement in the US House (1873-2016)
#=========================================================================================#
# Loading pretrained model and label names:
model = Doc2Vec.load('2_build/models/usa/house')
# model = Doc2Vec.load('C:\\Users\\Sean Hambali\\Desktop\\DATA\\dataverse_files\\dataverse_files\\models\\house200')
label_dict = labels.party_labels('USA')
fullnames, parties, cols, mkers = labels.party_tags(model, 'USA')
labs = [label_dict[p] for p in parties]
M = model.vector_size; P = len(parties)

# Fitting PCA dimensionality reduction model for plotting:
z = np.zeros((P,M))
for i in range(P):
    z[i,:] = model.dv[parties[i]]
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

# Reproducing Figure 2
# 2a
plots.plot_2a(Z, labs, cols, mkers, savepath='3_docs/figures/figure2a.pdf')
print("Saved Figure 2a to file figures/figure2a.pdf")
# 2b
plots.plot_timeseries(Z, fullnames, cols, dimension=1, savepath='3_docs/figures/figure2b.pdf', legend='upper left')
print("Saved Figure 2b to file figures/figure2b.pdf")
# 2c
plots.plot_timeseries(Z, fullnames, cols, dimension=2, savepath='3_docs/figures/figure2c.pdf', legend='lower left')
print("Saved Figure 2c to file figures/figure2c.pdf")