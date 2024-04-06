'''
This file reproduces figures 3a
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import lib.labels as labels
import lib.plots as plots

#=========================================================================================#
# Figure 3a: Party Placement in the UK (1935-2014)
#=========================================================================================#

ukmodel = Doc2Vec.load("2_build/models/uk_model")
uk_dict = labels.party_labels('UK')
uknames, ukparties, ukcols, ukmkers = labels.party_tags(ukmodel, 'UK')
uklabs = [uk_dict[p] for p in ukparties]
Muk = ukmodel.vector_size; Puk = len(ukparties)

# Fitting PCA dimensionality reduction model for plotting:
zuk = np.zeros((Puk,Muk))
for i in range(Puk):
    zuk[i,:] = ukmodel.docvecs[ukparties[i]]
pca_uk = PCA(n_components = 2)
Zuk = pd.DataFrame(pca_uk.fit_transform(zuk), columns = ['dim1', 'dim2'])
Zuk['label'] = uklabs

# Re-orienting the first axis for substantive interpretation:
if Zuk[Zuk.label=='Labour 2010'].dim1.values[0] > Zuk[Zuk.label=='Cons 2010'].dim1.values[0]:
    Zuk['dim1'] = Zuk.dim1 * (-1)

####### Figure 3a ########
plots.plot_3a(Zuk, uklabs, ukcols, ukmkers, savepath='3_docs/figures/figure3a.pdf')
# print("Saved Figure 3a to file figures/figure3a.pdf")
