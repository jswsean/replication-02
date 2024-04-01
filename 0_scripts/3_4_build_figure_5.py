import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\lib')
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import labels
import plots 
import re

#=========================================================================================#
# Figure 5: Ideological Placement of Senators (114th Congress)
#=========================================================================================#
# Loading model with legislator level embeddings
senator_model = Doc2Vec.load('2_build/models/usa/senate114')
dems = [k for k in senator_model.dv.index_to_key if "_D" in k]
reps = [k for k in senator_model.dv.index_to_key if "_R" in k]
sm_parties = dems + reps
nsm = len(sm_parties)
zsm = np.zeros((nsm, 200))
for i in range(nsm):
    zsm[i,:] = senator_model.dv[sm_parties[i]]
pca_sm = PCA(n_components=2)
Zsm = pd.DataFrame(pca_sm.fit_transform(zsm), columns = ['dim1', 'dim2'])

# Figure 5
plots.plot_5(Zsm, dems, reps, savepath='3_docs/figures/figure5.pdf')
print('Saved Figure 5 to file 3_docs/figures/figure5.pdf')
