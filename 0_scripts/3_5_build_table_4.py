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
sys.path.append('C:\\Users\\Public\\Anaconda3\\lib\\site-packages')
import accuracy
import tabulate

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

# Construct dataframe to merge senator ID with senator embeddings
speakerid = [int(re.match(r"(\d+)", k).group(1)) for k in sm_parties]
Zsm = pd.DataFrame({
    'speakerid': speakerid,
    'dim1': Zsm.dim1, 
    'dim2': Zsm.dim2
})

# Loading Senator-level gold standards for congress 114
gold_senators = pd.read_csv('1_raw/usa/goldstandard_senate114.csv').merge(
    Zsm, how = "left", on = "speakerid"
)

# Flip PCA axes from dim 1
if np.mean(gold_senators[gold_senators.party == "D"].dim1) > np.mean(gold_senators[gold_senators.party == "R"].dim1):
    gold_senators['dim1'] = gold_senators['dim1'] * (-1)

# Compute pairwise accuracy and correlation
results = np.zeros(( 6, 2 ), dtype=object)
gold_scores = ['nominate_dim1', 'nokken_poole_dim1', 'ACU_2016', 'ACU_2015', 'ACU_Life', 'govtrack']
for idx, g in enumerate(gold_scores):
    corr = '%0.3f' %gold_senators.dim1.corr(gold_senators[g])
    acc = '%0.2f%%' %accuracy.pairwise_accuracy(gold_senators[g].tolist(), gold_senators.dim1.tolist())
    results[idx,:] = [corr, acc]

results = pd.DataFrame(results, columns=['Correlation','Pairwise Accuracy'])
results.insert(loc=0, column='Gold Standard', value=['DW-NOMINATE',
                                                     'Nokken-Poole',
                                                     'ACU 2016',
                                                     'ACU 2015',
                                                     'ACU Life',
                                                     'GovTrack'])

with open('3_docs/tables/table4.txt', 'w') as f:
    print("Table 4: Accuracy of Senator Ideological Placement", file=f)
    print("-"*57, file=f)
    print(tabulate.tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl"), file=f)
print("Saved Table 4 to file 3_docs/tables/table4.txt")