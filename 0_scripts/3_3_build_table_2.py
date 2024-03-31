import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\lib')
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import labels
import plots 
import interpret
sys.path.append('C:\\Users\\Public\\Anaconda3\\lib\\site-packages')
import accuracy
import tabulate


#=========================================================================================#
# Table 2: Accuracy of Party Placement against Gold Standards
#=========================================================================================#
# Collect results
# 1. USA
model = Doc2Vec.load('2_build/models/usa/house')
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

# Add senate model, and fit PCA dimensionality reduction model for Senate:
senmodel = Doc2Vec.load('2_build/models/usa/senate')
zsen = np.zeros((P,M))
for i in range(P):
    zsen[i,:] = senmodel.dv[parties[i]]
pca_sen = PCA(n_components = 2)
Zsen = pd.DataFrame(pca_sen.fit_transform(zsen), columns = ['dim1', 'dim2'])
Zsen['label'] = labs

# Joining projections with external gold standards
gold_house = pd.read_csv('1_raw/usa/goldstandard_house.csv').merge(Z, on='label', how='left')
gold_senate = pd.read_csv('1_raw/usa/goldstandard_senate.csv').merge(Zsen, on='label', how='left')

gold_scores = ['voteview', 'experts_stand', 'rile', 'vanilla', 'legacy']

# Temporarily only have 2 tuples, but should have 4, including can and UK
countries = [('US House', gold_house),
             ('US Senate', gold_senate)]
results = np.zeros(( 10, 2 ), dtype=object)

for idx, (c, df) in enumerate(countries):
    jdx = 0
    for g in gold_scores:
        if g=='voteview' and 'voteview' not in df.columns:
            results[jdx:(jdx+2),idx] = ['','']
        else:
            temp = df[pd.notnull(df[g])]
            corr = '%0.3f' %temp.dim1.corr(temp[g])
            acc = '%0.2f%%' %accuracy.pairwise_accuracy(temp[g].tolist(), temp.dim1.tolist())
            results[jdx:(jdx+2),idx] = [corr, acc]
        jdx += 2

results = pd.DataFrame(results, columns = [c for c,df in countries])
results.insert(loc=0,column='Metric',value=['Correlation', 'Accuracy']*5)
results.insert(loc=0,column='Gold Standard',value=[item for item in ['Voteview', 'Experts Surveys', 'rile', 'vanilla', 'legacy'] for i in range(2)])

with open('3_docs/tables/table2.txt', 'w') as f:
    print("Table 2: Accuracy of Party Placement against Gold Standards\n"+"-"*83, file=f)
    print(tabulate.tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl"), file=f)
print("Saved Table 2 to file tables/table2.txt")