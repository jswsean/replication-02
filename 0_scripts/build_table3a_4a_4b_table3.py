'''
This file reproduces figures 3a, 4a, 4b and table 2

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import utils.labels as labels
import utils.plots as plots
from utils.interpret import Interpret
from utils.accuracy import pairwise_accuracy
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print("Changed directory to:", os.getcwd())
#=========================================================================================#
# Figure 2: Party Placement in the US House (1873-2016)
#=========================================================================================#
#Loading pretrained model and label names:
model = Doc2Vec.load('models/house')
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
# Figure 2: Party Placement in the UK (1935-2014)
#=========================================================================================#
ukmodel = Doc2Vec.load("models/uk_model")
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
plots.plot_3a(Zuk, uklabs, ukcols, ukmkers, savepath='newfigs/figure3a.pdf')
# print("Saved Figure 3a to file figures/figure3a.pdf")


### Joining projections with external gold standards
gold_house = pd.read_csv('data/goldstandard_house.csv').merge(Z, on='label', how='left')
gold_uk = pd.read_csv('data/goldstandard_uk.csv').merge(Zuk, on='label', how='left')

# #=========================================================================================#
# # Figure 4: Comparison with WordFish Estimates
# #=========================================================================================#
# # Loading fitted WordFish models
wf_full = pd.read_excel("data/house_document_scores_pivoted.xlsx")
wf_short = pd.read_excel("data/house_document_scores_mostrecentpivoted.xlsx")

pfull = wf_full[['year','score_R','score_D']]
pfull.rename(columns = {'score_R':'republican', 'score_D':'democrat'}, inplace = True)
pfull = pfull.sort_values(by='year')

pfull_anc = wf_full[['year','score2_R','score2_D']]
pfull_anc.rename(columns = {'score2_R':'republican', 'score2_D':'democrat'}, inplace = True)
pfull_anc = pfull_anc.sort_values(by='year')

pshort = wf_short[['year','score_R','score_D']]
pshort.rename(columns = {'score_R':'republican', 'score_D':'democrat'}, inplace = True)
pshort = pshort.sort_values(by='year')

pshort_anc = wf_short[['year','score2_R','score2_D']]
pshort_anc.rename(columns = {'score2_R':'republican', 'score2_D':'democrat'}, inplace = True)
pshort_anc = pshort_anc.sort_values(by='year')

# Figure 4a
plots.plot_4a(pfull, savepath='newfigs/figure4a_full.pdf')
print("Saved Figure 4a to file figures/figure4a.pdf")

plots.plot_4a(pfull_anc, savepath='newfigs/figure4a_full_anc.pdf')
print("Saved Figure 4a to file figures/figure4a.pdf")

# Figure 4b
plots.plot_4b(pshort, savepath='newfigs/figure4b_short.pdf')
print("Saved Figure 4b to file figures/figure4b.pdf")

plots.plot_4b(pshort_anc, savepath='newfigs/figure4b_short_anc.pdf')
print("Saved Figure 4b to file figures/figure4b.pdf")

# #=========================================================================================#
# # Table 3: Accuracy of Party Placement in the US House: WordFish and Party Embeddings
# #=========================================================================================#

##anc subscript wordfish estimates with anchors 

wf_short= pd.read_excel("data/house_document_scores_mostrecent.xlsx")
wf_short.drop(columns = 'text', inplace = True)
wf_full= pd.read_excel("data/house_document_scores.xlsx")
wf_full.drop(columns = 'text', inplace = True)

wf_full['party'] = np.where(wf_full['party'] == 'D', 'Dem', 'Rep')
wf_full['label'] = wf_full['party'] + ' '+ wf_full['year'].astype(str) 
wf_full.drop(columns = ['session', 'party', 'year'], inplace = True)

wf_short['party'] = np.where(wf_short['party'] == 'D', 'Dem', 'Rep')
wf_short['label'] = wf_short['party'] + ' '+ wf_short['year'].astype(str) 
wf_short.drop(columns = ['session', 'party', 'year'], inplace = True)
wf_full.head()

wf_full.rename(columns = {'score':'wf_full_anc', 'score2':'wf_full'}, inplace = True)
wf_short.rename(columns = {'score':'wf_short_anc', 'score2':'wf_short'}, inplace = True)

gold_house = gold_house.merge(wf_full, on='label', how='left')
gold_house = gold_house.merge(wf_short, on='label', how='left')

results = np.zeros(( 3, 4 ), dtype=object)
for idx, placement in enumerate(['wf_full_anc', 'dim1']):
    results[1:3,idx] = ['%0.3f' %gold_house[placement].corr(gold_house.voteview),
                       '%0.2f%%' %pairwise_accuracy(gold_house.voteview.tolist(), gold_house[placement].tolist())]
temp = gold_house[pd.notnull(gold_house.wf_short)]
for idx, placement in enumerate(['wf_short_anc', 'dim1']):
    results[1:3,idx+2] = ['%0.3f' %temp[placement].corr(temp.voteview),
                       '%0.2f%%' %pairwise_accuracy(temp.voteview.tolist(), temp[placement].tolist())]
results[0,:] = ['WordFish', 'Embeddings']*2
results = pd.DataFrame(results, columns=['1921-2016', '1921-2016', '2007-2016','2007-2016'])
results.insert(loc=0, column='Metric', value=['','Correlation','Pairwise Accuracy'])

with open('newtable/table3_anc.txt', 'w') as f:
    print("Table 3: Accuracy of Party Placement in the US House: WordFish and Party Embeddings\n"+"-"*76, file=f)
    print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl"), file=f)
print("Saved Table 3 to file newtable/table3_anc.txt")

