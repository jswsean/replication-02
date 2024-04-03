'''
This file reproduces Table 3
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

# #=========================================================================================#
# # Table 3: Accuracy of Party Placement in the US House: WordFish and Party Embeddings
# #=========================================================================================#

##anc subscript wordfish estimates with anchors 
wf_short= pd.read_excel("2_build/data/house_document_scores_mostrecent.xlsx")
wf_short.drop(columns = 'text', inplace = True)
wf_full= pd.read_excel("2_build/data/house_document_scores.xlsx")
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

### Joining projections with external gold standards
gold_house = pd.read_csv('1_raw/usa/goldstandard_house.csv')
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

with open('3_docs/tables/table3_anc.txt', 'w') as f:
    print("Table 3: Accuracy of Party Placement in the US House: WordFish and Party Embeddings\n"+"-"*76, file=f)
    print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl"), file=f)
print("Saved Table 3 to file 3_docs/tables/table3_anc.txt")
