import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Sean Hambali\\Documents\\GitHub\\PPOL6081\\replication-02\\0_scripts\\lib')
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import labels
sys.path.append('C:\\Users\\Public\\Anaconda3\\lib\\site-packages')
import accuracy
import tabulate
import re


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

# Load other country models: 
# 2. UK 
ukmodel = Doc2Vec.load('2_build/models/uk/uk_model')
uk_dict = labels.party_labels('UK')
uknames, ukparties, ukcols, ukmkers = labels.party_tags(ukmodel, 'UK')
uklabs = [uk_dict[p] for p in ukparties]
Muk = ukmodel.vector_size; Puk = len(ukparties)

# Fitting PCA dimensionality reduction model for plotting UK results:
zuk = np.zeros((Puk,Muk))
for i in range(Puk):
    zuk[i,:] = ukmodel.dv[ukparties[i]]
pca_uk = PCA(n_components = 2)
Zuk = pd.DataFrame(pca_uk.fit_transform(zuk), columns = ['dim1', 'dim2'])
Zuk['label'] = uklabs

# Re-orienting the first axis for substantive interpretation:
if Zuk[Zuk.label=='Labour 2010'].dim1.values[0] > Zuk[Zuk.label=='Cons 2010'].dim1.values[0]:
    Zuk['dim1'] = Zuk.dim1 * (-1)

# Load other country models:
# 3. CAN
# load in Canada model
canmodel = Doc2Vec.load('2_build/models/canada/updated_model')
can_dict = labels.party_labels('Canada')
cannames, canparties, cancols, canmkers = labels.party_tags(canmodel, 'Canada')

# invert the dictionary to map from values to keys
can_dict_inverted = {value: key for key, value in can_dict.items()}

# changing this so that it only looks at ones that are actually in the dictionary
canlabs = [can_dict_inverted[p] for p in canparties if p in can_dict_inverted]
Mcan = canmodel.vector_size
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

# remove underscores from labels
Zcan['label'] = Zcan['label'].str.replace('_', ' ')

# Re-orienting the first axis for substantive interpretation:
if Zcan[Zcan.label=='NDP 2015'].dim1.values[0] > Zcan[Zcan.label=='Cons 2015'].dim1.values[0]:
    Zcan['dim1'] = Zcan.dim1 * (-1)

# Joining projections with external gold standards
gold_house = pd.read_csv('1_raw/usa/goldstandard_house.csv').merge(Z, on='label', how='left')
gold_senate = pd.read_csv('1_raw/usa/goldstandard_senate.csv').merge(Zsen, on='label', how='left')
gold_uk = pd.read_csv('1_raw/uk/goldstandard_uk.csv').merge(Zuk, on='label', how='left')
gold_can = pd.read_csv('1_raw/canada/goldstandard_canada.csv').merge(Zcan, on='label', how='left')

gold_scores = ['voteview', 'experts_stand', 'rile', 'vanilla', 'legacy']

# Temporarily only have 2 tuples, but should have 4, including can and UK
countries = [('US House', gold_house),
             ('US Senate', gold_senate), 
             ('Britain', gold_uk), 
             ('Canada', gold_can)]
results = np.zeros(( 10, 4 ), dtype=object)

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
print("Saved Table 2 to file 3_docs/tables/table2.txt")