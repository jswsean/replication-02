#!/usr/bin/env python
# coding: utf-8

# In[50]:


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
import interpret
import accuracy
from accuracy import pairwise_accuracy
from interpret import Interpret


# ## Figure 3- Party Placement in Canada

# In[67]:


# load in Canada model

#canmodel = Doc2Vec.load('model/updated_model')
canmodel = Doc2Vec.load('model/updated_model')

can_dict = labels.party_labels('Canada')
cannames, canparties, cancols, canmkers = labels.party_tags(canmodel, 'Canada')

# invert the dictionary to map from values to keys
#can_dict_inverted = {value: key for key, value in can_dict.items()}

# changing this so that it only looks at ones that are actually in the dictionary
#canlabs = [can_dict_inverted[p] for p in canparties if p in can_dict_inverted]

#Mcan = canmodel.vector_size
#Pcan = len(canparties)


# In[68]:


canmodel.docvecs.index_to_key


# In[69]:


# updating the canparties year variable to match the parliament year instead
# adjusting year range
year_ranges = {
    9.0: '1901-1903',
    10.0: '1904-1907',
    11.0: '1908-1910',
    12.0: '1911-1917',
    13.0: '1918-1921',
    14.0: '1922-1924',
    15.1: '1925',
    15.2: '1926',
    16.0: '1927-1929',
    17.0: '1930-1934',
    18.0: '1935-1939',
    19.0: '1940-1944',
    20.0: '1945-1948',
    21.0: '1949-1952',
    22.0: '1953-1956',
    23.0: '1957',
    24.0: '1958-1961',
    25.0: '1962',
    26.0: '1963-1964',
    27.0: '1965-1967',
    28.0: '1968-1971',
    29.0: '1972-1973',
    30.0: '1974-1978',
    31.0: '1979-1980',
    32.0: '1981-1983',
    33.0: '1984-1987',
    34.0: '1988-1992',
    35.0: '1993-1996',
    36.0: '1997-1999',
    37.0: '2000-2003',
    38.0: '2004-2005',
    39.0: '2006-2007',
    40.0: '2008-2010',
    41.0: '2011-2014',
    42.0: '2015-2019'
}

output = []

for value in canparties:
    # Extract the year from the value
    party, year = value.split('_')
    
    party = str(party)
    year = int(year)
    
    # Replace party names with abbreviations
    party = re.sub(r'New Democratic Party', 'NDP', party)
    party = re.sub(r'Conservative \(1867-1942\)', 'Conservative', party)
    party = re.sub(r'Bloc Québécois', 'Bloc', party)
    #party = re.sub(r'Progressive Conservative', 'Conservative', party)
    party = re.sub(r'Reform Party', 'RefAll', party)
    party = re.sub(r'Canadian Alliance', 'RefAll', party)
    
    # making sure that every party is lumped into these categories
    #party_keywords = {
    #'ndp': 'NDP',
    #'National Democratic Party':'NDP',
    #'bloc': 'Bloc',
    #'Liberal': 'Liberal',
    #'Cons': 'Conservative',
    #'reform': 'Reform-Alliance',
    #'Canadian Alliance': 'Reform-Alliance'       
#}
    for keyword, party_name in party_keywords.items():
        if keyword in party:
            party = party_name
            break

    # checking what session range the year falls under and converting it to the corresponding key
    key_of_range = None
    for key, year_range in year_ranges.items():
        if '-' in year_range:
            start, end = map(int, year_range.split('-'))
            if start <= year <= end:
                key_of_range = key
                break
        # if the session is just a single year
        else:
            single_year = int(year_range)
            if year == single_year:
                key_of_range = key
                break
    # sets the key equal to party_session            
    if key_of_range is not None:
        output.append(f'{party}_{key_of_range}')
        


# In[70]:


unique_output = set(output)
for value in unique_output:
    print(value)


# In[71]:


# GRACE - NEED TO TAKE OUT IF STATEMENT WITH NEW MODEL
canlabs = [can_dict[p] for p in output if p in can_dict]
Mcan = canmodel.vector_size; 
Pcan = len(canlabs)


# In[72]:


zcan = np.zeros((Pcan, Mcan))
for i in range(Pcan):
    zcan[i,:] = canmodel.docvecs[canparties[i]]
pca_can = PCA(n_components = 2)
Zcan = pd.DataFrame(pca_can.fit_transform(zcan), columns = ['dim1', 'dim2'])

Zcan['label'] = canlabs


# In[73]:


Zcan


# In[56]:


# Re-orienting the first axis for substantive interpretation:
if Zcan[Zcan.label=='NDP 2015'].dim1.values[0] > Zcan[Zcan.label=='Cons 2015'].dim1.values[0]:
    Zcan['dim1'] = Zcan.dim1 * (-1)


# In[57]:


print(len(canlabs))
print(len(cancols))
print(len(canmkers))


# In[1]:


#plot_3b(Zcan, canlabs, cancols, canmkers, savepath='figures/figure3b.pdf')
#print("Saved Figure 3b to file figures/figure3b.pdf")


# ## Table 2- accuracy of placements 

# In[74]:


gold_can = pd.read_csv('data/goldstandard_canada.csv').merge(Zcan, on='label', how='left')


# In[75]:


gold_scores = ['voteview', 'experts_stand', 'rile', 'vanilla', 'legacy']
countries = [('Canada', gold_can)]
results = np.zeros(( 10, 1 ), dtype=object)

for idx, (c, df) in enumerate(countries):
    jdx = 0
    for g in gold_scores:
        if g=='voteview' and 'voteview' not in df.columns:
            results[jdx:(jdx+2),idx] = ['','']
        else:
            temp = df[pd.notnull(df[g])]
            corr = '%0.3f' %temp.dim1.corr(temp[g])
            acc = '%0.2f%%' %pairwise_accuracy(temp[g].tolist(), temp.dim1.tolist())
            results[jdx:(jdx+2),idx] = [corr, acc]
        jdx += 2

results = pd.DataFrame(results, columns = [c for c,df in countries])
results.insert(loc=0,column='Metric',value=['Correlation', 'Accuracy']*5)
results.insert(loc=0,column='Gold Standard',value=[item for item in ['Voteview', 'Experts Surveys', 'rile', 'vanilla', 'legacy'] for i in range(2)])


# In[76]:


results


# In[83]:


with open('tables/table2.txt', 'w') as f:
    print("Table 2: Accuracy of Party Placement against Gold Standards\n"+"-"*83, file=f)
    print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl"), file=f)
print("Saved Table 2 to file tables/table2.txt")


# In[ ]:




