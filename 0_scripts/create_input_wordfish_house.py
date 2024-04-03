import pandas as pd

''' Script to create input data for wordfish
Input: Pre-Processed US house corpus text
Output: dataframe with each row as unique party congress session pair. text column contains all concated debates for party congress session '''


df = pd.DataFrame(columns=['session', 'chamber', 'party', 'text'])
sessions = []
chambers = []
texts = []
parties = []
counter = 0

with open("data/preprocessed_congress", 'r') as f:
    for line in f:
        ls = line.split('\t')
        chamber = ls[5]
        if chamber == 'H':
            chambers.append(chamber)
            text = ls[10].replace('\n', '')
            texts.append(text)
            session = ls[0]
            sessions.append(session)
            party = ls[7]
            parties.append(party)
            counter += 1
            if counter % 100000 == 0:
                print(f'Done with {counter} lines')

df['session'] = sessions
df['chamber'] = chambers
df['party'] = parties
df['text'] = texts

## Join texts such that each party - congress sesion pair has one document 
grouped = df.groupby(['session', 'party'])['text'].agg(lambda x: ' '.join(x))

congress = [i for i in range(43,115)]
years = [i for i in range(1873,2017,2)]
usa_index_toyear = {str(c):y for c,y in zip(congress, years)}

df2 = pd.DataFrame(grouped)
df2.reset_index(inplace = True)

df2['year'] = df2.session.astype(str).map(usa_index_toyear)

df2.to_excel('outputs/wf_set.xlsx', index = False)