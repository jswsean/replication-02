import numpy as np
import pandas as pd
from tabulate import tabulate
from gensim.models import Doc2Vec

def pairwise_accuracy(gold, test):
    assert len(gold)==len(test)
    pairs = 0
    correct = 0
    # Loop over unique pairs.
    for i in range(len(test)):
        for j in range(i+1,len(test)):
            # Count pairs correctly ordered.
            if np.sign(gold[i]-gold[j])==np.sign(test[i]-test[j]):
                correct += 1
            pairs += 1
    # Return pairwise accuracy as percentage.
    return (correct/pairs)*100

def analogies(model):
    total_score, section_scores = model.wv.evaluate_word_analogies('data/questions-words.txt', restrict_vocab=10000)
    names = []
    results = []
    for section in section_scores:
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            score = correct / (correct + incorrect)
            names.append(section['section'])
            results.append("%.1f%% (%i/%i)" %(100.0 * score, correct, correct + incorrect))
        else:
            names.append(section['section'])
            results.append("")
    return names, results

def word_similarities(model):
    c1, c2, _ = model.wv.evaluate_word_pairs('data/wordsim353.tsv')
    return ('%.4f (%.2e)' %(c1[0], c1[1])), ('%.4f (%.2e)' % (c2[0], c2[1]))

def hyperparameter_tables(params):

    if params=='layer_size':

        results = np.zeros(( 12, 2 ), dtype=object)
        countries = [('house','voteview','House'), ('senate','voteview','Senate'), ('uk','legacy','UK'), ('canada', 'legacy','Canada')]
        tested = ['ep5_dim100_delta20_lr025', 'ep5_dim200_delta20_lr025', 'ep5_dim300_delta20_lr025']

        idx = 0
        for c, metric, _ in countries:
            projections = pd.read_csv('data/' + c + '_parameterization.csv')
            temp = pd.read_csv('data/goldstandard_' + c + '.csv').merge(projections, on='label', how='left')
            for jdx, t in enumerate(tested):
                results[idx + jdx,0] = '%0.3f' %temp[t].corr(temp[metric])
                results[idx + jdx,1] = '%0.3f%%' %pairwise_accuracy(temp[metric].tolist(), temp[t].tolist())
            idx = idx + 3

        results = pd.DataFrame(results, columns = ['Pearson Correlation', 'Pairwise Accuracy'], dtype=object)
        results.insert(loc=0, column='M', value=['100','200','300']*4)
        results.insert(loc=0, column='Corpus', value=[item for _, _, item in countries for i in range(3)])

        with open('tables/tableA6.txt', 'w') as f:
            print("Table A6: Effect of Layer Size on Accuracy\n"+"-"*64, file=f)
            print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl", floatfmt=".3f"), file=f)

    elif params=='Part1':

        results = np.zeros(( 28, 2 ), dtype=object)
        countries = [('house','voteview','House'), ('senate','voteview','Senate')]
        tested = ['ep5_dim100_delta15_lr025', 'ep5_dim200_delta15_lr025', 'ep5_dim300_delta15_lr025',
                  'ep5_dim100_delta20_lr025', 'ep5_dim200_delta20_lr025', 'ep5_dim300_delta20_lr025',
                  'ep5_dim100_delta30_lr025', 'ep5_dim200_delta30_lr025', 'ep5_dim300_delta30_lr025',
                  'ep5_dim200_delta20_lr01', 'ep5_dim200_delta20_lr02', 'ep5_dim200_delta20_lr03',
                  'ep5_dim200_delta20_lr04', 'ep5_dim200_delta20_lr05']    

        idx = 0
        for c, metric, _ in countries:
            projections = pd.read_csv('data/' + c + '_parameterization.csv')
            temp = pd.read_csv('data/goldstandard_' + c + '.csv').merge(projections, on='label', how='left')
            for jdx, t in enumerate(tested):
                results[idx + jdx,0] = '%0.3f' %temp[t].corr(temp[metric])
                results[idx + jdx,1] = '%0.3f%%' %pairwise_accuracy(temp[metric].tolist(), temp[t].tolist())
            idx = idx + 14

        results = pd.DataFrame(results, columns = ['Pearson Correlation', 'Pairwise Accuracy'], dtype=object)
        results.insert(loc=0, column='Learning Rate', value=['0.025']*9 + ['0.01','0.02','0.03','0.04','0.05'] + ['0.025']*9 + ['0.01','0.02','0.03','0.04','0.05'])
        results.insert(loc=0, column='Delta', value=['15']*3 + ['20']*3 + ['30']*3 + ['20']*5 + ['15']*3 + ['20']*3 + ['30']*3 + ['20']*5)
        results.insert(loc=0, column='M', value=['100','200','300']*3 + ['200']*5 + ['100','200','300']*3 + ['200']*5)
        results.insert(loc=0, column='Corpus', value=[item for item in ['House','Senate'] for i in range(14)])

        with open('tables/tableA9.txt', 'w') as f:
            print("Table A9:Extended Accuracy Results for Various Parameterizations, Part 1\n"+"-"*92, file=f)
            print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl", floatfmt=".3f"), file=f)

    elif params=='Part2':

        results = np.zeros(( 28, 2 ), dtype=object)
        countries = [('canada','legacy'), ('uk','legacy')]
        tested = ['ep5_dim100_delta15_lr025', 'ep5_dim200_delta15_lr025', 'ep5_dim300_delta15_lr025',
                  'ep5_dim100_delta20_lr025', 'ep5_dim200_delta20_lr025', 'ep5_dim300_delta20_lr025',
                  'ep5_dim100_delta30_lr025', 'ep5_dim200_delta30_lr025', 'ep5_dim300_delta30_lr025',
                  'ep5_dim200_delta20_lr01', 'ep5_dim200_delta20_lr02', 'ep5_dim200_delta20_lr03',
                  'ep5_dim200_delta20_lr04', 'ep5_dim200_delta20_lr05']    

        idx = 0
        for c, metric in countries:
            projections = pd.read_csv('data/' + c + '_parameterization.csv')
            temp = pd.read_csv('data/goldstandard_' + c + '.csv').merge(projections, on='label', how='left')
            for jdx, t in enumerate(tested):
                results[idx + jdx,0] = '%0.3f' %temp[t].corr(temp[metric])
                results[idx + jdx,1] = '%0.3f%%' %pairwise_accuracy(temp[metric].tolist(), temp[t].tolist())
            idx = idx + 14

        results = pd.DataFrame(results, columns = ['Pearson Correlation', 'Pairwise Accuracy'], dtype=object)
        results.insert(loc=0, column='Learning Rate', value=['0.025']*9 + ['0.01','0.02','0.03','0.04','0.05'] + ['0.025']*9 + ['0.01','0.02','0.03','0.04','0.05'])
        results.insert(loc=0, column='Delta', value=['15']*3 + ['20']*3 + ['30']*3 + ['20']*5 + ['15']*3 + ['20']*3 + ['30']*3 + ['20']*5)
        results.insert(loc=0, column='M', value=['100','200','300']*3 + ['200']*5 + ['100','200','300']*3 + ['200']*5)
        results.insert(loc=0, column='Corpus', value=[item for item in ['Canada','UK'] for i in range(14)])

        with open('tables/tableA10.txt', 'w') as f:
            print("Table A10: Extended Accuracy Results for Various Parameterizations, Part 2\n"+"-"*92, file=f)
            print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl", floatfmt=".3f"), file=f)

    elif params=='Part3':

        results = np.zeros(( 20, 2 ), dtype=object)
        countries = [('house','voteview','House'), ('senate','voteview','Senate'), ('canada', 'legacy','Canada'), ('uk','legacy','UK')]
        tested = ['ep1_dim200_delta20_lr025', 'ep3_dim200_delta20_lr025', 'ep5_dim200_delta20_lr025', 
                 'ep10_dim200_delta20_lr025', 'ep15_dim200_delta20_lr025']    

        idx = 0
        for c, metric, _ in countries:
            projections = pd.read_csv('data/' + c + '_parameterization.csv')
            temp = pd.read_csv('data/goldstandard_' + c + '.csv').merge(projections, on='label', how='left')
            for jdx, t in enumerate(tested):
                results[idx + jdx,0] = '%0.3f' %temp[t].corr(temp[metric])
                results[idx + jdx,1] = '%0.3f%%' %pairwise_accuracy(temp[metric].tolist(), temp[t].tolist())
            idx = idx + 5

        results = pd.DataFrame(results, columns = ['Pearson Correlation', 'Pairwise Accuracy'], dtype=object)
        results.insert(loc=0, column='Epochs', value=['1','3','5','10','15']*4)
        results.insert(loc=0, column='Corpus', value=[item for _, _, item in countries for i in range(5)])

        with open('tables/tableA11.txt', 'w') as f:
            print("Table A11: Extended Accuracy Results for Various Parameterizations, Part 3\n"+"-"*69, file=f)
            print(tabulate(results, headers="keys", showindex=False, tablefmt="orgtbl", floatfmt=".3f"), file=f)

    else:

        raise ValueError("The options are either 'layer_size', 'Part1', 'Part2', or 'Part3'.")
