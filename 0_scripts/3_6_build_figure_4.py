'''
This file reproduces figures 4a and 4b

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
import lib.labels as labels
import lib.plots as plots
from lib.interpret import Interpret
from lib.accuracy import pairwise_accuracy
import os

# #=========================================================================================#
# # Figure 4: Comparison with WordFish Estimates
# #=========================================================================================#
# # Loading fitted WordFish models
wf_full = pd.read_excel("2_build/house_document_scores_pivoted.xlsx")
wf_short = pd.read_excel("2_build/house_document_scores_mostrecentpivoted.xlsx")

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
plots.plot_4a(pfull, savepath='3_docs/figures/figure4a_fullcorpus.pdf')
plots.plot_4a(pfull_anc, savepath='3_docs/figures/figure4a_fullcorpus_anc.pdf')

# Figure 4b
plots.plot_4b(pshort, savepath='3_docs/figures/figure4b_mostrecentongress.pdf')
plots.plot_4b(pshort_anc, savepath='3_docs/figures/figure4b_mostrecentcongress_anc.pdf')
