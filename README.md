# Overview

This repository hosts all materials that are used in the replication exercise for the [PPOL 6801: Text as Data](https://tiagoventura.github.io/PPOL_6801_2024) course assignment. For the assignment, [Grace](https://github.com/Gracej12), [Ayush](https://github.com/AyushLahiri) and I chose to replicate [Rheault and Cochrane (2020)](https://www.cambridge.org/core/journals/political-analysis/article/word-embeddings-for-the-analysis-of-ideological-placement-in-parliamentary-corpora/017F0CEA9B3DB6E1B94AC36A509A8A7B)'s article, "Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora". 

This article demonstrates the use of word embeddings, as augmented by political metadata, to infer ideological placement of political parties and legislators. The paper applies this method on several text corpuses, namely the US Congress as well as British and Canadian Parliament, and shows that these models could be better suited for political text than the typical ideological scaling methods.

The replication exercise aims to reproduce the figures and tables in the main section of the paper. Replication of the appendix figures/tables lies beyond the scope of this exercise.

The replication results are accessible [here](https://github.com/jswsean/replication-02/blob/master/3_docs/presentation/replication_presentation.pdf), and the replication report can be seen [here](https://github.com/jswsean/replication-02/blob/master/3_docs/report/replication_report.pdf). 

The following sections in this README provide information on the overall structure of this repository, as well as the original source for the replication materials.


# Structure

This repository has three main folders:

* `0_scripts`: this folder contains both the replication scripts. These scripts are mostly adapted from [the authors' original scripts](https://github.com/lrheault/partyembed/tree/master/src) with some modifications to ensure compatibility with more recent versions of `gensim` (4.X). There is a `lib` subfolder inside the `0_scripts` folder, which contains labeling, plotting, and evaluation functions that are used by the authors to produce the original results. The Python scripts are separated based on their sequences in the analytical stages as well as their outputs, and the tasks implemented by each scripts are as follows:

    - `1_1_build_congress_text.py`: Builds the raw U.S. corpus
    - `1_2_process_congress_text.py`: Pre-processes the raw U.S. corpus
    - `1_3_process_canada_text.py`: Pre-processes the raw Canadian corpus
    - `1_4_process_uk_text.py`: Pre-processes the raw British corpus
    - `1_5_process_input_wordfish_house.py`: Pre-processes inputs for the Wordfish estimation model
    - `2_1_train_partyembeddings_house.py`: Trains `Doc2Vec` model on U.S. House corpus 
    - `2_2_train_partyembeddings_senate.py`: Trains `Doc2Vec` model on U.S. Senate corpus
    - `2_3_train_senatorembeddings_senate114.py`: Trains the `Doc2Vec` model on the 114th Senate corpus to get senator-level embeddings
    - `2_4_train_partyembeddings_canada.py`: Trains the `Doc2Vec` model on the Canadian parliament corpus 
    - `2_5_train_partyembeddings_uk.py`: Trains the `Doc2Vec` model on the British parliament corpus 
    - `2_6_train_wordfish_house.py`: Trains `WordFish` model to provide alternative estimates on congress-level ideological placements 
    - `3_1_build_figure_2.py`: Builds Figure 2 of the original paper
    - `3_2_build_table_1.py`: Builds Table 1
    - `3_3_build_figure_3a.py`: Builds Figure 3a 
    - `3_4_build_figure_3b.py`: Builds Figure 3b
    - `3_5_build_table_2.py`: Builds Table 2 
    - `3_6_build_figure_4.py`: Builds Figure 4
    - `3_7_build_table_3.py`: Builds Table 3
    - `3_8_build_figure_5.py`: Builds Figure 5
    - `3_9_bukld_table_4.py`: Builds Table 4

* `2_build`: this folder stores the results of our trained models on each of the US, UK, and Canadian corpus. 

* `3_docs`: this folder contains the outputs of the replication exercise, both the figures (in the fig subfolder) and tables (in the tab subfolder). Additionally, this folder also contains the replication presentation and report.


# Data 

All materials for the replication, including the raw data, are sourced from this [Harvard Dataverse repository](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=ccbe4ad771f0ce57d8169462f553?persistentId=doi%3A10.7910%2FDVN%2FK0OYQF&version=&q=&fileTypeGroupFacet=%22Unknown%22&fileAccess=). While the workflow of the scripts is that we import data from the `1_raw` folder, pre-process it and export the resulting data to the `2_build` folder, we do not track both input and output data in this repository. Readers who wish to follow along can create the data folders in their own local machine.

# References

Rheault, L., & Cochrane, C. (2020). Word embeddings for the analysis of ideological placement in parliamentary corpora. _Political Analysis, 28_(1), 112-133.
