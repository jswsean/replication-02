from nltk.tokenize import ToktokTokenizer
import string
from sklearn.feature_extraction import text
from functools import reduce
import pandas as pd
import unicodedata
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=8)
import sys

'''
Script to read and pocess raw UK hansard corpus. Output is hdf5 file of dataframe with pre-processed text in cleaned_text colum.

'''

df = pd.read_csv("1_raw/uk/uk_hansard_1935_2014_BvW_2022.tsv", sep='\t', usecols=['year', 'parliament', 'party', 'speech_text'] )

def clean_text(text):
    import pandas as pd
    from nltk.tokenize import ToktokTokenizer
    import string
    from functools import reduce
    import pandas as pd
    import unicodedata
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True,nb_workers=8)

    from sklearn.feature_extraction import text as txt
    britain_stopwords = ['member','members','government','governments','opposition','opposite','leader',
     'hon','exminister','prime','minister','ministers','parliament','house',
     'ask','asked','asks','question','questioned','questions','bills','bill',
     'party','parties','mp','mps','sir','madam','mr','gentleman','gentlemen','lady','ladies',
     'speaker','chair','motion','motions','vote','votes','order','yes','deputy','secretary',
     'uk','british','britain',
     'pursuant','supply','supplementary','please','friend','s',
     'clause','amendment','i','ii','iii','section','sections', 'colleague', 'colleagues'] + list(txt.ENGLISH_STOP_WORDS)
    
    contractions = {"you'd": 'you would', "he'd": 'he would', "she's": 'she is', "where'd": 'where did', "might've": 'might have', "he'll": 'he will', "they'll": 'they will',  "mightn't": 'might not', "you'd've": 'you would have', "shan't": 'shall not', "it'll": 'it will', "mayn't": 'may not', "couldn't": 'could not', "they'd": 'they would', "so've": 'so have', "needn't've": 'need not have', "they'll've": 'they will have', "it's": 'it is', "haven't": 'have not', "didn't": 'did not', "y'all'd": 'you all would', "needn't": 'need not', "who'll": 'who will', "wouldn't've": 'would not have', "when's": 'when is', "will've": 'will have', "it'd've": 'it would have', "what'll": 'what will', "that'd've": 'that would have', "y'all're": 'you all are', "let's": 'let us', "where've": 'where have', "o'clock": 'oclock', "when've": 'when have', "what're": 'what are', "should've": 'should have', "you've": 'you have', "they're": 'they are', "aren't": 'are not', "they've": 'they have', "it'd": 'it would', "i'll've": 'i will have', "they'd've": 'they would have', "you'll've": 'you will have', "wouldn't": 'would not', "we'd": 'we would', "hadn't've": 'had not have', "weren't": 'were not', "i'd": 'i would', "must've": 'must have', "what's": 'what is', "mustn't've": 'must not have', "what'll've": 'what will have', "ain't": 'aint', "doesn't": 'does not', "we'll": 'we will', "i'd've": 'i would have', "we've": 'we have', "oughtn't": 'ought not', "you're": 'you are', "who'll've": 'who will have', "shouldn't": 'should not', "can't've": 'cannot have', "i've": 'i have', "couldn't've": 'could not have', "why've": 'why have', "what've": 'what have', "can't": 'cannot', "don't": 'do not', "that'd": 'that would', "who's": 'who is', "would've": 'would have', "there'd": 'there would', "shouldn't've": 'should not have', "y'all": 'you all', "mustn't": 'must not', "she'll": 'she will', "hadn't": 'had not', "won't've": 'will not have', "why's": 'why is', "'cause": 'because', "wasn't": 'was not', "shan't've": 'shall not have', "ma'am": 'madam', "hasn't": 'has not', "to've": 'to have', "how'll": 'how will', "oughtn't've": 'ought not have', "he'll've": 'he will have', "we'd've": 'we would have', "won't": 'will not', "could've": 'could have', "isn't": 'is not', "she'll've": 'she will have', "we'll've": 'we will have', "you'll": 'you will', "who've": 'who have', "there's": 'there is', "y'all've": 'you all have', "we're": 'we are', "i'll": 'i will', "i'm": 'i am', "how's": 'how is', "she'd've": 'she would have', "sha'n't": 'shall not', "there'd've": 'there would have', "he's": 'he is', "it'll've": 'it will have', "that's": 'that is', "y'all'd've": 'you all would have', "he'd've": 'he would have', "how'd": 'how did', "where's": 'where is', "so's": 'so as', "she'd": 'she would', "mightn't've": 'might not have'}

    tk = ToktokTokenizer()
    text = reduce(lambda a, kv: a.replace(*kv), contractions.items(), text.lower())
    text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = tk.tokenize(text)
    stopwords = britain_stopwords
    tokens = [w for w in tokens if w not in stopwords and len(w)>2 and w!=' ' and not w.isdigit()]
    return ' '.join(tokens)

df['cleaned_text'] = df['speech_text'].parallel_apply(clean_text)

df_export = df[['year', 'parliament', 'party', 'cleaned_text']]

df_export.to_hdf('2_build/processed_uk_corpus.h5', key = 's')