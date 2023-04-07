import numpy as np
import matplotlib.pyplot as pit
import os
import pandas as pd
import spacy
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from gensim import corpora, models, utils
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel 
from gensim.parsing.preprocessing import STOPWORDS, remove_stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


### read preprocessed data
df = pd.read_csv(os.path.join("Project2", "data", "agg_article_info.csv"))

abst = df['Abst'] # abstract

### Punctuation
regular_punct = list(string.punctuation) # list of punctuations
for punc in regular_punct:
    abst = abst.str.replace(punc, '') # remove punctuations

### Numbers
abst = abst.str.replace('\d+', '') # remove numbers ugh.. there are latin numbers as well..

### Lowercasing
abst = abst.str.lower()

### Stemming
abst_parsed = abst.str.split()
stemmer = PorterStemmer()
abst_stem = abst_parsed.apply(lambda x: [stemmer.stem(word) for word in x])

### stopWord removal
# Add custom stopwords to the default list
custom_stopwords = {'paper', 'research', 'study', 'ii', 'iii'}
STOPWORDS |= custom_stopwords
abst = abst.apply(lambda x: remove_stopwords(x)) 

### 3 n-gram Inclusion ㅇ