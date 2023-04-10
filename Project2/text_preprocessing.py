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
abst = abst.str.replace('\d+', ' ') # remove numbers

### Lowercasing
abst = abst.str.lower()

### Stemming
abst_parsed = abst.str.split()
stemmer = PorterStemmer()
abst_stem = abst_parsed.apply(lambda x: [stemmer.stem(word) for word in x])
abst_stem = abst_stem.apply(lambda x: ' '.join(x))

### stopWord removal
# Add custom stopwords to the default list
custom_stopwords = {'paper', 'research', 'study', 'literatures', 'article', 'ii', 'iii', 'john', 'wiley', 'sons'}
STOPWORDS |= custom_stopwords
abst = abst.apply(lambda x: remove_stopwords(x, stopwords=STOPWORDS)) 
abst_stem = abst_stem.apply(lambda x: remove_stopwords(x, stopwords=STOPWORDS)) 
print(abst.shape)
print(abst_stem.shape)

### 3 n-grams & Infrequently used terms
vectorizer1 = CountVectorizer(ngram_range=(1,3), min_df=0.01) # Initialize CountVectorizer with 1,2,3-gram and min_df of 1%
vector = vectorizer1.fit_transform(abst).toarray() # Fit the vectorizer to your text data
print(vector.shape)
print(len(vectorizer1.vocabulary_))
# for stemmed data
vectorizer2= CountVectorizer(ngram_range=(1,3), min_df=0.01) 
vector_stem = vectorizer2.fit_transform(abst_stem).toarray() # Fit the vectorizer to your text data
print(vector_stem.shape)

### compare with stemming vs. without stemming
wostm = list(vectorizer1.vocabulary_.keys()) # without stemming
print(sorted(wostm))
wstm = list(vectorizer2.vocabulary_.keys()) # with stemming
print(sorted(wstm))

### visualize word counts per journal
words_df = pd.DataFrame(vector) # without stemming