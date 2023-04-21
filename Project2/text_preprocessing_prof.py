import numpy as np
import matplotlib.pyplot as plt
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

import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


### read preprocessed data
NLP_finn = pd.read_csv(os.path.join("Project2", "data", "agg_article_info.csv"))

# Creating a corpus with all abstracts
corpus = [NLP_finn['Abst'][i] for i in range(len(NLP_finn['Abst']))]
          
# Cleaning the unnecessary terms and creating a cleaned corpus
corpusn = [i.replace('Research Summary','').replace('Research Abstract','').replace('Research summary','') for i in corpus]

# Modified code of Junki Hong's original code
corpusnn = []
for i in corpusn:
    sentences = sent_tokenize(i)
    for j in sentences:
        if 'Copyright (' in j:
            sentences.remove(j)
    cleanedAbst = ' '.join(sentences)
    corpusnn.append(cleanedAbst)
    
### Removing punctuations
corpus_P = [i.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip() for i in corpusnn]

### Removing numbers
# removing numbers: N
corpus_PN = [i.translate(str.maketrans('','',string.digits)) for i in corpus_P]
# lowercasing: L
corpus_PNL = [i.lower() for i in corpus_PN]

### Lemmatizing (M) the bill with spaCy library instead of stemming
# Use the text before tokenizing; a string need to be provided. spacy does both tokening and lemmatizing.
# It takes 4 to 5 minutes and increase nlp.max length to the length set above.
nlp = spacy.load("en_core_web_sm")
corpus_PNLM = [[j.lemma_ for j in nlp(i)] for i in corpus_PNL]
abst_wclm = [len(i) for i in corpus_PNLM]
print(len(abst_wclm))

# Removing empty strings from the results of lemmatizing
corpus_PNLMf = [' '.join(i).split() for i in corpus_PNLM]
print(len(corpus_PNLMf))
bill_wclms = [len(i) for i in corpus_PNLMf]
print(len(bill_wclms))

print(corpus [0])
print(corpus_PNLMf[0], end="")

# Stopword removal with gensim Library: W
# This applied to the lemmatized words! Not to the stemmed words.
all_stopwords_add = STOPWORDS.union(set(['x', 'y', 'I', 's', 'study']))
print (len (all_stopwords_add))
corpus_PNLMW = [[j for j in i if not j in all_stopwords_add] for i in corpus_PNLMf]

# Checking if stopword removal reduced words in each bill
print(len(corpus_PNLMW))
print(corpus_PNLMW[0], end = '')
bill_wc1 = [len(i) for i in corpus_PNLMW]

### latent semantic analysis (LSA) with scikit-learn
# Making document title for each bill
P_names = ['p' + str(i) for i in range(len(corpus_PNLMW))]

# Making document-term frequency matrix
def feed(wordlist): # Use this for feeding preprocessed tokens
    return wordlist

dtm_md = CountVectorizer(tokenizer = feed, min_df = 5, max_df = 0.5, ngram_range = (1,1), token_pattern = None, lowercase = False)
dtfm = dtm_md.fit_transform(corpus_PNLMW)
print(type(dtfm), dtfm.shape)