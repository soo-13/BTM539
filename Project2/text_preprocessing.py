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
# for unstemmed data
vector = vectorizer1.fit_transform(abst).toarray() # Fit the vectorizer to your text data
print(vector.shape)
print(len(vectorizer1.vocabulary_))
# for stemmed data
vectorizer2= CountVectorizer(ngram_range=(1,3), min_df=0.01) 
vector_stem = vectorizer2.fit_transform(abst_stem).toarray() # Fit the vectorizer to your text data
print(vector_stem.shape)

### compare with stemming vs. without stemming
wostm = list(vectorizer1.vocabulary_.keys()) # without stemming
#print(sorted(wostm))
wstm = list(vectorizer2.vocabulary_.keys()) # with stemming
#print(sorted(wstm))

### visualize word counts per journal 
journal_title = {'MANAGE SCI': 'MS', 'ORGAN SCI': 'OS', 'ACAD MANAGE J': 'AMJ', 'STRATEGIC MANAGE J': 'SMJ'}
def visualize_word_counts_per_journal(wc, title):
    colors = ['blue', 'green', 'red', 'purple']
    plt.clf()
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    for i, ax in enumerate(axs.flat):
        journal = df['Jabb'].unique()[i]
        tmp = wc[df['Jabb']==journal]
        ax.hist(tmp, bins=30, alpha=0.5, color=colors[i])
        ax.set_title(journal_title[journal])
        ax.set_xlabel('Word count')
        ax.set_ylabel('frequency')
        ax.xaxis.set_label_coords(1.1, -0.05) # move xlabel to the right side of x-axis
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.suptitle(title, fontweight='bold')
    plt.show()
       
# after preprocessing
words_count_after = pd.Series(np.sum(vector, axis=1)) # word count for each article as pd Series
visualize_word_counts_per_journal(words_count_after, 'Word count distribution by journal after preprocessing')
words_count_before = df['Abst'].apply(lambda x: len(x.split()))
visualize_word_counts_per_journal(words_count_before, 'Word count distribution by journal before preprocessing')

