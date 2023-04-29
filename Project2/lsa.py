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

def main():
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

    print(corpus[0])
    print(corpus_PNLMf[0], end="")

    # Stopword removal with gensim Library: W
    # This applied to the lemmatized words! Not to the stemmed words.
    all_stopwords_add = STOPWORDS.union(set(['x', 'y', 'I', 's', 'e', 'study', 'paper', 'research', 'study', 'literatures', 'article', 'ii', 'iii', 'john',
                                            'wiley', 'sons', 'use', 'examine', 'investigate', 'approach', 'argue', 'effect', 'positive', 'negative', 'result',
                                            'high', 'increase', 'subsequent', 'subsequently', 'r', 'r d', 'specific', 'specifically', 'think', 'test',
                                            'substantially', 'robust', 'second', 'report']))
    print(len(all_stopwords_add))
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

    dtm_md = CountVectorizer(tokenizer = feed, min_df = 0.01, max_df = 0.7, ngram_range = (1,3), token_pattern = None, lowercase = False)
    dtfm = dtm_md.fit_transform(corpus_PNLMW)
    print(type(dtfm), dtfm.shape)

    dtfmx = pd.DataFrame(dtfm.toarray(), index= P_names, columns= dtm_md.get_feature_names_out())
    print(dtfmx.head())
    '''
    # find optimal number of clusters
    explained_variance = []
    for num_topics in range(2,51):
        topics = ['topic'+ str(i+1) for i in range(num_topics)]
        pp_LSA_sklm = TruncatedSVD(n_components= num_topics, n_iter=7, random_state=42)
        pp_LSA_dtm = pp_LSA_sklm.fit_transform(dtfm)
        explained_variance.append(pp_LSA_sklm.explained_variance_ratio_.sum())
    
    plt.plot(range(2,51), explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Elbow Method for Optimal n_components')
    plt.show()
    opt_num = np.argmax(np.gradient(explained_variance))+2
    print("Optimal number of topics using scikit-learn elbow method: {}".format(opt_num))
    
    topics = ['topic'+ str(i+1) for i in range(opt_num)]
    pp_LSA_sklm = TruncatedSVD(n_components= opt_num, n_iter=7, random_state=42)
    pp_LSA_dtm = pp_LSA_sklm.fit_transform(dtfm)
    pp_topictermMtx = pd.DataFrame(pp_LSA_sklm.components_, index = topics, columns= dtm_md.get_feature_names_out())
    print(pp_topictermMtx.head())

    # Finding N top words for each topic
    n_words = 15
    idx_nmax_rf = [abs(row).nlargest(n_words).index for index, row in pp_topictermMtx.iterrows()]
    print(type(idx_nmax_rf))
    print(pd.DataFrame(idx_nmax_rf, index = topics).T)

    # Checking the performance: dtmf
    print(pp_LSA_sklm.explained_variance_ratio_)
    print(pp_LSA_sklm.explained_variance_ratio_.sum())
    print(pp_LSA_sklm.singular_values_)
    
    first_point = [0, explained_variance[0]]
    last_point = [len(explained_variance)-1, explained_variance[-1]]
    distances = []
    for i, point in enumerate(explained_variance):
        x = [i, point]
        distance = np.linalg.norm(np.cross(np.subtract(x, first_point), np.subtract(x, last_point))) / np.linalg.norm(np.subtract(last_point, first_point))
        distances.append(distance)
    
    # Find the point with the maximum distance
    opt_num_2 = np.argmax(distances) + 2
    print("Optimal number of topics using scikit-learn distance method: {}".format(opt_num_2))
    
    topics = ['topic'+ str(i+1) for i in range(opt_num_2)]
    pp_LSA_sklm = TruncatedSVD(n_components= opt_num_2, n_iter=7, random_state=42)
    pp_LSA_dtm = pp_LSA_sklm.fit_transform(dtfm)
    pp_topictermMtx = pd.DataFrame(pp_LSA_sklm.components_, index = topics, columns= dtm_md.get_feature_names_out())
    print(pp_topictermMtx.head())

    # Finding N top words for each topic
    n_words = 15
    idx_nmax_rf = [abs(row).nlargest(n_words).index for index, row in pp_topictermMtx.iterrows()]
    print(type(idx_nmax_rf))
    print(pd.DataFrame(idx_nmax_rf, index = topics).T)

    # Checking the performance: dtmf
    print(pp_LSA_sklm.explained_variance_ratio_)
    print(pp_LSA_sklm.explained_variance_ratio_.sum())
    print(pp_LSA_sklm.singular_values_)
    
    # Tracking topic changes across years for each journal
    df = pd.concat([NLP_finn, pd.DataFrame(pp_LSA_dtm, columns=['skl_topic'+ str(i+1) for i in range(opt_num_2)])], axis=1)
    
    def top_5_cols(row):
        return row.nlargest(5).index.tolist()
    
    # year variable in category - 1: 1991-1995, 2: 1996-2000, 3: 2001-2005, 4: 2006-2010, 5: 2011-2015, 6: 2016-2020, 7: 2021-2023
    df['catYear'] = (df['Year']-1991)//5+1
    with pd.option_context('display.max_colwidth', None):
        for journal in df['Jabb'].unique():
            tmp = df[df['Jabb']== journal]
            tmp = tmp.drop('Year', axis=1)
            tmp = tmp.groupby(['catYear']).mean()
            tmp['top_5_cols'] = tmp.apply(top_5_cols, axis=1)
            print("The top 5 topics per year category for the following journal: {}".format(journal))
            print(tmp['top_5_cols'])
    '''
    ### latent semantic analysis (LSA) with gensim
    # DTM to DTL by document
    fin_words = dtm_md.inverse_transform(dtfm)
    print(len(fin_words[0]), fin_words[0])
    # Mapping between unique words and word id (making a dictionary)
    pp_dict = Dictionary(fin_words)
    # Given a dictionary (pp_dict), make a word frequency table for each document.
    wordfreq_doc = [pp_dict.doc2bow(text) for text in fin_words] #corpus_PNLMW]
    print(len(wordfreq_doc))
    print(wordfreq_doc[0], end = " ")

    coherence_values = []
    model_list = []
    for n_topics in range(2,30):
        model = models.LsiModel(wordfreq_doc, num_topics= n_topics, chunksize = 64, id2word = pp_dict)
        model_list.append(model)
        coherencemodel = CoherenceModel(model= model, texts= fin_words, dictionary= pp_dict, coherence= 'c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("{}-th process finished!!".format(n_topics-1))

    x = range(2,30)
    plt.plot (x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc= 'best')
    plt.show()
    
    opt_num_3 = np.argmax(coherence_values)+2
    print("Optimal number of topics using Gensim, coherence value: {}".format(opt_num_3))
    model = model_list[opt_num_3 - 2]
    with pd.option_context('display.max_colwidth', None):
        print(pd.DataFrame(model.print_topics(num_words=15))) # get the topics
    
    opt_num_4 = max(np.where(coherence_values > 0.9*max(coherence_values))[0])+2
    print("Largest number of topics with coherence value with 90 percent of maximum or above: {}".format(opt_num_4))
    model = model_list[opt_num_4 - 2]
    with pd.option_context('display.max_colwidth', None):
        print(pd.DataFrame(model.print_topics(num_words=15))) # get the topics
    corpus_lsi = np.asarray(model[wordfreq_doc])[:,:,1] # (document, num_topic) distribution
    # Tracking topic changes across years for each journal
    df = pd.concat([NLP_finn, pd.DataFrame(corpus_lsi, columns=['topic'+ str(i+1) for i in range(opt_num_4)])], axis=1)
    
    def top_5_cols(row):
        return row.nlargest(5).index.tolist()
    
    # year variable in category - 1: 1991-1995, 2: 1996-2000, 3: 2001-2005, 4: 2006-2010, 5: 2011-2015, 6: 2016-2020, 7: 2021-2023
    df['catYear'] = (df['Year']-1991)//5+1
    with pd.option_context('display.max_colwidth', None):
        for journal in df['Jabb'].unique():
            tmp = df[df['Jabb']== journal]
            tmp = tmp.drop('Year', axis=1)
            tmp = tmp.groupby(['catYear']).mean()
            tmp['top_5_cols'] = tmp.apply(top_5_cols, axis=1)
            print("The top 5 topics per year category for the following journal: {}".format(journal))
            print(tmp['top_5_cols'])
    
if __name__ == "__main__":
    main()