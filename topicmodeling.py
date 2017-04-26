#!/home/hackpython/anaconda/bin/python

# Author: Abhishek Sharma
# Program: Implementation of Non-negative Matrix Factorization ( NMF ) and Latent Dirichlet Allocation ( LDA )

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import sys
import os

def display_topics(model, feature_names, no_top_words):
    
    for topic_idx, topic in enumerate(model.components_):
        print("Topic:", (topic_idx))
        print(" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
#  Data Set;
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

#  Number of Features to select.
total_features = 1000

#  Non Negative Matrix Factorization Algorithm Implementation, to know more about parameter visit
#  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=total_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#  Latent Dirichlet Allocation Implementation
#  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=total_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

num_topic = 5

model_nmf = NMF(n_components=num_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
model_lda = LatentDirichletAllocation(n_topics=num_topic, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(model_nmf, tfidf_feature_names, no_top_words)
display_topics(model_lda, tf_feature_names, no_top_words)
