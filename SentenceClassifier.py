# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 22:29:51 2017

@author: bharathiraja
"""
import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

sentences=[]
def GetIndex(sent):
    for Index,Value in enumerate(sentences):
        if(Value==sent):
            return Index
        
if __name__ == "__main__":
#        sentences = ["Nature is beautiful","I like green apples",
#                "We should protect the trees","Fruit trees provide fruits",
#                "Green apples are tasty"]
#        with open("sentences.txt") as f:
#            for line in f:
#                sentences.append(line)
    with open("Sentences.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(line.rstrip('\r\n'))
        nclusters= 30
        clusters = cluster_sentences(sentences, nclusters)
        WriteOuput = open('xyz30s.txt','w')
        for cluster in range(nclusters):
            WriteOuput.write("cluster "+str(cluster)+" :\n")
            for i,sentence in enumerate(clusters[cluster]):
                WriteOuput.write("\tExcel Id# "+str(GetIndex(sentences[sentence]))+ " Roll# "+ str(i) +" : "+ sentences[sentence]+"\n")
        WriteOuput.close()
                        
