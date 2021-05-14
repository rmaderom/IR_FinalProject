#goal is to first use sentiment lexicon on data to split the data into two bins
#one being a Strong positive bin and the other being a strong negative bin
#will do this by using a sentiment lexicon where we score each comment

#aftrwards we want to train a classifer over these two bins

#goal is to then use the classifer to try and classify comments from a web robot from reddit
#and classify peoples comments on these

from os import read
import numpy as np
import pandas as pd
import sys
import csv
from numpy.linalg import norm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
import re
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
#might need to run pip install -U textblob
#nltk.download('vader_lexiconcle') download once

def read_docs():
    post_data = pd.read_csv("bitcoin_post_comments.csv",index_col = 0).dropna() #fix with dropna
    #extract only the comment_body
    docs = post_data['comment_body'].tolist()
    # train_docs = docs[:round(len(docs)*0.9)]
    # test_docs = docs[round(len(docs)*0.9):]
    train_docs = docs[:round(len(docs) - 100)]
    test_docs = docs[round(len(docs) - 100):]
    

    return train_docs, test_docs

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

def experiment(train_docs, test_docs):
    # #using own lexicon
    # positive_train, negative_train, _ = classify_lexicon(train_docs)
    # positive_test, negative_test, order = classify_lexicon(test_docs)
    # print("Using Own Lexicon")
    # train_and_test(positive_train, negative_train, positive_test, negative_test, order)

    #using VADER for each word
    # positive_train, negative_train, _ = classify_vader(train_docs)
    # positive_test, negative_test, order = classify_vader(test_docs)
    # print("Using VADER for each word")
    # train_and_test(positive_train, negative_train, positive_test, negative_test, order)

    #using VADER for whole comments
    # print("Using VADER for whole comments")
    # compute_vader(test_docs)

    # #using TextBlob
    # positive_train, negative_train, _ = classify_vader(train_docs)
    # positive_test, negative_test, order = classify_vader(test_docs)
    # print("Using TextBlob for each word")
    # train_and_test(positive_train, negative_train, positive_test, negative_test, order)

    print("Using TextBlob for whole sentences")
    compute_textblob(test_docs)





def train_and_test(positive_train, negative_train, positive_test, negative_test, order):
    #train on both bins
    V_profile1 = dict()
    count1 = len(positive_train)
    V_profile2 = dict()
    count2 = len(negative_train)

    for i in range(len(positive_train)):
        keys = list(positive_train[i].keys())
        for key in keys:
            V_profile1[key] = positive_train[i].get(key,0) + V_profile1.get(key, 0)
        
    for i in range(len(negative_train)):
        keys = list(negative_train[i].keys())
        for key in keys:
            V_profile2[key] = negative_train[i].get(key,0) + V_profile2.get(key,0)

    for key in V_profile1.keys():
        V_profile1[key] = V_profile1[key]/count1
    for key in V_profile2.keys():
        V_profile2[key] = V_profile2[key]/count2
    

    positive_correct = 0
    negative_correct = 0
    for doc in positive_test:
        sim1 = cosine_sim(doc,V_profile1)
        sim2 = cosine_sim(doc,V_profile2)

        if(sim1 >= sim2):
            positive_correct = positive_correct + 1
        
    for doc in negative_test:
        sim1 = cosine_sim(doc,V_profile1)
        sim2 = cosine_sim(doc,V_profile2)

        if(sim1 <= sim2):
            negative_correct = negative_correct + 1
    
    print("Accuracy of Positive Comments: ", positive_correct/len(positive_test))
    print("Accuracy of Negative Comments: ", negative_correct/len(negative_test))
    print("Total Accuracy: ", (positive_correct + negative_correct) / (len(positive_test) + len(negative_test)))


    file = open("test_class.tsv")
    read_tsv = csv.reader(file)
    labels = list()
    predicted_sentiment = list()
    for row in read_tsv:
        labels.append(int(row[0]))
    correct = 0
    docs = list() #join together the lists to match the test_class.tsv file
    p_count = 0
    n_count = 0
    for i in range(len(order)):
        if order[i] == 1:
            docs.append(positive_test[p_count])
            p_count = p_count + 1
        else:
            docs.append(negative_test[n_count])
            n_count = n_count + 1

    for i in range(len(docs)):
        sim1 = cosine_sim(docs[i],V_profile1)
        sim2 = cosine_sim(docs[i],V_profile2)
        if sim1 >= sim2:
            predicted_sentiment.append(1)
        else:
            predicted_sentiment.append(-1)
   
        if(labels[i] == predicted_sentiment[i]):
            correct = correct + 1
    
    print("Total Accuracy Based on Labeled: ", correct/len(docs))       

    return


def dictdot(x, y):
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def classify_textblob(docs):
    positive_bin = list()
    negative_bin = list()
    order = list()
    for doc in docs:
        vec = dict()
        sentiment = 0
        for word in doc.split():
            wiki = TextBlob(doc)
            score = wiki.sentiment.polarity
            vec[word] = score
            sentiment = sentiment + score
        
        if sentiment > 0:
            positive_bin.append(vec)
            order.append(1)
        else:
            negative_bin.append(vec)
            order.append(-1)
    
    return positive_bin, negative_bin, order



def classify_vader(docs):
    sid = SIA()
    positive_bin = list()
    negative_bin = list()
    order = list()
    for doc in docs:
        vec = dict()
        sentiment = 0
        for word in doc.split():
            score = sid.polarity_scores(word)['compound']
            vec[word] = score
            sentiment = sentiment + score
        
        if sentiment > 0:
            positive_bin.append(vec)
            order.append(1)
        else:
            negative_bin.append(vec)
            order.append(-1)

    return positive_bin, negative_bin, order

def compute_vader(docs):
    sid = SIA()
    file = open("test_class.tsv")
    read_tsv = csv.reader(file)
    labels = list()
    predicted_sentiment = list()
    for row in read_tsv:
        labels.append(int(row[0]))
    correct = 0
    predicted_labels = list()
    for i in range(len(docs)):
        score = sid.polarity_scores(docs[i])['compound']
        if score >=0:
            predicted_sentiment.append(1)
        else:
            predicted_sentiment.append(-1)
        
        if predicted_sentiment[i] == labels[i]:
            correct = correct + 1

    print("Total Accuracy Based on Labeled: ", correct/len(docs))

def compute_textblob(docs):
    file = open("test_class.tsv")
    read_tsv = csv.reader(file)
    labels = list()
    predicted_sentiment = list()
    for row in read_tsv:
        labels.append(int(row[0]))
    correct = 0
    predicted_labels = list()
    for i in range(len(docs)):
        wiki = TextBlob(docs[i])
        score = wiki.sentiment.polarity
        if score >=0:
            predicted_sentiment.append(1)
        else:
            predicted_sentiment.append(-1)
        
        if predicted_sentiment[i] == labels[i]:
            correct = correct + 1

    print("Total Accuracy Based on Labeled: ", correct/len(docs))  

        
        


def classify_lexicon(docs):
    #option 1 is using a sentiment lexicon. We are using the Bitcoin.tsv which is a domain-specific lexicon
    #from reddit obtaiend from https://nlp.stanford.edu/projects/socialsent/
    #the lexicon contains the <word> <mean_sentiment> <std_sentiment>
    lexicon_file = open("Bitcoin.tsv")
    read_tsv = csv.reader(lexicon_file, delimiter="\t")
    #create dictionary of word and mean_sentiment
    lexicon = dict()
    for row in read_tsv:
        lexicon[row[0]] = float(row[1])
    
    
    #split docs into two bins
    #each bin contains a list dictionaries
    positive_bin = list()
    negative_bin = list()
    order = list()
    for doc in docs:
        #iterate through the doc and compute a vector
        vec = dict()
        sentiment = 0

        for word in doc.split():
            if word in lexicon:    
                vec[word] = lexicon[word]
                sentiment = sentiment + lexicon[word]
            else:
                vec[word] = 0
        if sentiment > 0:
            positive_bin.append(vec)
            order.append(1)
        else:
            negative_bin.append(vec)
            order.append(-1)

    return positive_bin, negative_bin, order

if __name__ == '__main__':
    train_docs, test_docs = read_docs()
    experiment(train_docs, test_docs)






