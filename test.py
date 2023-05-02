import json
from eval import main
import numpy as np
import nltk
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#import os
from sklearn.datasets import load_files
import autokeras as ak

def label_t(label):
    if label == 'SUPPORTS':
        t = 0
    elif label == 'REFUTES':
        t = 1
    elif label == 'NOT_ENOUGH_INFO':
        t = 2
    elif label == 'DISPUTED':
        t = 3
    else:
        return None
    return t


with open('dataset/evidence.json') as json_file:
    evidence = json.load(json_file)

with open('dataset/dev-claims.json') as json_file:
    dev_claim = json.load(json_file)

with open('dataset/dev-claims-baseline.json') as json_file:
    dev_claims_base = json.load(json_file)

with open('dataset/train-claims.json') as json_file:
    train = json.load(json_file)

with open('dataset/test-claims-unlabelled.json') as json_file:
    test = json.load(json_file)

claims_test = []
amount = []
for key in test.keys():
    test[key]['claim_text'] = test[key]['claim_text'].lower()
    sentence = test[key]['claim_text'].split()
    word = [i for i in (sentence) if i not in set(stopwords.words('english'))] 
    #print(word)
    amount += word
    word = ' '.join(word)
    claims_test.append(word)

word_amount = list(set(amount))
#print(len(word_amount))
vectorizer = CountVectorizer(max_features = len(amount))
bow_claim = vectorizer.fit_transform(claims_test).toarray()


responds = []
counter = 0
w_claim = []
word_amount = []
label = []
for key in train.keys():
    
    #rep = label_t(train[key]['claim_label'])
    #if rep == None:
        #print("error response")
        #break
    #else:
        #responds.append(rep)
    """
        counter +=1
        if rep in responds.keys():
            responds[rep] +=1
        else:
            responds[rep] =1
        """
    """
    train[key]['claim_text'] = train[key]['claim_text'].lower()
    sentence = train[key]['claim_text'].split()
    word = [i for i in (sentence) if i not in set(stopwords.words('english'))] 
    #print(word)
    word_amount += word
    word = ' '.join(word)
    w_claim.append(word)
    """
    sentence = train[key]['claim_label'].split()
    word = [i for i in (sentence) if i not in set(stopwords.words('english'))] 
    #print(word)
    #word_amount += word
    word = ' '.join(word) 
    responds.append(word)
    #responds.append(rep)

#word_amount = list(set(word_amount))
#print(len(word_amount))
#vectorizer = CountVectorizer(max_features = len(word_amount))
#bow_claim = vectorizer.fit_transform(w_claim).toarray()

#vectorizer = CountVectorizer()
#responds = vectorizer.fit_transform(responds).toarray()

sentence = []
for key in train.keys():
    sentence.append(train[key]['claim_text'])



x_train = np.array(sentence)
y_train = np.array(responds)

#print(y_train)


print(x_train.shape)  
print(y_train.shape)  
clf = ak.TextClassifier(overwrite=True, max_trials=5)  
clf.fit(x_train, y_train, epochs=5)
#clf.save()


test_claim = []
for key in test.keys():
    
    
    test[key]['claim_text'] = test[key]['claim_text'].lower()
    sentence = test[key]['claim_text'].split()
    word = [i for i in (sentence) if i not in set(stopwords.words('english'))] 
    #print(word)
    word = ' '.join(word)
    test_claim.append(word)

test_input = np.array(test_claim)

prediction = clf.predict(test_input)

print(prediction)