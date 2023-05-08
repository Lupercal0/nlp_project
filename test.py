import json
#from eval import main
import numpy as np
import nltk
import random
#import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#import os
from sklearn.datasets import load_files
import autokeras as ak
#require cuda v11.4 and cuDNN v8.2
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from keras.utils import plot_model

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

#with open('dataset/dev-claims-baseline.json') as json_file:
    #dev_claims_base = json.load(json_file)

with open('dataset/train-claims.json') as json_file:
    train = json.load(json_file)

with open('dataset/test-claims-unlabelled.json') as json_file:
    test = json.load(json_file)

"""
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
"""
responds = []
for i in train.keys():
    responds.append(train[i]['claim_label'])
sens = []
for i in dev_claim.keys():
    sens.append(dev_claim[i]['claim_label'])

tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")


max_len = 0
def gpt_vectorizer(input, tokenizer):
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    responds = []
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #max_len = 0
    for key in input.keys():
        claim = input[key]['claim_text'].lower()
        #if there is a sufficient way finding evidence, active two row below
        if 'evidences' in input[key].keys():
            for evi in input[key]["evidences"]:
                claim = claim + evidence[evi]
        res = tokenizer(claim, padding=False, truncation=True)['input_ids']
        res = map(str, res)
        #if len(res) > max_len:
            #max_len = len(res)
        res = ' '.join(res)
        responds.append(res)
    return responds

#word_amount = list(set(word_amount))
#print(len(word_amount))
#vectorizer = CountVectorizer(max_features = len(word_amount))
#bow_claim = vectorizer.fit_transform(w_claim).toarray()

#vectorizer = CountVectorizer()
#responds = vectorizer.fit_transform(responds).toarray()

#sentences = []
#for key in train.keys():
    #sentences.append(train[key]['claim_text'])
sentences = gpt_vectorizer(train, tokenizer_gpt)
sen = gpt_vectorizer(dev_claim, tokenizer_gpt)
#print(sentences)
#x_train = sentences
x_train = np.array(sentences)
x_valid = np.array(sen)
y_train = np.array(responds)
y_valid = np.array(sens)
#print(x_train)
#print(y_train)


test_input = np.array(gpt_vectorizer(test, tokenizer_gpt))

#print(x_train.shape)  
#print(y_train.shape)  
clf = ak.TextClassifier(max_trials=3, multi_label=False,overwrite=True)  
clf.fit(x_train, y_train, epochs=5,validation_data=(x_valid, y_valid))
#model = clf.export_model()
#model.summary()
#plot_model(model)
#clf.fit(x_train, y_train, epochs=10, validation_split = 0.3)
#clf.save()
prediction = clf.predict(test_input)

"""
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

#print(prediction)
"""


"""
w_evi = []
for key in train.keys():
    word = []
    for evi in train[key]['evidences']:
        word.append(evi[9:])
    word = ' '.join(word)
    w_evi.append(word)
    #print(word)
    #sentence = train[key]['claim_label'].split()
    #word = [i for i in (sentence) if i not in set(stopwords.words('english'))] 
    #print(word)
    #word_amount += word
    #word = ' '.join(word)
    #responds.append(word)



#vectorizer = CountVectorizer()
#evidence_train = vectorizer.fit_transform(w_evi).toarray()

x_train = np.array(sentences)
y_train = np.array(w_evi)

clf = ak.TextClassifier(overwrite=True, max_trials=5)  
clf.fit(x_train, y_train, epochs=5)

prediction = clf.predict(test_input)
"""

#print(prediction)

np.save('test', prediction)