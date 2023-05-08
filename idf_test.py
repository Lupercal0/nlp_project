# edit from news classifier from here: https://www.kaggle.com/code/dimitriosroussis/news-articles-classification-tf-idf-voting/notebook


import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(2023)


#dataset read in 
with open('dataset/evidence.json') as json_file:
    evidence = json.load(json_file)

with open('dataset/dev-claims.json') as json_file:
    dev_claim = json.load(json_file)

with open('dataset/train-claims.json') as json_file:
    train = json.load(json_file)

with open('dataset/test-claims-unlabelled.json') as json_file:
    test = json.load(json_file)

train_st = []