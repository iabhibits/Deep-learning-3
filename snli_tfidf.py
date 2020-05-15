#!/usr/bin/env python
# coding: utf-8
# author: Abhishek

import numpy as np
import pandas as pd
import nltk
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
# try:
# 	nltk.data.find('tokenizers/punkt')
# except LookupError:
# 	nltk.download('punkt')

try:
	nltk.data.find('tokenizers/stopwords')
except LookupError:
	nltk.download('stopwords')

# try:
# 	nltk.data.find('tokenizers/wordnet')
# except LookupError:
# 	nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
from joblib import dump, load
stopword = set(stopwords.words('english'))

def preprocessing_data(filename):
	dataset = pd.read_csv(filename,sep='\t')
	print(dataset.columns)
	print("Training: No. of Rows {} and no. of Columns {}\n".format(dataset.shape[0],dataset.shape[1]))
	#print(data['sentence1'].isnull().sum())
	sent1 = dataset['sentence1'].astype(str)
	sent2 = dataset['sentence2'].astype(str)
	label = dataset['label1'].astype(str)
	print("Name of Classes in labels is :{}\n".format(np.unique(label)))
	classes = ['contradiction','neutral','entailment']
	label = label.apply(classes.index)
	sent = sent1 + sent2
	return sent, label

def preprocess_data(sent):
	sentences = []
	cnt = 0
	stemmer = WordNetLemmatizer()
	for i in sent:
		try:
			a = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", i) # 
			a = re.sub(r'\d+', '', a) # remove numbers from the string
			a = a.translate(str.maketrans("","", string.punctuation))
			a = a.lower()
			a = a.split()

			a = [stemmer.lemmatize(word) for word in a]
			a = ' '.join(a)
			sentences.append(a)
		except:
			print("Error occured during extracting!, Error no. {}\n".format(cnt))
			print("Sentence 1","\n")
			print(j,"\n")
			print("Error occured at index {}\n".format(cnt))
	return np.array(sentences)

def tf_idf(train_sent,test_sent):
	vectorizer = TfidfVectorizer()
	tr_sent = vectorizer.fit_transform(train_sent)
	te_sent = vectorizer.transform(test_sent)
	print("tf-idf shape is : {}\n".format(tr_sent.shape))
	return tr_sent,te_sent

def train_model(data,label):
	lr = LogisticRegression(random_state=1)
	clf = lr.fit(data,label)
	dump(clf, 'tf_idfCLF.joblib')
	return clf

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--type', type=str, default='test', help = "Specify whether you want to train or test the model \n")
	parser.add_argument('--model_name', type=str, default='./model/tf_idfCLF.joblib', help = "Pre-trained model name with full path \n")
	args = parser.parse_args()
	if args.type == "train" :
		train_sent,tr_label = preprocessing_data("./snli_1.0/snli_1.0_train.txt")
		test_sent,te_label = preprocessing_data("./snli_1.0/snli_1.0_dev.txt")

		tr_sent,te_sent = tf_idf(train_sent,test_sent)
		#te_sent = tf_idf(test_sent,tfidfconverter,vectorizer)
		#vectorizer.get_feature_names()[-10:]
		#vectorizer.get_feature_names()[np.argmax(tr_sent[0])]
		model = train_lr(tr_sent)
		prediction = model.predict(te_sent)
		accuracy = accuracy_score(prediction,te_label)
		print("Accuracy of the trained model on test set is {}\n".format(accuracy * 100))
	elif args.type == "test":
		clf = load(args.model_name)
		train_sent,tr_label = preprocessing_data("./snli_1.0/snli_1.0_train.txt")
		test_sent,te_label = preprocessing_data("./snli_1.0/snli_1.0_dev.txt")
		tr_sent,te_sent = tf_idf(train_sent,test_sent)
		prediction = clf.predict(te_sent)
		accuracy = accuracy_score(prediction,te_label)
		print("Accuracy of the trained model on test set is {}\n".format(accuracy * 100))



if __name__ == '__main__':
	main()
