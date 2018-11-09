import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import sklearn
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import itertools
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing,decomposition,linear_model,naive_bayes
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def file_parser(in_file):
	text =[]
	label =[]
	with open(in_file) as file:
	    for line in file:
	        fields= line.split("\t")
	        label.append(fields[0])
	        txt = fields[-1]
	        text.append(txt.split('\n')[0])
	data =pd.DataFrame({'text':text,
						'label':label})
	data.fillna('None')
	return shuffle(data)

def file_parser_test(in_file):
	text =[]
	with open(in_file) as file:
	    for line in file:
	        fields= line.split("\n")
	        text.append(fields[0])
	data =pd.DataFrame({'text':text
							})
	data.fillna('None')
	return shuffle(data)

def feature_extractor(text,n_feat):
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=n_feat)
	tfidf_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{2,}', max_features=n_feat,ngram_range=(2,3))
	tfidf_vect.fit(text)
	tfidf_char.fit(text)
	# r = pd.DataFrame(tfidf_vect.transform(text).todense())
	# s = pd.DataFrame(tfidf_char.transform(text).todense())
	# t = pd.concat([r,s],axis=1)
	return tfidf_char.transform(text)

def label_encoder(labels,inv = False,encoder=None):
	sh = labels.shape
	if inv ==False:
		if encoder ==None:
			n_labels = np.unique(np.asarray(labels))
			label_encoder = sklearn.preprocessing.LabelEncoder()
			label_encoder.fit(n_labels)
			encoded = label_encoder.transform(labels)
		elif encoder != None:
			label_encoder = encoder
			encoded = encoder.transform(labels)
		return encoded,label_encoder

	elif inv==True:
		decoded = encoder.inverse_transform(labels)
		return decoded

def preprocess_features(features,n_feat):
	pca = decomposition.TruncatedSVD(n_components =n_feat)
	pca.fit(features)
	feat = preprocessing.normalize(pca.transform(features))
	return feat

def balance_data(data_x,data_y):
	sm = SMOTE(random_state=1)
	x_data,y_data = sm.fit_sample(data_x,data_y)
	return shuffle(x_data,y_data)

def train_valid(data_x,data_y,test_size):
	sh = len(data_y)
	ntr = int(sh*(1-test_size))
	train_x = data_x[:ntr,:]
	train_y = data_y[:ntr]
	valid_x = data_x[ntr:,:]
	valid_y = data_y[ntr:]
	return train_x,train_y,valid_x,valid_y












