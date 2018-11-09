import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import GridSearchCV


class model():
	def __init__(self,clf,train_x,train_y,valid_x,valid_y):
		self.clf = clf
		self.train_x = train_x
		self.train_y = train_y
		self.valid_x = valid_x
		self.valid_y = valid_y


	def param_optim(self,classifier,param_dist):

		gd_sr = GridSearchCV(estimator=classifier,
							param_grid= param_dist,
							scoring=None,
							cv=3,
							n_jobs= -1,
							verbose=0)
		gd_sr.fit(self.train_x,self.train_y)

		return gd_sr.best_params_

	def model_selection(self):
		n_trees =100
		seed = 7
		if self.clf =='LR':
			mod = linear_model.LogisticRegression(penalty='l1', C=c,n_jobs =-1,solver ='saga',multi_class='auto')
			param_dist={'C' : np.logspace(-2,7,5)}
			best =  param_optim(mod,param_dist)
			c_opt = best['C']
			opt_mod = linear_model.LogisticRegression(penalty='l1', C=c_opt,n_jobs =-1,solver ='saga',multi_class='auto')
			
		elif self.clf =='NV':
			mod = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
			param_dist ={'alpha':np.logspace(-2,7,5)}
			best =  param_optim(mod,param_dist)
			c_opt = best['alpha']
			opt_mod = MultinomialNB(alpha=c_opt, class_prior=None, fit_prior=True)

		elif self.clf =='DT':
			mod = DecisionTreeClassifier(max_features=10, random_state=seed)
			param_dist ={'max_features':['sqrt','log2']}
			best =  self.param_optim(mod,param_dist)
			c_opt = best['max_features']
			opt_mod =DecisionTreeClassifier(max_features=c_opt, random_state=seed)

		comb_mod = BaggingClassifier(base_estimator= opt_mod, n_estimators=n_trees, random_state=seed)

		return comb_mod


	def build_model(self,comb_mod):
		return comb_mod.fit(self.train_x,self.train_y)
	

class predict_model():
	def __init__(self,valid_x):
		self.valid_x = valid_x

	def predict_model(self,comb_mod):
		return comb_mod.predict(self.valid_x)

class stats():
	def __init__(self,pred,valid_y):
		self.pred =pred
		self.valid_y= valid_y

	def train_stats(self):
		acc= np.mean(self.pred==self.valid_y)*100
		f1 = f1_score(self.pred,self.valid_y,average=None)
		return acc,f1

	def cnf_mtx(self):
		return confusion_matrix(self.valid_y,self.pred)




	







	