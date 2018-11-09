import numpy as np
import os
from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

class store():
	def __init__(self,static_dir):
		self.static_dir=static_dir

	def dump(self,file,string):
		url = os.path.join(self.static_dir,string+'.pkl')
		joblib.dump(file, url)

	def load(self,file):
		url = os.path.join(self.static_dir,str(file)+'.pkl')
		return joblib.load(url)

	def plot_confusion_matrix(self,cm, classes,cnmt_n,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues,):
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    plt.figure(figsize=(7, 7))
	    np.set_printoptions(precision=2)
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    url = os.path.join(self.static_dir,cnmt_n)
	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')
	    plt.savefig(url,dpi=100,transparent=True)
	    plt.close()

	def plot_hist(self,data,hist_n):
		url = os.path.join(self.static_dir,hist_n)
		n_labels= np.unique(np.asarray(data))
		np.set_printoptions(precision=2)
		plt.figure(figsize=(7, 7))
		plt.hist(data,bins =len(n_labels),alpha=1.0,facecolor ='#FF0000',rwidth=0.5,density=True)
		plt.xticks(n_labels)
		plt.xlabel("Labels")
		plt.ylabel("Number of instances")
		plt.savefig(url,dpi=100,transparent=True)
		plt.close()
		


