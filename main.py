# Exploratory Data Analysis Packages
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import itertools
import utils
from sklearn.utils import shuffle
from model import model,predict_model,stats
from store import store


# API packages
import os
import io
import cloudstorage as gcs
import logging
from flask import Flask, flash, request, redirect, url_for, render_template,send_from_directory
from werkzeug.utils import secure_filename



# root = os.path.dirname(os.path.realpath(__file__))
# upload_path = os.path.join(root,'Uploads')
allowed_ext = set(['txt'])
allowed_clf = set(['LR','NB','DT'])

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config['upload_path'] = 'uploads'
app.config['static_path'] = 'static'

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['upload_path'],

                               filename)
@app.route('/uploads/')

def download_pred_file():
	if request.method =='POST' and 'Download':
		filename = 'pred.csv'
		return send_from_directory(app.config['upload_path'],
                               filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_ext

def allowed_classifier(cl):
	ncl = "".join(cl.upper().split(" "))
	if ncl in allowed_clf:
			return ncl


@app.route('/', methods=['GET','POST'])
def upload_train_file():
    if request.method == 'POST' and 'train':
        # check if the post request has the file part
        if 'train' not in request.files:
            flash('Please select a file to upload')
            return redirect(url_for('index'))
        file = request.files['train']
       
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):

        	file.filename ="data.txt"
        	filename = secure_filename(file.filename)
        	file.save(os.path.join(app.config['upload_path'],filename))
        	flash('File uploaded successfully !')
        	return redirect(url_for('index'))
            
        else:
       		flash('Format not supported. Please upload a file in .txt format.')
        	return redirect(url_for('index'))  
        # redirect(url_for('uploaded_file',filename=filename))
    return

@app.route('/index', methods=['GET','POST'])
def upload_test_file():
    if request.method == 'POST' and 'test':
        # check if the post request has the file part
        if 'test' not in request.files:
            flash('Please select a file to upload')
            return redirect(url_for('index'))
        file = request.files['test']
       
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
        	file.filename ="test.txt"
        	filename = secure_filename(file.filename)
        	file.save(os.path.join(app.config['upload_path'],filename))
        	flash('File uploaded successfully !')
        	return redirect(url_for('index'))
            
        else:
       		flash('Format not supported. Please upload a file in .txt format.')
        	return redirect(url_for('index'))  
        # redirect(url_for('uploaded_file',filename=filename))
    return

@app.route('/train_result', methods=['GET','POST'])

def train():

	clf = request.form['train']
	if allowed_classifier(clf):
		string = str('train')
		hist_n = string+"hist.jpeg"
		cnmt_n = string+"cnmt.jpeg"
		pkl_hnd = store(app.config['static_path'])

		# Feature extraction
		data = utils.file_parser(os.path.join(app.config['upload_path'],"data.txt"))
		features = utils.feature_extractor(data['text'],5000).todense()
		sh = data.shape

		# Preprocessing features and labels
		data_x = utils.preprocess_features(features,2500)
		data_y,enc = utils.label_encoder(data['label'],False,None)
		pkl_hnd.dump(enc,'enc') 								# storing encoder


		# Splitting data into training set and validation set 
		train_x,train_y,valid_x,valid_y= utils.train_valid(data_x,data_y,0.2)

		#Balancing data with SMOTE
		text,label =utils.balance_data(train_x,train_y)
		
		# Selecting model and tuning hyperparameters
		tr =model(clf,text[:sh[0],:],label[:sh[0]],valid_x,valid_y)
		comb_mod = tr.model_selection()

		# Fitting model and predicting
		mod = tr.build_model(comb_mod)
		pkl_hnd.dump(mod ,'model') 						 # storing the model
		pr = predict_model(valid_x) 							  
		pred = pr.predict_model(mod)

		#Training Statistics
		st = stats(pred,valid_y)
		acc,f1 =st.train_stats()

		#Plotting histogram and confusion matrix
		pkl_hnd.plot_hist(data['label'],hist_n)
		n_labels = np.unique(np.asarray(data['label']))
		pkl_hnd.dump(n_labels,'n_labels')							# storing labels
		cnf_matrix = st.cnf_mtx()
		pkl_hnd.plot_confusion_matrix(cnf_matrix, n_labels,cnmt_n,
	                          normalize=True,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues,)

		return render_template("train_result.html",accuracy = acc, 
								img_hist=url_for(app.config['static_path'],filename=hist_n),
								img_cfmt =url_for(app.config['static_path'],filename = cnmt_n),f1=f1)
	else:
       		flash('Please enter a valid classifier')
        	return redirect(url_for('index')) 


@app.route('/predict_result', methods=['GET','POST'])
def predict():
	string = str('test')
	hist_pred_n = string+"hist_pred.jpeg"

	# Loading from .pkl files 
	pkl_hnd = store(app.config['static_path'])
	clf = pkl_hnd.load('model')
	n_labels = pkl_hnd.load('n_labels')
	enc = pkl_hnd.load('enc')

	# Feature extraction
	data = utils.file_parser_test(os.path.join(app.config['upload_path'],"test.txt"))
	features = utils.feature_extractor(data['text'],5000)

	# Preprocessing features 
	data_x = utils.preprocess_features(features,2500)

	# Predicting
	pr = predict_model(data_x)	
	pred_enc=pr.predict_model(clf)

	# Decoding the encoded prediction
	pred = utils.label_encoder(pred_enc,True,enc)
	pkl_hnd.save_pred(data_x,pred)
	# Saving predicted value and data into .csv file

	#Plotting histogram of prediction
	pkl_hnd.plot_hist(pred,hist_pred_n)

	return render_template("predict_result.html",img_hist_pred =url_for(app.config['static_path'],filename = hist_pred_n),
						)


if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)