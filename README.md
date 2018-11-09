
# ML App Using Flask

This is an ML web application using Flask. It allows uploading .txt file which is then parsed and trained using the classifier the user inputs.I am hosting it on Google App Engine thorugh GCP. I will update the GCP version soon.

## Training

The web application allows users to input text files with training data of the form 'label_1 \t text_1 \n' and takes user input to choose the base classifier model from Logistic Regression, Naive Bayes and Decision Tree. 

The text data is converted to features using TF-IDF vectorization. A truncated Singular Valued Decomposition(SVD) algorithm is applied to the features for dimensionality reduction. The features are then normalized and balanced of any class imbalances using the SMOTE algorithm. The labels are encoded using a label binarizer. The base classifier chosen is coupled up with an ensemble algorithm, Bagged Decision Tree for performance. The hyperparameters for the base estimator are found using gridsearch. The model is fit on the data and  saved for reteieval for prediction.

The training results will be displayed once the model has been fit and prediction on the validation set is complete. The confusion matrix and a histogram plot of the class distribution of the entire are displayed.

## Testing

A text file can be uploaded from the application. The features are extracted and processed as in the training process. The fitted classied is unpickled from the stored folder (static) and labels of the features are predicted. The predicted classes are decoded and a histogram plot of the class distribution of the predicted labels is displayed.The predicted labels and the data it corresponds to can be downloaded from the application.


## File Hierarchy

### App
1. main.py - Main Python function which runs the Flask app.
2. model.py - Contains classes for handling the chosen classifier models. 
3. store.py - Contains class to Pickle variables and plot histogram and confucion matrix.
4. utils.py - Contains helper functions for all files.
5. templates - Contains HTML files for the Flask app.
6. static - Contains CSS files and pickled files.
7. uploads - Contains user uploaded data for retrival during training and testing.
8. requiremnets.txt - Setup file for Flask app environment
9. app.yaml - Setup file for Google App Engine

## Setup 

Clone the respository into a folder and install the required libraries from requirements.txt

```
cd path_to_cloned_repo
pip install -r requirements.txt
```
Open main.py and edit the port number if needed.

```
app.run(host="127.0.0.1",port=8080,debug=True)
```
## Run

Run main.py and paste 'http://127.0.0.1:8080' in browser to view the app.
