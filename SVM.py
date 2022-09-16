#Eshaan Vora
#EshaanVora@gmail.com
#Support Vector Machine Algorithm

#This program implements a SVM algorithm to classify whether a plant is an iris-setosa or not

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Import Dataset
file_path = "Iris.csv"

#Load dataset
column_names = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
iris_data = pd.read_csv(file_path, names = column_names, header = 0)

#Convert classification variable to binary representation
iris_data = iris_data.replace({"Species": {"Not-Iris-setosa":0, "Iris-setosa":1,}})

#Set variable for species; We will be classifying this variable
y = iris_data[column_names[4]]

#Split into training/test set
X_train, X_test, y_train, y_test = train_test_split(iris_data, y, test_size=0.2)

#Create support vector classifier object w/ Linear Kernel
classifier = svm.SVC(kernel='linear')

#Train the model using the training sets
classifier.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)

#Decode predictions from binary back to string representation to print predicitions in relevant labels
decoded_predictons = []
for i in y_pred:
    if i == 0:
        decoded_predictons.append("Not-Iris-setosa")
    else:

        decoded_predictons.append("Iris-setosa")

#Print predictions; print both vector and labels
print(decoded_predictons)
print(y_pred)

#Evaluating Model Performance:
#Model Accuracy: How often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: What percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: What percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
