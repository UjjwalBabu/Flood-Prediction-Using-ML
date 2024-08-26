# Flood-Prediction-Using-ML

INTRODUCTION

In the last 17 years, world has faced more than 3000 natural disasters  which include drought , earthquake, extreme temperature, floods and landslides etc.Flooding is the most common natural disaster on the planet, affecting hundreds of millions of people and causing between 6,000 TO 18,000 fatalities every year – of which 20 percentage are in india .Flooding occurs when an extreme volume of water is carried by rivers, creeks and many others geographical features into areas where the water cannot be drained adequately.In recent years, there were many parts of countries which are prone to flood like Assam, Bihar, Kerala, Tamil nadu,.For training our model here we have considering a data set from the Kaggle which includes india statewise information about the flood and its factors for the  64 years .After the model building, our model will predicts whether flood will affect or not for the coming years based on the inputs.

IMPLEMENTATION

![image](https://github.com/user-attachments/assets/162d94d8-de03-4906-a3e1-4d62bb2d56ce)

1) Dataset Preparation
The weather data of India for 65 years is taken from Kaggle . The data is from the Indian 
Meteorological Department (IMD). The information on flood occurrence for a certain month 
and year was collected from a variety of sources including annual flood reports, newspapers, 
research papers, etc. and then merged with the weather data of IMD to create an updated dataset 
that can be found in which consists of 20544 instances. The dataset contains information for 
28 districts of India. Some of the important attributes of the dataset include Rainfall, Cloud 
Coverage, Relative Humidity, Minimum Temperature, Wind Speed, etc.

2) Data Cleaning 
Data cleaning is an essential step in Exploratory Data Analysis (EDA) that involves identifying 
and correcting errors, inconsistencies, and missing values in the data. It ensures that the data 
used for analysis is accurate, reliable, and consistent.

3) Outlier Detection
Outlier detection is an essential step in Exploratory Data Analysis (EDA) that involves 
identifying and handling extreme values or observations that are significantly different from 
the other observations in the dataset. Outliers can occur due to various reasons such as 
measurement errors, data entry errors, or natural variability in the data. The presence of outliers 
in a dataset can distort the results obtained from the analysis and affect the accuracy and 
reliability of the conclusions drawn from it. Therefore, outlier detection is necessary to identify 
and handle these extreme observations appropriately.

4) Scaling
Since the features of this dataset vary in units, range and magnitude, it was required to scale 
or normalize the data. In this work, we applied z-score normalization for this purpose. It is 
used to standardize the data by setting mean value to zero and scaling to unit variance.

5) Gridsearch CV
GridSearchCV is a function in the scikit-learn library for Python that performs an exhaustive 
search over a specified parameter grid to find the best hyperparameters for a given machine 
learning model. Hyperparameters are the model parameters that are not learned during the 
training process, but are set by the user prior to the training process. Examples of 
hyperparameters include the learning rate in a neural network, the number of trees in a random 
forest, or the regularization parameter in a linear regression model

![image](https://github.com/user-attachments/assets/5ed8011a-54fa-4b25-9ce8-a956f050edc8)

6) Data Splitting
Data splitting is the process of dividing a dataset into two or more subsets, typically a training 
set and a test set, to evaluate the performance of a machine learning model. The goal of data 
splitting is to train a model on a subset of the data and then evaluate its performance on a 
different subset of data that it has not seen before, to estimate how well the model will 
generalize to new, unseen data.

7) Model building and Prediction
Model building and prediction are critical components of machine learning that involve 
creating a mathematical model that can make predictions or decisions based on input data. 
Model building typically involves selecting an appropriate algorithm, choosing relevant 
features or variables, training the model on a dataset, and optimizing its performance through 
hyperparameter tuning or other techniques. The goal is to create a model that can accurately 
capture the patterns and relationships in the data and make predictions or decisions on new, 
unseen data.Once the model is built, prediction involves providing new data as input to the 
model and obtaining a predicted output or decision. The performance of the model is evaluated 
based on how well it predicts the output or decision on the new data.


SAMPLE CODE

#Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#GridsearchCV - for finding the best model with suitable hyper paramete
rs
model_param = {
 'svm' : {
 'model' : SVC(gamma = 'auto'),
 'params' : {
 'C' : [1,10,20,30],
 'kernel' : ['linear', 'rbf'],
 }
 },
 'rfc' : {
 'model' : RandomForestClassifier(),
 'params' : {
 'n_estimators' : [1,5,10,20,30,40],
 }
 },
 'lr' : {
 'model' : LogisticRegression(),
 'params' : {
'C' : [1,5,10,20]
 }
 },
 'dc' : {
 'model' : DecisionTreeClassifier(),
 'params' : {
 'criterion' : ['gini','entropy']
 }
 },
 'knn' : {
 'model' : KNeighborsClassifier(),
 'params' : {
 'n_neighbors' : [2,3,5,7]
 }
 }
}
score = []
for m , par in model_param.items():
 clf = GridSearchCV(par['model'], par['params'], cv = 5, return_train_
score = False)
 clf.fit(X,Y)
 score.append({
 'model' : par['model'],
 'best_score' : clf.best_score_,
 'best_params' : clf.best_params_
 })
#Building the model
mod = RandomForestClassifier(n_estimators = 40)
mod.fit(X_train, Y_train)
mod.score(X_test, Y_test)
#Prediction
Y_pred = mod.predict(X_test)
Y_pred





