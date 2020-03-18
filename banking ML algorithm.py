import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
dataset1=pd.read_csv(r"D:\python\datasets from UCI\bank marketing\bank.csv",sep=";")
print(dataset1.head())
label_encoder = preprocessing.LabelEncoder()
dataset1['job']= label_encoder.fit_transform(dataset1['job'])
dataset1['marital']= label_encoder.fit_transform(dataset1['marital'])
dataset1['education']= label_encoder.fit_transform(dataset1['education'])
dataset1['default']= label_encoder.fit_transform(dataset1['default'])
dataset1['housing']= label_encoder.fit_transform(dataset1['housing'])
dataset1['loan']= label_encoder.fit_transform(dataset1['loan'])
dataset1['contact']= label_encoder.fit_transform(dataset1['contact'])
dataset1['month']= label_encoder.fit_transform(dataset1['month'])
dataset1['poutcome']= label_encoder.fit_transform(dataset1['poutcome'])
dataset1['y']= label_encoder.fit_transform(dataset1['y'])
ar=dataset1.values
X=ar[:,0:16]
Y=ar[:,16]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4,
                                                    random_state=1)
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
predicted=clf.predict(X_test)
print(confusion_matrix(y_test, predicted))
print ('Accuracy Score :',accuracy_score(y_test, predicted))
#dataset1.info()