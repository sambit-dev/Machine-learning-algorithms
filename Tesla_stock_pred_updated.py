import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df= pd.read_csv("D:\python\Datasets for ML\Tesla.csv")
print(df.head())
df.info()
df['Volmume']=df['Volume']/100000
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'Date'.
df['Date'] = label_encoder.fit_transform(df['Date'])
print(df.head())
x=df[['Date','Open','High','Low','Volume']]
y=df['Adj Close']

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.25)

regr = LinearRegression()

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
test_set_r2 = r2_score(y_test, y_pred)
print(test_set_r2)