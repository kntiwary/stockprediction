import pandas as pd
import numpy as np
from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LinearRegression
import math


#preprocessing---->for sclaling data usuallu done on feature, helps us with accuracy and processig speed
#model_selection/cross_validation ----> creat and test training sample /sample/ good for seperation of data and stats

# url = https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BSE:TCS&apikey=Z2RJFYNYL3R2IDQT@datatype=csv
# url = https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ICICIBANK:TCS&apikey=Z2RJFYNYL3R2IDQT@datatype=csv

df =pd.read_csv("daily_BSE_ICICIBANK.csv")
# print(df.count())
# print(df.head(5))

# df = df[['timestamp','open',  'high',  'low',  'close', 'volume']]
df = df[['open',  'high',  'low',  'close', 'volume']]

df['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0

df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

# print(df.head(5))

df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]
# print(df.head(10))
forcast_col = 'close'
df.fillna(-99999,inplace=True)
forcast_out = int(math.ceil(0.01*len(df)))
print(forcast_out)
#
df['label']= df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)
#
# print(df.head())
#<<<<<<<<<<<<----------------------------Train and test------------------>>>>>>>>>>


# Drop everything except Lable
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])


#scale X before feeding it to through classifier, only for real time data , helps in testing and traing and also processing time we can skip these for high frequcy trading
X = preprocessing.scale(X)
y = np.array(df['label'])
# print(len(X),len(y))

# This is going to take our features and label and shuffle them up keeping x and y connected and outputs X_train,y _train we use to fit classifier

X_train, X_test ,y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)


# lets define a classifier and this could be used to predict future

# clf = LinearRegression()

# for multiple thread to make it fast or -1 for max from processor availble
clf = LinearRegression(n_jobs=-1)

# lets try to use swithc to diferent algorithm-- svm

# clf = svm.SVR()
# clf = svm.SVR(kernel='poly')
# and fit features and labels


# fit is synonymous train with and score  is with test
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)


#--------------->>   prediction in future--------------


