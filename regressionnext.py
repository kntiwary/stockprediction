import arrow
import pandas as pd
import numpy as np
from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LinearRegression
import math,datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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
# print(forcast_out)
#
df['label']= df[forcast_col].shift(-forcast_out)

#
# print(df.head())
#<<<<<<<<<<<<----------------------------Train and test------------------>>>>>>>>>>


# Drop everything except Lable
# print(df.drop('label',axis=1))
X = np.array(df.drop('label',axis=1))
# print(X)
X = preprocessing.scale(X)
X_lately =X[-forcast_out:]
X=X[:-forcast_out] # stuff that we are going to predict again so we have x , we need to figure out m and b


y = np.array(df['label'])

#scale X before feeding it to through classifier, only for real time data , helps in testing and traing and also processing time we can skip these for high frequcy trading

df.dropna(inplace=True)
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

# print(accuracy)


#--------------->>   prediction in future--------------
#predict based on x
forcast_set = clf.predict(X_lately) # we can either pass a single value of an array

print(forcast_set,accuracy,forcast_out)
df['Forcast '] =np.nan

last_date = df.iloc[-1].name
# last_unix = last_date.timestamp()
last_unix = arrow.get(last_date).timestamp
one_day = 86400
next_unix = last_unix + one_day # its like next day

#
# for i in forcast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)

#     next_unix += next_date
#     df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] +[i]
#
#
# df['close'].plot()
# df['Forcast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
#

