import pandas as pd
import numpy as np
import math ,datetime
from sklearn import preprocessing, model_selection as cross_validation , svm
from sklearn.linear_model import LinearRegression
import time
import calendar

import quandl
quandl_api_key = "htqjj94SMXxLb_7nErx_"
quandl.ApiConfig.api_key = quandl_api_key



import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
dff =pd.read_csv("daily_BSE_ICICIBANK.csv")
# dff.set_index(df['timestamp'])
# print(dff.set_index(df['timestamp']).head())
newdf=dff[['timestamp','open',  'high',  'low',  'close', 'volume']]
newdf['Date']=newdf['timestamp']

newdf = newdf[['Date','open',  'high',  'low',  'close', 'volume']]
newdf["Date"]= pd.to_datetime(newdf["Date"])
newdf.set_index('Date',inplace=True)


# print(newdf.head())

new_data = newdf.rename(columns={"open": "Adj. Open",
                     "high": "Adj. High",
                     "low":"Adj. Low",
                      "close":"Adj. Close",
                      "volume":"Adj. Volume"})
# print(new_data.head())

df = new_data
print(df.head())
df = df.sort_index(ascending=True)
print(df.tail())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-999999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)


x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)  # multithreading
# clf = svm.SVR(kernel='poly')
clf.fit(x_train, y_train)

# pickle_in = open("linearregression.pickle",'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_lately)

# print(accuracy)
print(forecast_set, accuracy)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)



last_unix = last_date.timestamp()


one_day = 86400
next_unix = last_unix + one_day
#
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
#
