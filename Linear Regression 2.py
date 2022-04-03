import pandas as pd
import math, datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle

style.use('ggplot')
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
df = pd.read_csv("C:/Users/oguzh/OneDrive/Masaüstü/GOOG.csv")
df = df[['open','high','low','close','volume']]
df['hl_pct'] = (df['high'] - df['close']) / df['close'] * 100
df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100
df = df[['close','hl_pct','pct_change','volume']]

forecast_col = 'close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)
df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()
