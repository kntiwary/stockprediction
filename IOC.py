import requests
import pandas as pd

r = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BSE:IOC&apikey=Z2RJFYNYL3R2IDQT&outputsize=full')
x = r.json()
# print(x)
df = pd.DataFrame(x['Time Series (Daily)'])
# dfff.set_index('')
df =df.transpose().head()
nw = pd.DataFrame()
new_data = df.rename(columns={"1. open": "Adj. Open",
                     "2. high": "Adj. High",
                     "3. low":"Adj. Low",
                      "4. close":"Adj. Close",
                      "5. volume":"Adj. Volume"})
print(new_data)
