import pandas as pd

df = pd.read_csv('Netflix_Stock.csv')
df.head()
df.info()
df.describe()
df['Date'] = pd.to_datetime(df['Date'])