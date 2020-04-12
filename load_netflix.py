import pandas as pd
import gc

data1 = pd.read_csv('/home/vcoscrato/Documents/Data/combined_data_1.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'})
data2 = pd.read_csv('/home/vcoscrato/Documents/Data/combined_data_2.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'})
data = pd.concat([data1, data2])
del data1, data2
gc.collect()
data3 = pd.read_csv('/home/vcoscrato/Documents/Data/combined_data_3.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'})
data = pd.concat([data, data3])
del data3
gc.collect()
data4 = pd.read_csv('/home/vcoscrato/Documents/Data/combined_data_4.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'})
data = pd.concat([data, data4])
del data4
gc.collect()

print(data.head())