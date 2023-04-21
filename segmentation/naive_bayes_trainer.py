import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd

path_to_train = '/Users/antonioneto/Downloads/train_data_final.csv'

train_data = pd.read_csv(path_to_train, delimiter= ',')
print(np.array(train_data)[:, :-1])

nb = GaussianNB()
X_train = np.array(train_data)[:, :-1]
y_train = np.array(train_data)[:, -1]
# Multiply all values by 100, convert to int, and round to nearest integer
y_train = np.rint(y_train * 100).astype(int)
nb.fit(X_train, y_train)

print(nb.get_params())