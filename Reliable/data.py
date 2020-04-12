import numpy as np
from surprise import Dataset
from sklearn.model_selection import train_test_split

def build_data(name='ml-100k', test_size=0.1, random_state=0):
    data = Dataset.load_builtin(name)
    users = [a[0] for a in data.raw_ratings]
    train, test = train_test_split(data.raw_ratings, shuffle=True, random_state=random_state, test_size=test_size, stratify=users)
    data.raw_ratings = train
    test = [test[:3] for test in test]
    # Delete a few test itens that are not in the training set
    train_itens = np.unique([train[1] for train in train])
    for i in np.unique([test[1] for test in test]):
        if i not in train_itens:
            for id in reversed(np.argwhere(np.array([test[1] for test in test]) == i)):
                del test[id[0]]
    return data, test