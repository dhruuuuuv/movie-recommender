import csv
import math
import operator
import collections

import numpy as np

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter='\t')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(int, data[i]))
            data[i] = point
        return data

# produces a rating matrix from a list of user data
def form_ratings_matrix(users, items, user_data):
    max_user = max(users)
    max_item = max(items)

    # plus one so not 0 indexed
    ratings_matrix = np.zeros((max_user + 1, max_item + 1))

    for row in user_data:
        user    = row[0]
        item    = row[1]
        rating  = row[2]
        ratings_matrix[user, item] = rating

    return ratings_matrix

# produces a testing array to use to test the other functions
def test_data():
    td = [
        [1, 1, 5], [1, 2, 4], [1, 3, 3],
        [2, 1, 5], [2, 2, 4], [2, 3, 0],
        [3, 1, 5], [3, 2, 0], [3, 3, 3],
    ]
    td = np.array(td)
    return td


def main():
    # print floats instead of using scientific notation
    np.set_printoptions(suppress=True)

    # import the data and remove timestamp
    train_data = import_dataset("train.txt")
    test_data = import_dataset("train.txt")
    train_data = np.array(train_data)
    test_data = np.array(train_data)
    data = data[:, :-1]

    # split data into sorted by users and items
    users = list(set(data[:,0]))
    items = list(set(data[:,1]))

    user_data = data[data[:,0].argsort()]
    item_data = data[data[:,1].argsort()]

    # produce the ratings matrix
    rm = form_ratings_matrix(users, items, user_data)

    # optional for testing using the small testing matrix
    td = test_data()
    tdm = form_ratings_matrix(list(set(td[:,0])), list(set(td[:,0])), td[td[:,0].argsort()])

    # test using k = 10 or k = 3 by commenting / uncommenting
    training, test = data_split(data, 10)
    # training, test = data_split(data, 3)

    print("training dataset")
    print(training)
    print("\n")

    print("test dataset")
    print(test)
    print("\n")

    error = rmse(test)

if __name__ == '__main__':
    main()
