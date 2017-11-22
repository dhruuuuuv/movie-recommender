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

# split the data into two distinct training and testing rating matrices
def data_split(data, k):
    users = list(set(data[:,0]))
    items = list(set(data[:,1]))

    user_data = data[data[:,0].argsort()]
    item_data = data[data[:,1].argsort()]

    training_data = []
    test_data = []

    last_user = user_data[0][0]
    count = k

    # push the first k instances from the user data into the testing set
    for i, row in enumerate(user_data):
        user = row[0]
        if (user == last_user):
            if count > 0:
                test_data.append(row)
                count -= 1
            else:
                training_data.append(row)
        else:
            last_user = user
            count = k
            test_data.append(row)
            count -= 1

    test_data = np.array(test_data)
    training_data = np.array(training_data)

    training_matrix = form_ratings_matrix(users, items, training_data)
    test_matrix = form_ratings_matrix(users, items, test_data)

    return (training_matrix, test_matrix)

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

# calculate a user's average rating from the ratings matrix, assuming that unrated items have value of 0
def avg_user_rating(ratings_matrix):

    # calculates the average of each row (user)
    urv = np.ma.average(ratings_matrix, axis=1, weights=ratings_matrix.astype(bool))

    # sets all the unassigned values equal to 0
    urv[urv.mask] = 0
    return urv

# given two items, produce the normalised vectors of those items
def norm_item_vecs(ratings_matrix, i, j):
    avg_user_vector = avg_user_rating(ratings_matrix)

    # produce copy so that the original ratings matrix is unaffected
    rm_copy = np.copy(ratings_matrix)

    ri = rm_copy[:,i]
    rj = rm_copy[:,j]

    # from below assume that items that haven't been rated have value 0
    for i, row in enumerate(ri):
        if row != 0:
            ri[i] = ri[i] - avg_user_vector[i]

    for i, row in enumerate(rj):
        if row != 0:
            rj[i] = rj[i] - avg_user_vector[i]

    return (np.array(ri), np.array(rj))

# given a user and an item, return the predicted rating for that item
def prediction(ratings_matrix, user, item):
    # get the user vector from the matrix
    user_vec = ratings_matrix[user,:]

    numerator   = []
    denominator = []

    for n in range(1, len(user_vec)):
        if user_vec[n] != 0:
            # use below for similarities in range [-1, 1]
            # x = similarity(ratings_matrix, item, n)
            # else use below for mapped values to [1,5]
            x = rescale_similarity(similarity(ratings_matrix, item, n))

            # print("similarity: {}, item: {}, n: {}".format(x, item, n))

            numer = user_vec[n] * x

            # comment / uncomment relevant lines to try with / without absolute denominator
            denom = abs(x)
            # denom = x

            numerator.append(numer)
            denominator.append(denom)

    numerator = sum(numerator)
    denominator = sum(denominator)

    # to ensure that the denominator isn't 0
    if denominator == 0:
        prediction = 0
    else:
        prediction = numerator / denominator

    # print("user: {}, item: {}, prediction: {}".format(user, item, prediction))

    return prediction

# given two items, calculate the adjusted cosine similarity between them
def similarity(ratings_matrix, i, j):
    ri, rj = norm_item_vecs(ratings_matrix, i, j)

    x = np.dot(ri, rj)

    # calculate the euclidian norm of both vectors
    ni = np.linalg.norm(ri)
    nj = np.linalg.norm(rj)

    # check to ensure the denominator isn't 0
    if nj == 0 or ni == 0:
        sim = 0
    else:
        sim = x / (ni*nj)

    return sim

# maps the values from [-1,1] to [1, 5]
def rescale_similarity(x):
    return ((2 * x) + 3)

# iterate over all the ratings and compare the predicted results against the actual ratings
def rmse(ratings_matrix):
    number_ratings = np.count_nonzero(ratings_matrix)
    # print("number of ratings: {}".format(number_ratings))

    n, m = ratings_matrix.shape
    pred_matrix = np.zeros((n, m))
    a = []

    # manually iterate through each option
    for i, row in enumerate(ratings_matrix):
        for j, item in enumerate(ratings_matrix):
            if ratings_matrix[i, j] != 0:
                # print("actual rating: {}".format(ratings_matrix[i, j]))

                # use below for just prediction matrix
                # x = prediction(ratings_matrix, i, j)
                # pred_matrix[i, j] = x

                # else use the below for the rmse matrix
                x = prediction(ratings_matrix, i, j)
                y = (x - ratings_matrix[i, j]) ** 2
                a.append(y)

    root_squared_error = math.sqrt((sum(a) / number_ratings))
    print("RMSE: {}".format(root_squared_error))

    return root_squared_error


def main():
    # print floats instead of using scientific notation
    np.set_printoptions(suppress=True)

    # import the data and remove timestamp
    data = import_dataset("u.data")
    data = np.array(data)
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
