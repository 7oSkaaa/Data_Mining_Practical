from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math


def KNN(data_train, data_test, target_train, target_test, row):

    # getting k
    k = math.isqrt(row)

    # define number of neighbours to deal with -> sqrt(rows of the data_set) preferred odd number
    neighbours = KNeighborsClassifier(n_neighbors=k + (k % 2 == 0))

    # train Model using Training Sets
    neighbours.fit(data_train, target_train)

    # getting the predicted answers
    predicted = neighbours.predict(data_test)

    # check the accuracy of the prediction by comparing it with the answer
    KNN_Accuracy = metrics.accuracy_score(target_test, predicted)

    # return the KNN accuracy
    return KNN_Accuracy
