from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def Naive_Bayes(data_train, data_test, target_train, target_test):

    # create a GaussianNB Classifier
    model = GaussianNB()

    # train Model using Training Sets
    model.fit(data_train, target_train)

    # getting the predicted answers
    predicted = model.predict(data_test)

    # check the accuracy of the prediction by comparing it with the answer
    Naive_Accuracy = metrics.accuracy_score(target_test, predicted)

    # return Naive accuracy
    return Naive_Accuracy
