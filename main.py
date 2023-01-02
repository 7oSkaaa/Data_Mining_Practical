import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from Algorithms.KNN import KNN
from Algorithms.Naive_Bayes import Naive_Bayes


def main():

    # open the dataset
    Data_set = pd.read_csv("Data_Sets/bodyPerformance.csv", delimiter=",")

    # number of rows and columns in the data_set
    rows, cols = Data_set.shape

    # features name of the train set
    features_names = Data_set.columns[0:11]

    # the target set
    target = Data_set["class"].tolist()

    # get the unique value of the target -> remove redundancy
    target = list(set(target))

    # the training set
    data_training_set = Data_set[features_names].values

    # the target set
    target_set = Data_set["class"]

    # Data Preprocessing
    label_gender = preprocessing.LabelEncoder()
    label_gender.fit(["F", "M"])
    data_training_set[:, 1] = label_gender.transform(data_training_set[:, 1])

    # splitting the sets
    data_train, data_test, target_train, target_test = train_test_split(
        data_training_set, target_set, test_size=0.4, random_state=5
    )

    # KNN algorithm
    KNN_Accuracy = KNN(data_train, data_test, target_train, target_test, rows)

    # Naive algorithm
    Naive_Accuracy = Naive_Bayes(data_train, data_test, target_train, target_test)

    # print the accuracy of the KNN algorithm
    print(f"\nKNN Accuracy: {KNN_Accuracy}")

    # print the accuracy of the naive algorithm
    print(f"\nNaive Accuracy: {Naive_Accuracy}")


if __name__ == "__main__":
    main()
