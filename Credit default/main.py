# ESSENTIAL:
import sys
import numpy as np
import pandas as pd

# plot:
import matplotlib.pyplot as plt
import seaborn as sns

# models for cv
import models_for_cv as md

# models used in Adasyn
import models_for_overset as ld

if __name__ == '__main__':
    credit_whole = pd.read_csv("creditcard.csv")

    # check whether there is invalid input:
    if len(credit_whole.columns) != 31:
        print("Invalid input: The CSV file should have exactly 31 columns.")
        sys.exit()
    # Check if the last column consists of only 0 and 1
    last_column_values = set(credit_whole.iloc[:, -1])
    if not all(val in {0, 1} for val in last_column_values):
        print("Invalid input: The last column should consist of only 0 and 1.")
        sys.exit()
    print("Input is valid:")

    #train test split
    credit_train, credit_test = md.split_data(credit_whole)

    # #time box plot
    # time_box = credit_train['Time']
    # sns.boxplot(data=time_box)
    # plt.xlabel('Time')
    # plt.title('Box Plot of time')
    # plt.show()

    # #amount box plot
    # time_box = credit_train['Amount']
    # sns.boxplot(data=time_box)
    # plt.xlabel('Amount')
    # plt.title('Box Plot of amount')
    # plt.show()

    # standardization
    credit_train, credit_test = md.standardization(credit_train,credit_test )

    # # visualize density
    # md.estimate_density_and_visualize(credit_train)

    # count numbers of default before resample
    y_test = credit_test.iloc[:, -1]
    print("testdata:Class distribution:", md.Counter(y_test))
    y_train = credit_train.iloc[:, -1]
    print("Class distribution before any resampling:", md.Counter(y_train))
    # data resample
    credit_train_over = md.smote_sampling(credit_train)
    credit_train_adas = md.ADASYN_sampling(credit_train)
    credit_train_unde = md.under_sampling(credit_train)

    # ######## model training ##############

    # 1.doing cv in undersample to get hyperparameters
    # the result is used in models_for_overset
    # print('KNN')
    # result = md.knn(credit_train_unde, credit_test)
    # print(result[0])
    # print(result[1])
    #
    # print('logistic regression')
    # result = md.logi(credit_train_unde, credit_test)
    # print(result[0])
    # print(result[1])
    #
    # print('SVM')
    # result = md.svm(credit_train_unde, credit_test)
    # print(result[0])
    # print(result[1])
    #
    # print('random forest')
    # result = md.random_forest(credit_train_unde, credit_test)
    # print(result[0])
    # print(result[1])

    # print('xgboost')
    # result = md.xgboost(credit_train_unde, credit_test)
    # print(result[0])
    # print(result[1])

    # 2.using hyperparameters in oversample set
    scores = []
    names = ['knn', 'logi', 'smv', 'forest', 'boost']
    print('KNN')
    result = ld.knn(credit_train_adas, credit_test)
    print(result[0])
    print(result[1])
    scores.append(result[2])

    print('logistic regression')
    result = ld.logi(credit_train_adas, credit_test)
    print(result[0])
    print(result[1])
    scores.append(result[2])

    print('SVM')
    # result = ld.svm(credit_train_adas, credit_test)
    print(result[0])
    print(result[1])
    scores.append(result[2])

    print('random forest')
    result = ld.random_forest(credit_train_adas, credit_test)
    print(result[0])
    print(result[1])
    scores.append(result[2])

    print('xgboost')
    result = ld.xgboost(credit_train_adas, credit_test)
    print(result[0])
    print(result[1])
    scores.append(result[2])

    ld.totalplot(y_test, scores, names)




