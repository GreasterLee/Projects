# ESSENTIAL:
import numpy as np
import pandas as pd

# DATA PREPROCESSING:
from sklearn.model_selection import train_test_split  # train_test_split
from sklearn.preprocessing import RobustScaler  # standardization
import imblearn as imba  # resample method
from collections import Counter  # resample method

# VISUALIZATION:
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, auc

# CROSS VALIDATION:
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# MACHINE MODEL:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ERROR METRICS:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# CHUNK ONE: data preprocessing ###########################################################
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def x_y_train_test(train_df, test_df):
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_true = test_df.iloc[:, -1]
    return X_train, y_train, X_test, y_true


# data preprocessing: scaler
def standardization(train_df, test_df):
    scaler = RobustScaler()
    # amount
    train_df.iloc[:, -2:-1] = scaler.fit_transform(train_df.iloc[:, -2:-1])
    test_df.iloc[:, -2:-1] = scaler.transform(test_df.iloc[:, -2:-1])
    # time
    train_df.iloc[:, 0:1] = scaler.fit_transform(train_df.iloc[:, 0:1])
    test_df.iloc[:, 0:1] = scaler.transform(test_df.iloc[:, 0:1])
    return train_df, test_df


# data preprocessing: resample
# SMOTE over
def smote_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]

    # Apply SMOTE with custom sampling_strategy
    smote = imba.over_sampling.SMOTE(random_state=42)
    X_re, y_re = smote.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after smote over_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


# ADASYN over
def ADASYN_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]
    adasyn = imba.over_sampling.ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_re, y_re = adasyn.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after ADASYN over_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


# Random under
def under_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]

    # Apply RandomUnderSampler with custom sampling_strategy
    undersampler = imba.under_sampling.RandomUnderSampler(random_state=42)

    X_re, y_re = undersampler.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after under_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


# CHUNK TWO: visualization ###########################################################
def estimate_density_and_visualize(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]
    # Select only samples with the label of interest
    X_label_of_interest = X[y == 1]

    # Train a K-nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_label_of_interest)

    # Calculate distances and indices of k_neighbors nearest neighbors
    distances, indices = nn_model.kneighbors(X_label_of_interest)

    # Calculate density as the inverse of the average distance to k_neighbors nearest neighbors
    densities = 1 / (distances.sum(axis=1) / 5)

    # Visualize the density distribution with a histogram
    plt.hist(densities, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Density')
    plt.ylabel('Frequency')
    plt.title(f'Density Distribution of Label {1} Samples')
    plt.show()

    return densities


def totalplot(y_test, scores,names):
    fig, ax = plt.subplots()

    # 对每个模型绘制Precision-Recall曲线并保存图像
    for i in range(5):
        # 计算Precision-Recall曲线和AUC
        precision, recall, thresholds = precision_recall_curve(y_test, scores[i])
        auprc = auc(recall, precision)

        # 绘制曲线
        ax.plot(recall, precision, label=f'{names[i]} (AUPRC={auprc:.2f})')

    # 添加标题和标签
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    plt.show()


# CHUNK THREE: models ###########################################################


# KNN
def knn(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = knn_classifier.predict(X_test)
    y_scores = knn_classifier.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    return confusion, report, y_scores


# Logistic
def logi(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    logi_classifier = LogisticRegression(solver='sag',max_iter=100000,C=10)
    logi_classifier.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = logi_classifier.predict(X_test)
    y_scores = logi_classifier.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    return confusion, report, y_scores


# SVM
def svm(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    svm_classifier = SVC(probability=True, kernel='linear', C=1)
    svm_classifier.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = svm_classifier.predict(X_test)
    y_scores = svm_classifier.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    return confusion, report, y_scores


# random forest
def random_forest(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    best_model = RandomForestClassifier(
        max_depth=9,
        n_estimators=150,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    return confusion, report, y_scores


#XgBoost
def xgboost(train_df, test_df):  # model 1: simple multi-logistic regression
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    best_model = xgb.XGBClassifier(
        max_depth=7,
        n_estimators=194,
        learning_rate=0.1428,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    return confusion, report, y_scores
