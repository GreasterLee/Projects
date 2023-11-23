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


def AUPRC(y_true, y_scores,name):
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute AUPRC
    auprc = auc(recall, precision)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}{name}',)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()
    print(f'AUPRC: {auprc:.2f}')


# CHUNK THREE: models ###########################################################


# KNN
def knn(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    knn_classifier = KNeighborsClassifier()

    # grid search and refit#
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'n_neighbors': [i for i in range(1, 30)]}
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=stratified_kfold, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = grid_search.predict(X_test)
    y_scores = grid_search.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    print(grid_search.best_params_) # valid when doing CV
    AUPRC(y_true, y_scores,"knn")
    return confusion, report


# Logistic
def logi(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    logi_classifier = LogisticRegression(solver='sag',max_iter=100000)

    # random search and refit#
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    random_search = RandomizedSearchCV(logi_classifier, param_grid, cv=stratified_kfold, n_iter=3,
                                       scoring='accuracy')
    random_search.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = random_search.predict(X_test)
    y_scores = random_search.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    print(random_search.best_params_) # valid when doing CV
    AUPRC(y_true, y_scores,'logi')
    return confusion, report


# SVM
def svm(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # model-part#
    svm_classifier = SVC(probability=True, kernel='linear')

    # random search and refit#
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    random_search = RandomizedSearchCV(svm_classifier, param_grid, cv=stratified_kfold, n_iter=3,
                                       scoring='accuracy')
    random_search.fit(X_train, y_train)

    # predict and evaluation#
    y_pred = random_search.predict(X_test)
    y_scores = random_search.predict_proba(X_test)[:, 1]
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)

    # print plot and return#
    print(random_search.best_params_) # valid when doing CV
    AUPRC(y_true, y_scores, 'svm')
    return confusion, report


# random forest
def random_forest(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # 1. Define Objective in optimization of hyperparameter
    def objective(params):
        max_depth = int(params['max_depth'])
        n_estimators = int(params['n_estimators'])

        rf_classifier = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_jobs=-1
        )
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=stratified_kfold,
                                    scoring='accuracy')
        # Calculate the mean cross-validation accuracy
        score = np.mean(cv_scores)

        return {'loss': 1 - score, 'status': STATUS_OK}

    # 2. Define the hyperparameter search space
    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
    }

    trials = Trials()

    # 3. Optimize hyperparameters and get the best hyperparameters
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_max_depth = int(best['max_depth'])
    best_n_estimators = int(best['n_estimators'])

    # 4. Train the final model with the best hyperparameters
    best_model = RandomForestClassifier(
        max_depth=best_max_depth,
        n_estimators=best_n_estimators,
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
    print('best_max_depth is', best_max_depth)
    print('n_estimators is', best_n_estimators)
    AUPRC(y_true, y_scores,'forest')
    return confusion, report


#XgBoost
def xgboost(train_df, test_df):  # model 1: simple multi-logistic regression
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # 1. Define Objective in optimization of hyperparameter
    def objective(params):
        max_depth = int(params['max_depth'])
        n_estimators = int(params['n_estimators'])
        learning_rate = params['learning_rate']

        model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_jobs=-1
        )
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold,
                                    scoring='accuracy')
        # Calculate the mean cross-validation accuracy
        score = np.mean(cv_scores)

        return {'loss': 1 - score, 'status': STATUS_OK}

    # 2. Define the hyperparameter search space
    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3))
    }

    trials = Trials()

    # 3. Optimize hyperparameters and get the best hyperparameters
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_max_depth = int(best['max_depth'])
    best_n_estimators = int(best['n_estimators'])
    best_learning_rate = best['learning_rate']

    # 4. Train the final model with the best hyperparameters
    best_model = xgb.XGBClassifier(
        max_depth=best_max_depth,
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
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
    print('best_max_depth is', best_max_depth)
    print('n_estimators is', best_n_estimators)
    print('best_learning_rate is ', best_learning_rate)
    AUPRC(y_true, y_scores,'boost')
    return confusion, report
