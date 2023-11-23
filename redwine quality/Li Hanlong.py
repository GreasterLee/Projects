# essential
import numpy as np
import pandas as pd

# data  basic treatment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# resampling
import imblearn as imba
from collections import Counter

# learning model
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

# hyperparameter selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, tpe, hp

# error metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import make_classification

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def x_y_train_test(train_df, test_df):
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_true = test_df.iloc[:, -1]
    return X_train, y_train, X_test, y_true


def standardization(train_df, test_df):
    scaler = StandardScaler()
    train_df.iloc[:, :-1] = scaler.fit_transform(train_df.iloc[:, :-1])
    test_df.iloc[:, :-1] = scaler.transform(test_df.iloc[:, :-1])
    return train_df, test_df


def over_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]

    # Apply SMOTE with custom sampling_strategy
    smote = imba.over_sampling.SMOTE(random_state=42, sampling_strategy={"bad": 1057, "normal": 1057, "good": 1057})
    # {"bad": 1057, "normal": 1057, "good": 1057}
    # {"bad": 981, "normal": 981, "good": 981}
    X_re, y_re = smote.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after over_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


def under_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]

    # Apply RandomUnderSampler with custom sampling_strategy
    undersampler = imba.under_sampling.RandomUnderSampler(random_state=42, sampling_strategy={"bad": 52, "normal": 170,
                                                                                              "good": 170})
    # {"bad": 45, "normal": 157,"good": 157}
    # {"bad": 52, "normal": 170,"good": 170}

    X_re, y_re = undersampler.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after under_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


def hybrid_sampling(df):
    X = df.iloc[:, :-1]  # Features (all columns except the last)
    y = df.iloc[:, -1]

    resampling_pipeline = imba.pipeline.Pipeline([
        ('oversampler', imba.over_sampling.SMOTE(random_state=42,
                                                 sampling_strategy={"bad": int(1057 * 0.6), "good": int(1057 * 0.6)})),
        # Adjust the oversampling strategy
        ('undersampler',
         imba.under_sampling.RandomUnderSampler(random_state=42, sampling_strategy={"normal": int(1057 * 0.6)}))
        # Adjust the undersampling strategy
    ])
    # {"bad": int(981 * 0.6), "good": int(981 * 0.6)} and  {"normal": int(981 * 0.6)}
    # {"bad": int(1057 * 0.6), "good": int(1057 * 0.6)} and {"normal": int(1057 * 0.6)}
    X_re, y_re = resampling_pipeline.fit_resample(X, y)
    class_distribution_after = Counter(y_re)
    print("Class distribution after hybrid_resampling:", class_distribution_after)
    df = pd.concat([X_re, y_re], axis=1)
    return df


def tri_quality(df):
    def classify_quality(quality):
        if quality <= 4:
            return "bad"
        elif 5 <= quality <= 6:
            return "normal"
        else:
            return "good"

    df['tri_quality'] = df['quality'].apply(classify_quality)
    df = df.drop(columns=['quality'])
    return df


def scorer(y_true, y_pred):
    # Calculate recall for the "bad" class
    recall_bad = 0
    # recall_score(y_true, y_pred, labels=["bad"], average="macro")

    # Calculate precision for the "good" class
    precision_good = precision_score(y_true, y_pred, labels=["good"], average="macro")
    # recall_good = recall_score(y_true, y_pred, labels=["good"], average="macro")
    # Assign weights to recall and precision (you can adjust these weights)
    weight_recall = 1  # Weight for "bad" class recall
    weight_precision = 1  # Weight for "good" class precision
    # Calculate the custom score by combining recall and precision with weights
    custom_score_number = weight_recall * recall_bad + weight_precision * precision_good
    return custom_score_number


def model_simple_logi(train_df, test_df):  # model 1: simple multi-logistic regression
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    report = classification_report(y_true, y_pred)
    return confusion, report


def model_lasso_logi(train_df, test_df):  # model 2: Lasso multi-logistic regression
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    from sklearn.preprocessing import LabelEncoder
    original_class_labels = ['bad', 'normal', 'good']
    # Define the search space (e.g., learning rate and max_depth for a gradient boosting model)
    para_space = {
        'C': hp.loguniform('C', -5, 5)
    }

    # Define the objective function (e.g., a classification task with X_train, y_train)
    def objective(params):
        model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1',
                                   max_iter=100000)
        custom_scorer = make_scorer(scorer, greater_is_better=True)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy').mean()
        return -score  # Negative because Hyperopt minimizes the objective function

    # Perform hyperparameter optimization
    best_para = fmin(fn=objective, space=para_space, algo=tpe.suggest, max_evals=5)
    best_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1',
                                    max_iter=100000, C=best_para['C'])
    best_model.fit(X_train, y_train)
    beta_coefficients = best_model.coef_
    y_pred = best_model.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(best_para)
    print(beta_coefficients)
    print(scorer(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    return confusion, report


def model_ridge_logi(train_df, test_df):  # model 2: Lasso multi-logistic regression
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    from sklearn.preprocessing import LabelEncoder
    original_class_labels = ['bad', 'normal', 'good']
    # Define the search space (e.g., learning rate and max_depth for a gradient boosting model)
    para_space = {
        'C': hp.loguniform('C', -5, 5)
    }

    # Define the objective function (e.g., a classification task with X_train, y_train)
    def objective(params):
        model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2',
                                   max_iter=100000)
        custom_scorer = make_scorer(scorer, greater_is_better=True)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy').mean()
        return -score  # Negative because Hyperopt minimizes the objective function

    # Perform hyperparameter optimization
    best_para = fmin(fn=objective, space=para_space, algo=tpe.suggest, max_evals=50)
    best_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1',
                                    max_iter=100000, C=best_para['C'])
    best_model.fit(X_train, y_train)
    beta_coefficients = best_model.coef_
    y_pred = best_model.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(best_para)
    print(beta_coefficients)
    print(scorer(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    return confusion, report


def model_simple_tree(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    custom_scorer = make_scorer(scorer, greater_is_better=True)
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [i for i in range(1, 30)]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=custom_scorer)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(scorer(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    return confusion, report


def model_random_forest(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    custom_scorer = make_scorer(scorer, greater_is_better=True)
    # model-part#
    model = RandomForestClassifier()
    param_grid = {'max_depth': [i for i in range(1, 30, 1)],
                  'n_estimators': [i for i in range(100, 1000, 300)]
                  }
    randm_search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=10, scoring=custom_scorer)
    randm_search.fit(X_train, y_train)
    # prediction and evaluation
    y_pred = randm_search.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(randm_search.best_params_)
    print(randm_search.best_score_)
    print(scorer(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    return confusion, report



def model_xg_boost(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    original_class_labels = ['bad', 'normal', 'good']
    label_encoder = LabelEncoder()
    label_encoder.fit(original_class_labels)
    ynumeric_train = label_encoder.transform(y_train)
    # Define the search space (e.g., learning rate and max_depth for a gradient boosting model)
    para_space = {
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'n_estimators': hp.quniform('n_estimators', 300, 500, 50)
    }

    # Define the objective function (e.g., a classification task with X_train, y_train)
    def objective(params):
        model = XGBClassifier(
            objective='multi:softmax',  # For multi-class classification
            num_class=3,  # Number of classes
            learning_rate=0.001,  # Step size shrinkage to prevent overfitting
            subsample=0.8,  # Fraction of samples used for training
            colsample_bytree=0.8,  # Fraction of features used for training
            random_state=42,

        )
        custom_scorer = make_scorer(scorer, greater_is_better=True)
        score = cross_val_score(model, X_train, ynumeric_train, cv=5, scoring='balanced_accuracy').mean()
        return -score  # Negative because Hyperopt minimizes the objective function

    # Perform hyperparameter optimization
    best_para = fmin(fn=objective, space=para_space, algo=tpe.suggest, max_evals=50)
    best_model = XGBClassifier(
        objective='multi:softmax',  # For multi-class classification
        num_class=3,  # Number of classes
        learning_rate=0.001,  # Step size shrinkage to prevent overfitting
        max_depth=int(best_para['max_depth']),
        subsample=0.8,  # Fraction of samples used for training
        colsample_bytree=0.8,  # Fraction of features used for training
        random_state=42,
        n_estimators=int(best_para['n_estimators'])
    )

    best_model.fit(X_train, ynumeric_train)

    y_pred = best_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(best_para)
    report = classification_report(y_true, y_pred)
    return confusion, report


def modelknn(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    # model-part#
    knn_classifier = KNeighborsClassifier()  # Change the value of n_neighbors as needed
    param_grid = {'n_neighbors': [i for i in range(1, 30)]}
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)

    # prediction and evaluation
    y_pred = grid_search.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)
    print(grid_search.best_params_)
    return confusion, report


def modelknn_wei(train_df, test_df):
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)
    # model-part#
    knn_classifier = KNeighborsClassifier(weights='distance')  # Change the value of n_neighbors as needed
    param_grid = {'n_neighbors': [i for i in range(1, 30)]}
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, y_train)

    # prediction and evaluation
    y_pred = grid_search.predict(X_test)
    class_labels = ['bad', 'normal', 'good']
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
    report = classification_report(y_true, y_pred)
    print(grid_search.best_params_)
    return confusion, report


#final
def recall_for_class_1(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[1], average=None)


def precise_for_class_1(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=[1], average=None)


def bad_or_not(df):
    def classify_quality(tri_quality):
        if tri_quality == 'bad':
            return 1
        else:
            return 0

    df['bad_or_not'] = df['tri_quality'].apply(classify_quality)
    df = df.drop(columns=['tri_quality'])
    return df


def good_or_not(df):
    def classify_quality(tri_quality):
        if tri_quality == 'good':
            return 1
        else:
            return 0

    df['good_or_not'] = df['tri_quality'].apply(classify_quality)
    df = df.drop(columns=['tri_quality'])
    return df


def first_model_lasso_logi(train_df, test_df):  # model 2: Lasso multi-logistic regression
    train_df = bad_or_not(train_df)
    test_df = bad_or_not(test_df)
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    # Define the search space (e.g., learning rate and max_depth for a gradient boosting model)
    para_space = {
        'C': hp.loguniform('C', -5, 5)
    }

    # Define the objective function (e.g., a classification task with X_train, y_train)
    scorer = make_scorer(recall_for_class_1)

    def objective(params):
        model = LogisticRegression(multi_class='ovr', solver='saga', penalty='l1',
                                   max_iter=100000)

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='recall_micro').mean()
        return -score  # Negative because Hyperopt minimizes the objective function

    # Perform hyperparameter optimization
    best_para = fmin(fn=objective, space=para_space, algo=tpe.suggest, max_evals=20)
    best_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1',
                                    max_iter=100000, C=best_para['C'])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))
    print(best_para)
    print(recall_for_class_1(y_true, y_pred))
    return confusion, y_pred, y_prob


def second_model_xgboost(train_df, test_df):
    train_df = good_or_not(train_df)
    test_df = good_or_not(test_df)
    X_train, y_train, X_test, y_true = x_y_train_test(train_df, test_df)

    rf = RandomForestClassifier(max_depth=18, random_state=44, bootstrap=False)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    class_labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))

    return confusion, y_pred, y_prob


def ultimateClassifier(y_pred1, y_prob1, y_true, y_pred2, y_prob2):
    y_prob1 = np.maximum(y_prob1[:, 0], y_prob1[:, 1])
    y_prob2 = np.maximum(y_prob2[:, 0], y_prob2[:, 1])

    data = {'pred1': y_pred1, 'prob1': y_prob1, 'pred2': y_pred2, 'prob2': y_prob2}
    df = pd.DataFrame(data)

    def label_predictions(row):
        if row['pred1'] == 1 and row['pred2'] == 0:
            return 'bad'
        elif row['pred1'] == 0 and row['pred2'] == 1:
            return 'good'
        elif row['pred1'] == 0 and row['pred2'] == 0:
            return 'normal'
        elif row['pred1'] == 1 and row['pred2'] == 1:
            if row['prob1'] > row['prob2']:
                return 'bad'
            else:
                return 'good'

    # Apply the function to create a new 'label' column
    class_labels = ['bad', 'normal', 'good']
    y_pred = df.apply(label_predictions, axis=1)
    confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion = pd.DataFrame(confusion, index=pd.MultiIndex.from_product([['true'], class_labels]),
                             columns=pd.MultiIndex.from_product([['pred'], class_labels]))


    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size for better readability
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['bad','normal', 'good'],
                yticklabels=['bad','normal', 'good'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    return confusion


if __name__ == '__main__':
    # import the dataset
    wine_whole = pd.read_csv("winequality-red.csv")
    wine_train, wine_test = split_data(wine_whole)
    quality = wine_train.iloc[:, -1]

    # histogram
    # sns.histplot(quality, bins=20, kde=False, color='blue')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Quality distribution')
    # plt.show()

    # 3-class of quality
    wine_train = tri_quality(wine_train)
    wine_test = tri_quality(wine_test)

    # standardization
    wine_train, wine_test = standardization(wine_train, wine_test)

    # resampling
    y_count = wine_test.iloc[:, -1]
    print("testdata:Class distribution before any resampling:", Counter(y_count))
    y_count = wine_train.iloc[:, -1]
    print("Class distribution before any resampling:", Counter(y_count))
    wine_train_over = over_sampling(wine_train)
    wine_train_uder = under_sampling(wine_train)
    wine_train_hybr = hybrid_sampling(wine_train)

    # # # model 1: simple multi-logistic regression
    print('simple multi-logistic regression')
    kk = model_simple_logi(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_simple_logi(wine_train_uder, wine_test)[1])
    # print(model_simple_logi(wine_train_hybr, wine_test)[1])

    # # model 2: lasso multi-logistic regression
    print('lasso multi-logistic regression')
    kk = model_lasso_logi(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_lasso_logi(wine_train, wine_test)[1])
    # print(model_lasso_logi(wine_train_uder, wine_test)[1])
    # print(model_lasso_logi(wine_train_hybr, wine_test)[1])

    # # # model 3:  ridge multi-logistic regression
    print('ridge multi-logistic regression')
    kk = model_ridge_logi(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_ridge_logi(wine_train, wine_test)[1])
    # print(model_ridge_logi(wine_train_uder, wine_test)[1])
    # print(model_ridge_logi(wine_train_hybr, wine_test)[1])

    # #model 4: decision tree
    print('decision tree')
    kk = model_simple_tree(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_simple_tree(wine_train, wine_test)[1])
    # print(model_simple_tree(wine_train_uder, wine_test)[1])
    # print(model_simple_tree(wine_train_hybr, wine_test))

    # # model 5: random forest
    print('random forest')
    kk = model_random_forest(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_random_forest(wine_train, wine_test)[1])
    # print(model_random_forest(wine_train_uder, wine_test)[1])
    # print(model_random_forest(wine_train_hybr, wine_test)[1])

    # # # model 6: xgboost
    print('xgboost')
    kk = model_xg_boost(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(model_xg_boost(wine_train, wine_test)[1])
    # print(model_xg_boost(wine_train_uder, wine_test)[1])
    # print(model_xg_boost(wine_train_hybr, wine_test)[1])

    # # # model7: KNN
    print('knn')
    kk = modelknn(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(modelknn(wine_train, wine_test)[1])
    # print(modelknn(wine_train_uder, wine_test)[1])
    # print(modelknn(wine_train_hybr, wine_test)[1])

    # # # model_haha: KNN_cv
    print('knn weighted')
    kk = modelknn_wei(wine_train_over, wine_test)
    print(kk[0])
    print(kk[1])
    # print(modelknn_wei(wine_train, wine_test)[1])
    # print(modelknn_wei(wine_train_uder, wine_test)[1])
    # print(modelknn_wei(wine_train_hybr, wine_test)[1])

    print('final best')
    y_true = wine_test.iloc[:, -1]
    confusion1, y_pred1, y_prob1 = first_model_lasso_logi(wine_train_hybr, wine_test)
    print(confusion1)
    confusion2, y_pred2, y_prob2 = second_model_xgboost(wine_train_hybr, wine_test)
    print(confusion2)
    print(ultimateClassifier(y_pred1, y_prob1, y_true, y_pred2, y_prob2))
