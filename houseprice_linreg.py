# imports
import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders.binary import BinaryEncoder


from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from scipy.stats import loguniform
from sklearn.decomposition import TruncatedSVD
from scipy import stats

from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
import matplotlib.pyplot as plt

# data visualization packages
import seaborn as sns
# ignore warnings
import warnings

def ignore_warn(*args, **kwargs):
    """
    ignore warnings
    :param args: NA
    :param kwargs: NA
    :return: NA
    """
    pass

# ignore warnings
if __name__ == '__main__':
    warnings.warn = ignore_warn

# load and show data
if __name__ == '__main__':
    # read the data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # drop trains ID
    train = train.drop('Id', axis=1)
    # store test ID
    IDS = test.Id
    # drop test ID
    test = test.drop('Id', axis=1)

# heatmap for numerical variables
if __name__ == '__main__':
    corrs = train.corr()
    top_cols = corrs.nlargest(16, 'SalePrice')['SalePrice'].index
    fin_corr = train[top_cols].corr()
    sns.heatmap(fin_corr, annot=True)
    # plt.show()

# uni-variate boxplot
if __name__ == '__main__':
    f, axes = plt.subplots(3, 5, figsize=(16, 12))
    counter = 0
    tops = top_cols[1:]
    for i in range(len(tops)):
        num = counter % 5
        row = counter // 5
        sns.boxplot(x=tops[i], hue=i, data=train, ax=axes[row, num - 1 if num != 0 else 4])
        counter += 1
    # plt.show()

# uni-variate histogram
if __name__ == '__main__':
    f, axes = plt.subplots(3, 5, figsize=(16, 12))
    counter = 0
    tops = top_cols[1:]
    for i in range(len(tops)):
        num = counter % 5
        row = counter // 5
        sns.histplot(x=tops[i], hue=i, data=train, ax=axes[row, num - 1 if num != 0 else 4])
        counter += 1
    # plt.show()

# bi-variate scatterplot
if __name__ == '__main__':
    f, axes = plt.subplots(3, 5, figsize=(16, 12))
    counter = 0
    tops = top_cols[1:]
    for i in range(len(tops)):
        num = counter % 5
        row = counter // 5
        sns.scatterplot(train[tops[i]], train['SalePrice'], ax=axes[row, num - 1 if num != 0 else 4])
        counter += 1
    # plt.show()

# noises processing
if __name__ == '__main__':
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    train = train.drop(train[(train['GarageArea'] == 0) | (train['SalePrice'] > 700000)].index)
    train = train.drop(train[(train['GarageArea'] > 1200) & (train['SalePrice'] < 300000)].index)
    train = train.drop(train[(train['TotalBsmtSF'] == 0) | (train['SalePrice'] > 700000)].index)
    train = train.drop(train[(train['1stFlrSF'] >= 2500)].index)
    train = train.drop(train[(train['BsmtFinSF1'] > 4000)].index)
    train = train.drop(train[(train['LotFrontage'] > 300)].index)

# construct new feature
if __name__ == '__main__':
    train['YearRemodPass'] = 2020 - train['YearRemodAdd']
    test['YearRemodPass'] = 2020 - test['YearRemodAdd']
    train['RepYear'] = 2020 - (train['YearBuilt'] + train['GarageYrBlt']) / 2
    test['RepYear'] = 2020 - (test['YearBuilt'] + test['GarageYrBlt']) / 2

# target numerical features
if __name__ == '__main__':
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']  # log encoder
    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder

# fill nas for numerical data
if __name__ == '__main__':
    for i in log_encodes:
        mx = max(list(train[i]), key=list(train[i]).count)
        train[i] = train[i].fillna(mx)
        mx1 = max(list(test[i]), key=list(test[i]).count)
        test[i] = test[i].fillna(mx)

    for j in std_encodes:
        train[j] = train[j].fillna(train[j].mean())
        test[j] = test[j].fillna(test[j].mean())

# categorical data processing
if __name__ == '__main__':
    all_cols = list(train.columns)
    num_cols = train.corr().columns
    for i in num_cols:
        all_cols.remove(i)
    cates = train[all_cols]
    lst = []
    for i in all_cols:
        if any(train[i].isna()):
            lst.append(i)
    for j in lst:
        all_cols.remove(j)

    train['Alley'] = train['Alley'].fillna('No')
    train[cates.columns[17]] = train[cates.columns[17]].fillna('None')
    train[cates.columns[23]] = train[cates.columns[23]].fillna('None')
    train[cates.columns[25]] = train[cates.columns[25]].fillna('None')
    train[cates.columns[29]] = train[cates.columns[29]].fillna('None')
    train[cates.columns[32]] = train[cates.columns[32]].fillna('None')

    test['Alley'] = test['Alley'].fillna('No')
    test[cates.columns[17]] = test[cates.columns[17]].fillna('None')
    test[cates.columns[23]] = test[cates.columns[23]].fillna('None')
    test[cates.columns[25]] = test[cates.columns[25]].fillna('None')
    test[cates.columns[29]] = test[cates.columns[29]].fillna('None')
    test[cates.columns[32]] = test[cates.columns[32]].fillna('None')

    # specs = []
    # for i in test.columns:
    #     if test[i].isnull().sum() > 0:
    #         specs.append(i)
    # for j in specs:
    #     test[j] = test[j].fillna('None')

# categorical visualization
if __name__ == '__main__':
    f, axes = plt.subplots(17, 2, figsize=(15, 35))
    counter = 0
    tops = all_cols
    for i in range(len(tops)):
        num = counter % 2
        row = counter // 2
        sns.barplot(x=train[tops[i]].value_counts().index, y=train[tops[i]].value_counts(),
                    ax=axes[row, num - 1 if num != 0 else 1])
        counter += 1
    # plt.show()

# target categorical features
if __name__ == '__main__':
    bin_encoders = ['Street', 'CentralAir', 'Utilities']
    glmm_encode = list(set(all_cols) - set(bin_encoders))  # one-hot

# log the target variable
if __name__ == '__main__':
    sns.histplot(train['SalePrice'])
    # plt.show()
    train['SalePrice'] = np.log(train['SalePrice'])

# histogram after taking log
if __name__ == '__main__':
    sns.histplot(train['SalePrice'])
    # plt.show()

# LinearReg
def lr_model(df, col, tst):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    # encode
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    label_transformer = Pipeline(steps=[
        ('LabelEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    def take_log(column):
        return np.log(column)

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(take_log)),
        ('StdEncoder', StandardScaler())
    ])

    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    std_transformer = Pipeline(steps=[
        ('StdEncoder', StandardScaler())
    ])

    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder
    rbst_transformer = Pipeline(steps=[
        ('RobustEncoder', RobustScaler())
    ])

    bin_encodes = ['Street', 'CentralAir', 'Utilities']
    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())
    ])

    onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            ('std', std_transformer, std_encodes),
            ('robust', rbst_transformer, robust_encodes),
            ('binary', binary_transformer, bin_encodes),
            ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),  #
        # 'Logistic Regression': LogisticRegression(),
        # 'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),  #
        # 'KNN Regression': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),  #
        # 'Naive Bayes': GaussianNB(),
        # 'Hidden Markov': hmm.GaussianHMM()
    }

    # Choice of Model
    mdl_name = 'Linear Regression'

    # related tunes
    fit_intercept = [True]
    normalize = [False]

    parameters = {
        'fit_intercept': fit_intercept,
        'normalize': normalize
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pl.fit(X_train, y_train)

    scores = np.sqrt(-cross_val_score(pl, X=X_train, y=y_train, verbose=1, cv=5, scoring='neg_mean_squared_error'))

    cv_df = pd.DataFrame(scores.reshape(1, -1))
    cv_df.columns = ['cv' + str(i) for i in range(1, 6)]
    cv_df.index = ['RMSE']
    print(cv_df)
    final_score = np.mean(scores)
    print("cv score: {}".format(np.mean(scores)))

    print('参数选择:{}'.format(pl['regressor'].best_params_))

    prediction = pl.predict(X_test)
    RMSE = np.sqrt(np.mean((y_test - prediction) ** 2))
    print('RMSE on test set: {}'.format(RMSE))
    print('Pipeline score: {}'.format(pl.score(X_test, y_test)))

    #     print('intercetpt: {}'.format(pl['regressor'].intercept_))
    #     print('Coefficients')
    #     print(len(pl['regressor'].coef_))
    #     prediction = search.predict(tst)

    #     pd.DataFrame({'Id': IDS, 'SalePrice': prediction}).to_csv('TMP.csv', index=False)
    return RMSE


# LASSO
def ls_model(df, col, tst):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    # encode
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    label_transformer = Pipeline(steps=[
        ('LabelEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    def take_log(column):
        return np.log(column)

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(take_log)),
        ('StdEncoder', StandardScaler())
    ])

    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    std_transformer = Pipeline(steps=[
        ('StdEncoder', StandardScaler())
    ])

    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder
    rbst_transformer = Pipeline(steps=[
        ('RobustEncoder', RobustScaler())
    ])

    bin_encodes = ['Street', 'CentralAir', 'Utilities']
    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())
    ])

    onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            ('std', std_transformer, std_encodes),
            ('robust', rbst_transformer, robust_encodes),
            ('binary', binary_transformer, bin_encodes),
            ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),  #
        # 'Logistic Regression': LogisticRegression(),
        'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),  #
        # 'KNN Regression': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),  #
        # 'Naive Bayes': GaussianNB(),
        # 'Hidden Markov': hmm.GaussianHMM()
    }

    # Choice of Model
    mdl_name = 'LASSO'

    # related tunes
    alpha = [0.00035]
    fit_intercept = [True]
    normalize = [False]
    warm_start = [True]
    positive = [False]

    parameters = {
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        'warm_start': warm_start,
        'positive': positive
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pl.fit(X_train, y_train)

    scores = np.sqrt(-cross_val_score(pl, X=X_train, y=y_train, verbose=1, cv=5, scoring='neg_mean_squared_error'))

    cv_df = pd.DataFrame(scores.reshape(1, -1))
    cv_df.columns = ['cv' + str(i) for i in range(1, 6)]
    cv_df.index = ['RMSE']
    print(cv_df)
    final_score = np.mean(scores)
    print("cv score: {}".format(np.mean(scores)))

    print('参数选择:{}'.format(pl['regressor'].best_params_))

    prediction = pl.predict(X_test)
    RMSE = np.sqrt(np.mean((y_test - prediction) ** 2))
    print('RMSE on test set: {}'.format(RMSE))
    print('Pipeline score: {}'.format(pl.score(X_test, y_test)))

    return RMSE

# ridge regression
def rg_model(df, col, tst):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    # encode
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    label_transformer = Pipeline(steps=[
        ('LabelEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    def take_log(column):
        return np.log(column)

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(take_log)),
        ('StdEncoder', StandardScaler())
    ])

    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    std_transformer = Pipeline(steps=[
        ('StdEncoder', StandardScaler())
    ])

    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder
    rbst_transformer = Pipeline(steps=[
        ('RobustEncoder', RobustScaler())
    ])

    bin_encodes = ['Street', 'CentralAir', 'Utilities']
    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())
    ])

    onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            ('std', std_transformer, std_encodes),
            ('robust', rbst_transformer, robust_encodes),
            ('binary', binary_transformer, bin_encodes),
            ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),  #
        # 'Logistic Regression': LogisticRegression(),
        'LASSO': Lasso(),
        'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),  #
        # 'KNN Regression': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),  #
        # 'Naive Bayes': GaussianNB(),
        # 'Hidden Markov': hmm.GaussianHMM()
    }

    # Choice of Model
    mdl_name = 'Ridge'

    # related tunes
    alpha = [10.5416]
    solver = ['auto']
    fit_intercept = [True]
    normalize = [False]
    copy_X = [True]

    parameters = {
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        'solver': solver,
        'copy_X': copy_X
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pl.fit(X_train, y_train)

    scores = np.sqrt(-cross_val_score(pl, X=X_train, y=y_train, verbose=1, cv=5, scoring='neg_mean_squared_error'))

    cv_df = pd.DataFrame(scores.reshape(1, -1))
    cv_df.columns = ['cv' + str(i) for i in range(1, 6)]
    cv_df.index = ['RMSE']
    print(cv_df)
    final_score = np.mean(scores)
    print("cv score: {}".format(np.mean(scores)))

    print('参数选择:{}'.format(pl['regressor'].best_params_))

    prediction = pl.predict(X_test)
    RMSE = np.sqrt(np.mean((y_test - prediction) ** 2))
    print('RMSE on test set: {}'.format(RMSE))
    print('Pipeline score: {}'.format(pl.score(X_test, y_test)))

    return RMSE

# Elastic Net
def en_model(df, col, tst):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    # encode
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    label_transformer = Pipeline(steps=[
        ('LabelEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    def take_log(column):
        return np.log(column)

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(take_log)),
        ('StdEncoder', StandardScaler())
    ])

    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    std_transformer = Pipeline(steps=[
        ('StdEncoder', StandardScaler())
    ])

    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder
    rbst_transformer = Pipeline(steps=[
        ('RobustEncoder', RobustScaler())
    ])

    bin_encodes = ['Street', 'CentralAir', 'Utilities']
    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())
    ])

    onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            ('std', std_transformer, std_encodes),
            ('robust', rbst_transformer, robust_encodes),
            ('binary', binary_transformer, bin_encodes),
            ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),  #
        # 'Logistic Regression': LogisticRegression(),
        'LASSO': Lasso(),
        'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),  #
        # 'KNN Regression': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),  #
        # 'Naive Bayes': GaussianNB(),
        # 'Hidden Markov': hmm.GaussianHMM()
        'Elastic Net': ElasticNet()
    }

    # Choice of Model
    mdl_name = 'Elastic Net'

    # related tunes
    alpha = [0.01]
    l1_ratio = [0, 0.01, 0.02]
    fit_intercept = [True]
    normalize = [False]
    copy_X = [True]
    warm_start = [True]
    positive = [False]

    parameters = {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        'copy_X': copy_X,
        'warm_start': warm_start,
        'positive': positive
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pl.fit(X_train, y_train)

    scores = np.sqrt(-cross_val_score(pl, X=X_train, y=y_train, verbose=1, cv=5, scoring='neg_mean_squared_error'))

    cv_df = pd.DataFrame(scores.reshape(1, -1))
    cv_df.columns = ['cv' + str(i) for i in range(1, 6)]
    cv_df.index = ['RMSE']
    print(cv_df)
    final_score = np.mean(scores)
    print("cv score: {}".format(np.mean(scores)))

    print('参数选择:{}'.format(pl['regressor'].best_params_))

    prediction = pl.predict(X_test)
    RMSE = np.sqrt(np.mean((y_test - prediction) ** 2))
    print('RMSE on test set: {}'.format(RMSE))
    print('Pipeline score: {}'.format(pl.score(X_test, y_test)))

    return RMSE

# Bayesian Ridge
def br_model(df, col, tst):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    # encode
    label_encodes = ['GarageCars', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']  # Label Encoder
    label_transformer = Pipeline(steps=[
        ('LabelEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    def take_log(column):
        return np.log(column)

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(take_log)),
        ('StdEncoder', StandardScaler())
    ])

    std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']  # standardize
    std_transformer = Pipeline(steps=[
        ('StdEncoder', StandardScaler())
    ])

    robust_encodes = ['YearRemodPass', 'RepYear']  # robust encoder
    rbst_transformer = Pipeline(steps=[
        ('RobustEncoder', RobustScaler())
    ])

    bin_encodes = ['Street', 'CentralAir', 'Utilities']
    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())
    ])

    onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            ('std', std_transformer, std_encodes),
            ('robust', rbst_transformer, robust_encodes),
            ('binary', binary_transformer, bin_encodes),
            ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),  #
        # 'Logistic Regression': LogisticRegression(),
        'LASSO': Lasso(),
        'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),  #
        # 'KNN Regression': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),  #
        # 'Naive Bayes': GaussianNB(),
        # 'Hidden Markov': hmm.GaussianHMM()
        'Elastic Net': ElasticNet(),
        'Bayesian Ridge': BayesianRidge()
    }

    # Choice of Model
    mdl_name = 'Bayesian Ridge'

    # related tunes
    alpha_1 = [0.00001, 0.001, 0.01, 0.1, 0.5, 1.0]
    alpha_2 = [0.00001, 0.001, 0.01, 0.1, 0.5, 1.0]
    lambda_1 = [0.00001, 0.001, 0.01, 0.1, 0.5, 1.0]
    lambda_2 = [0.00001, 0.001, 0.01, 0.1, 0.5, 1.0]
    compute_score = [True, False]
    fit_intercept = [True, False]
    normalize = [True, False]
    copy_X = [True, False]
    verbose = [True, False]

    parameters = {
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'compute_score': compute_score,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        'copy_X': copy_X,
        'verbose': verbose
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pl.fit(X_train, y_train)

    scores = np.sqrt(-cross_val_score(pl, X=X_train, y=y_train, verbose=1, cv=5, scoring='neg_mean_squared_error'))

    cv_df = pd.DataFrame(scores.reshape(1, -1))
    cv_df.columns = ['cv' + str(i) for i in range(1, 6)]
    cv_df.index = ['RMSE']
    print(cv_df)
    print("cv score: {}".format(np.mean(scores)))

    print('参数选择:{}'.format(pl['regressor'].best_params_))

    prediction = pl.predict(X_test)
    RMSE = np.sqrt(np.mean((y_test - prediction) ** 2))
    print('RMSE on test set: {}'.format(RMSE))
    print('Pipeline score: {}'.format(pl.score(X_test, y_test)))
    return RMSE


if __name__ == '__main__':
    lr_score = lr_model(train, 'SalePrice', test)
    ls_score = ls_model(train, 'SalePrice', test)
    rg_score = rg_model(train, 'SalePrice', test)
    en_score = en_model(train, 'SalePrice', test)
    # preds = en_score.predict(test)
    # test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    # submission.SalePrice = np.expm1(submission.SalePrice)
    # submission.to_csv('submission.csv', index=False)

    from trees import reg_output
    results = reg_output(train, test, glmm_encode)
    rf_score = results.rf_result
    dt_score = results.dt_result
    lg_score = results.lg_result
    bg_score = results.bg_result

    target_df = pd.DataFrame(
        {'RMSE': [lr_score, ls_score, rg_score, en_score, rf_score, dt_score, lg_score, bg_score]},
        index=['OLS', 'Lasso', 'Ridge', 'Elastic Net', 'Random Forest', 'Decision Tree', 'LGBM', 'Bagging Regression']
    )
    print(target_df)