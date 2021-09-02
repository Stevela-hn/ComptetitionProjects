# imports
import numpy as np
import pandas as pd
import os
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

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
from sklearn.ensemble import BaggingRegressor


from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from scipy.stats import loguniform
from sklearn.decomposition import TruncatedSVD
from scipy import stats

from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

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
    test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].median())

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

    for k in robust_encodes:
        train[k] = train[k].fillna(train[k].median())
        test[k] = test[k].fillna(test[k].median())

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

# for i in std_encodes + robust_encodes:
#     if any(test[i].isna()):
#         print(i)
# print(glmm_encode)
# Decision Tree Regressor
# take_log
for col in ['GrLivArea', 'GarageArea', 'LotFrontage']:
    train[col] = train[col].apply(lambda x: np.log(x) if x != 0 else 0)
    test[col] = test[col].apply(lambda x: np.log(x) if x != 0 else 0)


def dt_model(df, col, tst, glmm_encode):
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
        return column.apply(lambda x: np.log(x) if x != 0 else 0)

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
        'Linear Regression': LinearRegression(),
        # 'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state = 66),
        'Random Forest': RandomForestRegressor(random_state = 63),
    }

    # Choice of Model
    mdl_name = 'Decision Tree'

    # related tunes
    criterion = ['mse']
    max_depth = [8]
    min_samples_split = [23]
    min_samples_leaf = [11]

    parameters = {
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

# Random Forest Regressor
def rf_model(df, col, tst, glmm_encode):
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
        return column.apply(lambda x: np.log(x) if x != 0 else 0)

    def besame(column):
        return column

    log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
    log_transformer = Pipeline(steps=[
        ('Transfer', FunctionTransformer(besame)),
        # ('StdEncoder', StandardScaler())
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

    onehot_encodes = ['LotConfig', 'HouseStyle', 'MSZoning', 'Exterior1st', 'BldgType',
                      'Neighborhood', 'ExterQual', 'HeatingQC', 'GarageFinish', 'Condition1',
                      'Foundation', 'SaleType', 'Functional', 'SaleCondition', 'LandContour', 'PavedDrive']
    # onehot_encodes = glmm_encode
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
        'Linear Regression': LinearRegression(),
        # 'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state = 66),
        'Random Forest': RandomForestRegressor(n_estimators = 50, max_depth = 11, random_state = 10),
    }

    # Choice of Model
    mdl_name = 'Random Forest'

    # related tunes
    n_estimators = (10, 250)
    criterion = ['mse']

    max_depth = (5, 15)
    min_samples_split = (2, 25)
    # min_samples_leaf =
    max_features = (0.1, 0.999)

    parameters = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'max_features': max_features,
        # 'criterion': criterion,
        'max_depth': max_depth,
        # 'min_samples_leaf': min_samples_leaf,
    }

    def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
        return RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),  # float
                               max_depth=int(max_depth),
                               random_state=2
        )

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', RandomForestRegressor(n_estimators = 50, max_depth = 11, random_state = 10))
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

    return pl.predict(test)


# LightGB Regressor
def lg_model(df, col, tst, glmm_encode):
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

    onehot_encodes = ['LotConfig', 'HouseStyle', 'MSZoning', 'Exterior1st', 'BldgType',
                      'Neighborhood', 'ExterQual', 'HeatingQC', 'GarageFinish', 'Condition1',
                      'Foundation', 'SaleType', 'Functional', 'SaleCondition', 'LandContour', 'PavedDrive']
    # onehot_encodes = glmm_encode
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
        'Linear Regression': LinearRegression(),
        # 'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state = np.random.randint(100)),
        'Random Forest': RandomForestRegressor(random_state = np.random.randint(100)),
        'LightGBM': lgb.LGBMRegressor(random_state=np.random.randint(100))
    }

    # Choice of Model
    mdl_name = 'LightGBM'

    # related tunes
    n_estimators = [100]
    max_depth = [-1]
    num_leaves = [31]
    subsample = [0.6]
    colsample_bytree = [0.5]
    learning_rate = [0.05]

    parameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'learning_rate': learning_rate
    }

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('regressor', GridSearchCV(candidates[mdl_name], param_grid=parameters, cv=5))
    ])

    # Cross Validation(交叉验证), split dataset
    X = df.drop('SalePrice', axis=1)
    y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

    return pl.predict(test)

# Bagging Regressor
def bg_model(df, col, tst, glmm_encode):
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

    onehot_encodes = ['LotConfig', 'HouseStyle', 'MSZoning', 'Exterior1st', 'BldgType',
                      'Neighborhood', 'ExterQual', 'HeatingQC', 'GarageFinish', 'Condition1',
                      'Foundation', 'SaleType', 'Functional', 'SaleCondition', 'LandContour', 'PavedDrive']
    # onehot_encodes = glmm_encode
    onehot_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre_proc = ColumnTransformer(
        transformers=[
            # ('label', label_transformer, label_encodes),
            ('log', log_transformer, log_encodes),
            # ('std', std_transformer, std_encodes),
            # ('robust', rbst_transformer, robust_encodes),
            # ('binary', binary_transformer, bin_encodes),
            # ('onehot', onehot_transformer, onehot_encodes)
        ]
    )

    # possible choice of models
    candidates = {
        'Linear Regression': LinearRegression(),
        # 'LASSO': Lasso(),
        # 'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state = 66),
        'Random Forest': RandomForestRegressor(random_state = 65),
        'Bagging': BaggingRegressor()
    }

    # Choice of Model
    mdl_name = 'Bagging'

    # related tunes
    n_estimators = [23]
    bootstrap = [True, False]
    oob_score = [True, False]
    criterion = ['mse']

    max_depth = [17]
    min_samples_split = [2]
    min_samples_leaf = [1]

    parameters = {
        # 'n_estimators': n_estimators,
        # 'bootstrap': bootstrap,
        # 'oob_score': oob_score,
        # 'criterion': criterion,
        # 'max_depth': max_depth,
        # 'min_samples_split': min_samples_split,
        # 'min_samples_leaf': min_samples_leaf
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

    return

class reg_output:

    def __init__(self, train, test, glmm_encode):
        self.train = train
        self.test = test
        self.gl = glmm_encode
        # self.rf_result = rf_model(train, 'SalePrice', test, glmm_encode)
        # self.dt_result = dt_model(train, 'SalePrice', test, glmm_encode)
        # self.lg_result = lg_model(train, 'SalePrice', test, glmm_encode)
        # self.bg_result = bg_model(train, 'SalePrice', test, glmm_encode)

if __name__ == '__main__':
    # dt_model(train, 'SalePrice', test, glmm_encode)
    # rf_model(train, 'SalePrice', test, glmm_encode)
    # lg_model(train, 'SalePrice', test, glmm_encode)
    sub = pd.DataFrame()
    # bg_model(train, 'SalePrice', test, glmm_encode)
    # sub['Id'] = IDS
    # sub['SalePrice'] = rf_model(train, 'SalePrice', test, [])
    # sub.to_csv('price_pred.csv', index=False)


# experiment on bayesian optimization
def trans(df):
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

    onehot_encodes = ['LotConfig', 'HouseStyle', 'MSZoning', 'Exterior1st', 'BldgType',
                      'Neighborhood', 'ExterQual', 'HeatingQC', 'GarageFinish', 'Condition1',
                      'Foundation', 'SaleType', 'Functional', 'SaleCondition', 'LandContour', 'PavedDrive']
    # onehot_encodes = glmm_encode
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
    return pre_proc, pre_proc.fit_transform(df)

log_encodes = ['GrLivArea', 'GarageArea', 'LotFrontage']
std_encodes = ['TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']
robust_encodes = ['YearRemodPass', 'RepYear']
bin_encodes = ['Street', 'CentralAir', 'Utilities']
onehot_encodes = ['LotConfig', 'HouseStyle', 'MSZoning', 'Exterior1st', 'BldgType',
                      'Neighborhood', 'ExterQual', 'HeatingQC', 'GarageFinish', 'Condition1',
                      'Foundation', 'SaleType', 'Functional', 'SaleCondition', 'LandContour', 'PavedDrive']


X = train.drop('SalePrice', axis=1)
y = train.SalePrice

_, new_final = trans(X)

def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    metrics = cross_val_score(
        RandomForestRegressor(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            max_depth=int(max_depth),
            random_state=2
        ),
        new_final,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    ).mean()
    return metrics

rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)}
)
rf_bo.maximize()
target_diction = rf_bo.max['params']
for key in target_diction:
    if key != 'max_features':
        target_diction[key] = [int(target_diction[key])]
    else:
        target_diction[key] = [target_diction[key]]

def rf_test(df, params, X, y):
    """
    This method helps encode or preprocess each of the features,
    making them milder to fit into the model, then input into the
    pipeline
    :param df: input cleaned data
    :param col: target variable to predict
    :return: predicted 1-D data
    """

    prep, _ = trans(df)

    mdl = RandomForestRegressor()

    # PPL
    pl = Pipeline(steps=[
        ('preprocessor', prep),
        ('regressor', GridSearchCV(mdl, params, cv=5))
    ])

    # Cross Validation, split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

    return pl.predict(test)

rf_test(train, target_diction, X, y)