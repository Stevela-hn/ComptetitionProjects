# NumPy
import numpy as np

# params adjust
import optuna

# Dataframe operations
import pandas as pd
import plotly
from sklearn.metrics import accuracy_score
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix

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

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
IDS = test_df.PassengerId
data_df = train_df.append(test_df) # The entire data: train + test.

data_df['Title'] = data_df['Name']
# Cleaning name and extracting Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
           'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

# Dropping Title feature
data_df.drop('Title', axis=1, inplace=True)

data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:",
      data_df.loc[data_df['Family_Survival'] != 0.5].shape[0])

for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passenger with family/group survival information: "
      + str(data_df[data_df['Family_Survival'] != 0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)

data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)

X = train_df.drop('Survived', 1)
y = train_df['Survived']

class data_output:
    def __init__(self):
        self.train_x = X
        self.train_y = y

X_test = test_df.copy()

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                           weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test)

# xgboost
import xgboost as xgb
xgc = xgb.XGBClassifier()

# --------------------------------------------------------------------------------
def objective(trial):
    data = train_df.drop('Survived', 1)
    target = train_df.Survived
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=42)
    param = {
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': 4000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBClassifier(**param)
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)
    return acc

study = optuna.create_study(direction='maximize')
n_trials=50
study.optimize(objective, n_trials=n_trials)
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")
print('Best parameters', study.best_params)
print("------------------------------------------------")

# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_parallel_coordinate(study).show()
# optuna.visualization.plot_param_importances(study).show()

gdxg_opt=xgb.XGBClassifier(**study.best_params)
gdxg_opt.fit(X, y)
y_pred_xg_opt = gdxg_opt.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_xg_opt
submit.to_csv('submission_XGB_OPT.csv', index=False)

# --------------------------------------------------------------------------------

hyperparams = {
    'learning_rate': [0.1],
    'max_depth': [4],
    'min_child_weight': [3],
    'gamma': [0],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'reg_alpha': [1]
}
gdxg=GridSearchCV(estimator = xgc, param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gdxg.fit(X, y)
print(gdxg.best_params_)
y_pred_xg = gdxg.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_xg
submit.to_csv('submission_XGB_NONOPT.csv', index=False)

# random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# --------------------------------------------------------------------------------
def objective(trial):
    data = train_df.drop('Survived', 1)
    target = train_df.Survived
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=42)
    param = {
        'n_estimators': trial.suggest_categorical('n_estimators', [10, 20, 30, 40, 50, 60, 70, 80]),
        'max_depth': trial.suggest_categorical('max_depth', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [2, 3, 4, 5, 6, 7, 8, 9, 10])
    }
    model = RandomForestClassifier(**param)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)
    return acc

study = optuna.create_study(direction='maximize')
n_trials=50
study.optimize(objective, n_trials=n_trials)
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")
print('Best parameters', study.best_params)
print("------------------------------------------------")

# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_parallel_coordinate(study).show()
# optuna.visualization.plot_param_importances(study).show()

rf_opt= RandomForestClassifier(**study.best_params)
rf_opt.fit(X, y)
y_pred_rf_opt = rf_opt.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_rf_opt
submit.to_csv('submission_RF_OPT.csv', index=False)

# --------------------------------------------------------------------------------

hyperparams = {
    'n_estimators':[41],
    'max_depth':[5],
    'min_samples_split':[4],
    'min_samples_leaf':[2],
    'max_features':[3]
}
rf_grid = GridSearchCV(estimator=rfc, param_grid=hyperparams, verbose=True, cv=10, scoring='roc_auc')
rf_grid.fit(X, y)
print(rf_grid.best_params_)
y_pred_rf = rf_grid.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

hyperparams = {
    'max_depth': [2,3,4,5,6,7,8],
    # 'min_samples_split': [0.2, 0.4, 0.6, 0.8],
    # 'min_samples_leaf': [4]
}
dtc_grid = GridSearchCV(estimator=dtc, param_grid=hyperparams, verbose=True, cv=10, scoring='roc_auc')
dtc_grid.fit(X, y)
print(dtc_grid.best_params_)
y_pred_dtc = dtc_grid.predict(X_test)


# Light GBM
from lightgbm import LGBMClassifier
lgc = LGBMClassifier()

# --------------------------------------------------------------------------------
def objective(trial):
    data = train_df.drop('Survived', 1)
    target = train_df.Survived
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=42)
    param = {
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20, 50]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100)
    }
    model = LGBMClassifier(**param)
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)
    return acc

study = optuna.create_study(direction='maximize')
n_trials=50
study.optimize(objective, n_trials=n_trials)
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")
print('Best parameters', study.best_params)
print("------------------------------------------------")

lg_grid_opt=LGBMClassifier(**study.best_params)
lg_grid_opt.fit(X, y)
y_pred_lg_opt = lg_grid_opt.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_lg_opt
submit.to_csv('submission_LGB_OPT.csv', index=False)

# --------------------------------------------------------------------------------

hyperparams = {
    'max_depth': [3],
    'min_child_weight': [7],
    'subsample': [0.2],
    'colsample_bytree': [0.6],
    'learning_rate': [0.1]
}
lg_grid = GridSearchCV(estimator=lgc, param_grid=hyperparams, verbose=True, cv=10, scoring='roc_auc')
lg_grid.fit(X, y)
print(lg_grid.best_params_)
y_pred_lg = lg_grid.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_lg
submit.to_csv('submission_XGB_NONOPT.csv', index=False)

# ADA Boost
from sklearn.ensemble import AdaBoostClassifier
adac = AdaBoostClassifier()
hyperparams = {
    'base_estimator': [
        RandomForestClassifier(n_estimators=41, max_depth=5, min_samples_split=4, min_samples_leaf=2, max_features=3),
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, min_samples_split=0.2)
    ],
    'n_estimators': [41],
    'learning_rate': [0.1]
}
ada_grid = GridSearchCV(estimator=adac, param_grid=hyperparams, verbose=True, cv=10, scoring='roc_auc')
ada_grid.fit(X, y)
print(ada_grid.best_params_)
y_pred_ada = ada_grid.predict(X_test)

# --------------------------------------------------------------------------------
def objective(trial):
    data = train_df.drop('Survived', 1)
    target = train_df.Survived
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25, random_state=42)
    param = {
        'base_estimator': RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_split=4, min_samples_leaf=2, max_features=3),
        'n_estimators': trial.suggest_int('n_estimators', 1, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.05, 0.15)
    }
    model = AdaBoostClassifier(**param)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)
    return acc

study = optuna.create_study(direction='maximize')
n_trials=100
study.optimize(objective, n_trials=n_trials)
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")
print('Best parameters', study.best_params)
print("------------------------------------------------")

ada_grid_opt=AdaBoostClassifier(**study.best_params)
ada_grid_opt.fit(X, y)
y_pred_ada_opt = ada_grid_opt.predict(X_test)

submit = pd.DataFrame()
submit['PassengerId'] = IDS
submit['Survived'] = y_pred_ada_opt
submit.to_csv('submission_ADA_OPT.csv', index=False)

# --------------------------------------------------------------------------------

class h2_output:

    def __init__(self):
        self.h2_pred = y_pred
        self.knn_mdl = knn
        self.xg_pred = y_pred_xg
        self.rf_pred = y_pred_rf
        # self.lg_pred = y_pred_lg
        self.ada_pred = y_pred_ada
        self.dtc_pred = y_pred_dtc
        self.y_pred_rf_opt = y_pred_rf_opt
        self.ada_pred_opt = y_pred_ada_opt