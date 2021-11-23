# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:21:40 2020

Grid search script used for finding some of the model parameters.

@author: jjnun
"""

from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np

pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\X_lvl2_rem_qids01__final.csv'
pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\y_lvl2_rem_qids01__final.csv'

X = np.genfromtxt(pathData, delimiter=',')
y = np.genfromtxt(pathLabel, delimiter=',')[1:,1]
X = X[1:,1:]

"""
model = SGDClassifier(loss='log', penalty='elasticnet', max_iter=100)
grid = GridSearchCV(estimator=model, n_jobs=6, scoring='balanced_accuracy', param_grid={
    'l1_ratio': [p/1000 for p in range(650,705,5)],
    'max_iter': [10000],
    'alpha': [0.05,0.06,0.07,0.08,0.09, 0.1,0.11],
    'power_t':[0.005,0.08, 0.09,0.01,0.011, 0.012]})
grid.fit(X,y)
print(grid)
# summarize the previous results of the grid search
print(f' The best balanced accuracy: {grid.best_score_}')
print(f'The best l1_ratio {grid.best_estimator_.l1_ratio}')
print(f'The best max_iter {grid.best_estimator_.max_iter}')
print(f'The best alpha {grid.best_estimator_.alpha}')
print(f'The best power_t {grid.best_estimator_.power_t}')
"""

"""
model = LogisticRegression(penalty='l2',solver='lbfgs')
grid = GridSearchCV(estimator=model, n_jobs=6, scoring='balanced_accuracy', param_grid={
    'tol' : [0.1, 0.01, 0.001, 'none'],
    'max_iter': [10000, 20000, 100000]})
    #'C': [p/1000 for p in range(90, 120, 1)]})
    #'solver': ['lbfgs', saga, liblinear]})

grid.fit(X,y)
print(grid)
# summarize the previous results of the grid search
print(f'The best balanced accuracy: {grid.best_score_:.4f}')
print(f'The best max_iter {grid.best_estimator_.max_iter}')
print(f'The best tol {grid.best_estimator_.tol}')
#print(f'The best C {grid.best_estimator_.C}')
#print(f'The best solver {grid.best_estimator_.solver}')
"""
xgb_model = xgb.XGBClassifier()
grid = GridSearchCV(xgb_model, n_jobs=1, 
                   scoring='balanced_accuracy',
                   param_grid={
                       'nthread':[15], #when use hyperthread, xgboost may become slower
                       'objective':['binary:logistic'],
                       'booster' : ['gbtree'],
                       'learning_rate': [0.05], #so called `eta` value
                       'max_depth': [6],
                       'min_child_weight': [11],
                       'silent': [1],
                       'subsample': [0.5],
                       'colsample_bytree': [0.8],
                       'n_estimators': [500], #number of trees, change it to 1000 for better previous results
                       'missing':[-999],
                       'seed': [1337],
                       'reg_lambda':[6,7,8,9,10,11,12,13,14,15],
                       'reg_alpha':[6,7,8,9,10,11,12,13,14,15]})
                  
#param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}


grid.fit(X,y)
print(grid)
# summarize the previous results of the grid search
print(f'The best balanced accuracy: {grid.best_score_:.4f} trying to be 0.70')
print(f'The best learning_rate {grid.best_estimator_.learning_rate}')
print(f'The best max_depth {grid.best_estimator_.max_depth}')
print(f'The best colsample_bytree {grid.best_estimator_.colsample_bytree}')
print(f'The best n_estimators {grid.best_estimator_.n_estimators}')
print(f'The best subsample {grid.best_estimator_.subsample}')
print(f'The best alpha {grid.best_estimator_.reg_alpha}')
print(f'The best lambda {grid.best_estimator_.reg_lambda}')