# -*- coding: utf-8 -*-
"""
Runs 1 run of the specified ML training and evaluation

"""
import re
from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from run_globals import DATA_DIR
import os

def RunMLRun(pathData, pathLabel, f_select, model, evl, ensemble_n=30, n_splits=10):
    """ 
    Trains and evaluates a machine learning model. Returns metrics, and models
    """
    testData = os.path.join(DATA_DIR, 'canbind_X_overlap_tillwk4_qids_sr.csv') # X data matrix over CAN-BIND, only overlapping features with STAR*D, subjects who have qids sr until at least week 4
    if evl == "extval_resp":
        testLabel = os.path.join(DATA_DIR, 'canbind_y_tillwk8_resp_qids_sr.csv') # y matrix from canbind, with subjects as above, targetting week 8 qids sr response
    elif evl == "extval_rem":
        testLabel = os.path.join(DATA_DIR, 'canbind_y_tillwk8_rem_qids_sr.csv') # y matrix from canbind, with subjects as above, targetting week 8 qids sr remission
    elif evl == "extval_rem_randomized": # A control to make sure our extval_rem results are robust, with the targets scrambled randomly
        testLabel = os.path.join(DATA_DIR, 'canbind_y_tillwk8_randomized.csv') # y matrix from canbind, with subjects as above, with targets scrambled
    elif evl == "cv": # Use randomized as a placeholder, won't be used for cv
        testLabel = os.path.join(DATA_DIR, 'canbind_y_tillwk8_randomized.csv') # y matrix from canbind, with subjects as above, with targets scrambled
 
    # read data and chop the header
    X_test = np.genfromtxt(testData, delimiter=',')[1:,1:]
    y_test = np.genfromtxt(testLabel, delimiter=',')[1:,1]
    
    X = np.genfromtxt(pathData, delimiter=',')
    y = np.genfromtxt(pathLabel, delimiter=',')[1:,1]
    X = X[1:,1:]
    
    n,m = X.shape
    kf = KFold(n_splits, shuffle=True)

    j=1
    accu = np.empty([10,], dtype=float)
    auc = np.empty([10,], dtype=float)
    bscore = np.empty([10,], dtype=float)
    specificity = np.empty([10,], dtype=float)
    sensitivity = np.empty([10,], dtype=float)
    precision = np.empty([10,], dtype=float)
    f1 = np.empty([10,], dtype=float)
    features_n = np.empty([10,], dtype=float)
    tps = np.empty([10,], dtype=float)
    fps = np.empty([10,], dtype=float)
    tns = np.empty([10,], dtype=float)
    fns = np.empty([10,], dtype=float)
    feature_importances =  np.empty([10,m], dtype=float) #Store feature importances relative to the original ordering.
    clfs = [None]*n_splits
        
    for train_index, test_index in kf.split(X):
        print("Fold:", j)
        
        if evl == 'cv':
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        elif evl == 'extval_resp' or evl == 'extval_rem' or 'extval_rem_randomized':
            X_train, _ = X[train_index], X[test_index]
            y_train, _ = y[train_index], y[test_index]
        else:
            Exception("Invalid evaluation type provided, must be cv or extval")

        # Feature selection
        if f_select == "chi":
            features = featureSelectionChi(X_train,y_train,30,50)
        elif f_select == "elas":
            features = featureSelectionELAS(X_train,y_train,31)
        elif f_select == "agglo":
            features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),20)
        elif f_select == "all":
            features = np.arange(m)
            
        # Array to store the importance of features, whether feature was used, and an int to stores number of features per classifier
        feature_importance = np.zeros(len(features))
        features_n_fold = 0
        
        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 classifiers
        
        clf = [None]*ensemble_n
        for i in range(ensemble_n):
            if model == "rf":
                clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = -1)
                clf[i].fit(training[i][:,features],label[i])

            elif model == "elnet":
                clf[i] = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.67, alpha=0.1, max_iter=10000, power_t=0.01)
                clf[i].fit(training[i][:,features],label[i])

            elif model == "gbdt":
                clf[i] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
                clf[i].fit(training[i][:,features],label[i])

            elif model == 'l2logreg':
                clf[i] = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=10000, C=0.092)
                clf[i].fit(training[i][:,features],label[i])
                
            elif model == "xgbt":
                param = {'nthread':-1, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1} # Original tuning by Joey

                num_round = 5            
                dtrain = xgb.DMatrix(training[i][:,features], label=label[i])
                bst = xgb.train(param, dtrain, num_round)
                clf[i] = bst
        
        # Prediction
        n = X_test.shape[0]
        if model == "xgbt":
            pred_prob = np.zeros((n,))
            dtest = xgb.DMatrix(X_test[:,features], label=y_test)
        else:
            pred_prob = np.zeros((n,2))
            
        # calculting the average probabilty of each class, as well as feature use/importance
        for i in range(ensemble_n):
            if model == "xgbt":
                pred_prob += clf[i].predict(dtest) # Bug before, was clf[0]
                feature_importance += xgbt_feature_importance(len(features), clf[i]) # Bug before, was clf[0]
            else:
                pred_prob += clf[i].predict_proba(X_test[:,features])
            if model == "rf" or model == "gbdt":
                # Feature Importance for tree-based methods in sklearn
                feature_importance += clf[i].feature_importances_
                features_n_fold += np.count_nonzero(clf[i].feature_importances_) 
            elif model == "elnet" or model == "l2logreg":
                # Feature importance for linear methods in sklearn                
                feature_importance += clf[i].coef_.flatten()
                features_n_fold += np.count_nonzero(clf[i].coef_) 
                
            
        pred_prob = pred_prob/ensemble_n
        # Pick the class with the greastest probability to be the prediction
        if model == "xgbt":
            pred = (pred_prob>=0.5)
            y_score = pred_prob
        else:
            pred = np.argmax(pred_prob,axis=1)
            y_score = pred_prob[:,1]

        # Store performance metrics accuracy and draw ROC curve
        auc[j-1] = drawROC(y_test, y_score)
        bscore[j-1] = balanced_accuracy_score(y_test, pred)
        tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
        specificity[j-1] = tn/(tn+fp)
        sensitivity[j-1] = tp/(tp+fn)
        precision[j-1] = tp/(tp+fp)
        tns[j-1] = tn/n
        fps[j-1] = fp/n
        fns[j-1] = fn/n
        tps[j-1] = tp/n
        f1[j-1] = 2*precision[j-1]*sensitivity[j-1]/(precision[j-1]+sensitivity[j-1])
        feature_importances[j-1,features] = feature_importance/ensemble_n
        features_n[j-1] = features_n_fold/ensemble_n
        score = sum(pred==y_test)/n
        accu[j-1] = score
        clfs[j-1] = clf

        j = j+1
    
    confus_mat ={}
    confus_mat['tp'] = sum(tps)/10
    confus_mat['fp'] = sum(fps)/10
    confus_mat['tn'] = sum(tns)/10
    confus_mat['fn'] = sum(fns)/10
    
    avg_accu = sum(accu)/10
    avg_bal_acc = sum(bscore)/10
    avg_auc = sum(auc)/10
    avg_sens = sum(sensitivity)/10
    avg_spec = sum(specificity)/10
    avg_prec = sum(precision)/10
    avg_f1 = sum(f1)/10
    avg_features_n = sum(features_n)/10
    avg_feature_importance = np.sum(feature_importances,axis=0)/10
    
    return(avg_accu, avg_bal_acc, avg_auc, avg_sens, avg_spec, avg_prec, avg_f1, avg_features_n, avg_feature_importance, confus_mat, clfs)


def xgbt_feature_importance(n_features, clf, impt_type='gain'):
    """
    Helper function to return feature importance from a xgboost classifier, 
    as xgbt does not have a built in feature_importance_
    
    Returns: feature_importance, the importance from this classifier
    """
    
    ft_impt_dict = clf.get_score(importance_type=impt_type)
    feature_importance = np.zeros(n_features)
    
    for key in ft_impt_dict:
        # Xgbt outputs a dict with feature importance coded with keys such as 
        # {'f379: 86.5'} so first we need to parse out that index number
        match = re.search(r'f(\d{1,3})', key)
        ft_index = int(match.group(1))  
        #print(ft_index)
        # Then add it
        feature_importance[ft_index] = feature_importance[ft_index] + ft_impt_dict[key]
    
    return feature_importance
