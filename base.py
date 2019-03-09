# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:13:46 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import seaborn as sns
#import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from catboost import CatBoostClassifier
import statsmodels.api as sm
import eli5
from eli5.sklearn import PermutationImportance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
import CyclicLR
from scipy.stats import norm, rankdata
import gc
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
#                else:
#                    print('WRONG!!!')
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)
#                else:
#                    print('WRONG!!')
    return df



def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            
            model = lgb.train(params,
                    train_data,
                    num_boost_round=20000,
                    valid_sets = [train_data, valid_data],
                    verbose_eval=1000,
                    early_stopping_rounds = 200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid).reshape(-1,)
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')
            
            y_pred = model.predict_proba(X_test)[:, 1]
            
        if model_type == 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1,)
            score = roc_auc_score(y_valid, y_pred_valid)
            
            y_pred = model_results.predict(X_test)
            
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss',  eval_metric='AUC', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]
            
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values  
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction, scores
    
    else:
        return oof, prediction, scores


def calculate_metrics(model, X_train: pd.DataFrame() = None, y_train: pd.DataFrame() = None, X_valid: pd.DataFrame() = None,
                      y_valid: pd.DataFrame() = None, columns: list = []) -> pd.DataFrame():
    columns = columns if len(columns) > 0 else list(X_train.columns)
    train_pred = model.predict_proba(X_train[columns])
    valid_pred = model.predict_proba(X_valid[columns])
    f1 = 0
    best_t = 0
    for t in np.arange(0.1, 1, 0.05):
        valid_pr = (valid_pred[:, 1] > t).astype(int)
        valid_f1 = metrics.f1_score(y_valid, valid_pr)
        if valid_f1 > f1:
            f1 = valid_f1
            best_t = t

    t = best_t
    train_pr = (train_pred[:, 1] > t).astype(int)
    valid_pr = (valid_pred[:, 1] > t).astype(int)
    train_f1 = metrics.f1_score(y_train, train_pr)
    valid_f1 = metrics.f1_score(y_valid, valid_pr)
    score_df = []
    print(f'Best threshold: {t:.2f}. Train f1: {train_f1:.4f}. Valid f1: {valid_f1:.4f}.')
    score_df.append(['F1', np.round(train_f1, 4), np.round(valid_f1, 4)])
    train_r = metrics.recall_score(y_train, train_pr)
    valid_r = metrics.recall_score(y_valid, valid_pr)

    score_df.append(['Recall', np.round(train_r, 4), np.round(valid_r, 4)])
    train_p = metrics.precision_score(y_train, train_pr)
    valid_p = metrics.precision_score(y_valid, valid_pr)

    score_df.append(['Precision', np.round(train_p, 4), np.round(valid_p, 4)])
    train_roc = metrics.roc_auc_score(y_train, train_pred[:, 1])
    valid_roc = metrics.roc_auc_score(y_valid, valid_pred[:, 1])

    score_df.append(['ROCAUC', np.round(train_roc, 4), np.round(valid_roc, 4)])
    train_apc = metrics.average_precision_score(y_train, train_pred[:, 1])
    valid_apc = metrics.average_precision_score(y_valid, valid_pred[:, 1])

    score_df.append(['APC', np.round(train_apc, 4), np.round(valid_apc, 4)])
    print(metrics.confusion_matrix(y_valid, valid_pr))
    score_df = pd.DataFrame(score_df, columns=['Metric', 'Train', 'Valid'])
    print(score_df)

    return score_df, t


#class Simple_NN(nn.Module):
#    def __init__(self, input_dim, hidden_dim, dropout=0.75):
#        super(Simple_NN, self).__init__()
#        self.input_dim = input_dim
#        self.hidden_dim = hidden_dim
#        self.relu = nn.ReLU()
#        self.dropout = nn.Dropout(dropout)
#        self.fc1 = nn.Linear(1, hidden_dim)
#        self.fc2 = nn.Linear(int(hidden_dim*input_dim), 1)
#    
#    def forward(self, x):
#        b_size = x.size(0)
#        x = x.view(-1, 1)
#        y = self.fc1(x)
#        y = self.relu(y)
#        y = y.view(b_size, -1)
#        out = self.fc2(y)
#        return out
#
#def sigmoid(x):
#    return 1 / (1+np.exp(-x))
#
#def trainNN(train_features, train_target, test_features, splits):
#    n_epochs = 40
#    batch_size = 256
#    train_preds = np.zeros(len(train_features))
#    test_preds = np.zeros(len(test_features))
#    X_test = np.array(test_features)
#    X_test = torch.Tensor(X_test)
#    test = torch.utils.data.TensorDataset(X_test)
#    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
#    avg_losses_f = []
#    avg_val_losses_f = []
#    
#    for i, (train_idx, valid_idx) in enumerate(splits):
#        x_train = np.array(train_features)
#        y_train = np.array(train_target)
#        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.float32)
#        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32)
#        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.float32)
#        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32)
#        loss_fn = torch.nn.BCEWithLogitsLoss()
#        model = Simple_NN(200,16)
#        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001,weight_decay=1e-5)
#        
#        ######################Cycling learning rate########################
#        step_size = 2000
#        base_lr, max_lr = 0.001, 0.005
#        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)
#        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='exp_range', gamma=0.99994)
#        
#        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
#        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
#    
#        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
#    
#        print(f'Fold {i + 1}')
#        for epoch in range(n_epochs):
#            start_time = time.time()
#            model.train()
#            avg_loss = 0.
#            #avg_auc = 0.
#            for i, (x_batch, y_batch) in enumerate(train_loader):
#                y_pred = model(x_batch)
#                ###################tuning learning rate###############
#                if scheduler:
#                    #print('cycle_LR')
#                    scheduler.batch_step()
#    
#                ######################################################
#                loss = loss_fn(y_pred, y_batch)
#    
#                optimizer.zero_grad()
#                loss.backward()
#    
#                optimizer.step()
#                avg_loss += loss.item()/len(train_loader)
#                #avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
#            model.eval()
#            
#            valid_preds_fold = np.zeros((x_val_fold.size(0)))
#            test_preds_fold = np.zeros((len(test_features)))
#            
#            avg_val_loss = 0.
#            #avg_val_auc = 0.
#            for i, (x_batch, y_batch) in enumerate(valid_loader):
#                y_pred = model(x_batch).detach()
#                
#                #avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
#                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
#                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
#                
#            elapsed_time = time.time() - start_time 
#            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
#                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
#            
#        avg_losses_f.append(avg_loss)
#        avg_val_losses_f.append(avg_val_loss) 
#        
#        for i, (x_batch,) in enumerate(test_loader):
#            y_pred = model(x_batch).detach()
#    
#            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
#            
#        train_preds[valid_idx] = valid_preds_fold
#        test_preds += test_preds_fold / len(splits)
#    
#    auc  =  round(roc_auc_score(train_target,train_preds),4)      
#    print('All \t loss={:.4f} \t val_loss={:.4f} \t auc={:.4f}'.format(np.average(avg_losses_f),np.average(avg_val_losses_f),auc))
#    return train_preds, test_preds

###############################################################################
###############################################################################
#####################  Start  #################################################
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#train = reduce_mem_usage(train)
#test = reduce_mem_usage(test)

X = train.drop(['ID_code', 'target'], axis=1)
y = train['target']
X_test = test.drop(['ID_code'], axis=1)


########Computing new features#############
merged = pd.concat([X, X_test])
for col in merged.columns:
    # Normalize the data, so that it can be used in norm.cdf(), 
    # as though it is a standard normal variable
    merged[col] = ((merged[col] - merged[col].mean()) 
    / merged[col].std()).astype('float32')

    # Square
    merged[col+'^2'] = merged[col] * merged[col]

    # Cube
    merged[col+'^3'] = merged[col] * merged[col] * merged[col]

    # 4th power
    merged[col+'^4'] = merged[col] * merged[col] * merged[col] * merged[col]

    # Cumulative percentile (not normalized)
    merged[col+'_cp'] = rankdata(merged[col]).astype('float32')

    # Cumulative normal percentile
    merged[col+'_cnp'] = norm.cdf(merged[col]).astype('float32')

X = merged[:200000]
X_test = merged[200000:]
del merged
gc.collect()

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)
splits = list(StratifiedKFold(n_splits=5, shuffle=True).split(X, y))
#train_nn_preds, test_nn_preds = trainNN(X, y, X_test, splits)
params = {'num_leaves': 8,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}

#params = { "objective" : "binary",
#            "metric" : "auc",
#            "max_depth" : 2,
#            "num_leaves" : 2,
#    		"learning_rate" : 0.055,
#    		"bagging_fraction" : 0.3,
#    		"feature_fraction" : 0.15,
#    		"lambda_l1" : 5,
#    		"lambda_l2" : 5,
#    		"bagging_seed" : 1,
#    		"verbosity" : 1,
#    		"seed": 2
#        }

oof_lgb, prediction_lgb, scores = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
sub = pd.read_csv('data/sample_submission.csv')
sub['target'] = prediction_lgb
sub.to_csv('lgb2.csv', index=False)

###ELI5
#model = lgb.LGBMClassifier(**params, n_estimators = 20000, n_jobs = -1)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
#model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=1000, early_stopping_rounds=200)
#
#####ELI5 didn't help up to eliminate features, but let's at least try to take top-100 and see how it helps.
#feature_weight = eli5.show_weights(model, targets=[0, 1], feature_names=list(X_train.columns), top=40, feature_filter=lambda x: x != '<BIAS>')
#
#top_features = [i for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature if 'BIAS' not in i][:100]
#X1 = X[top_features]
#X_train, X_valid, y_train, y_valid = train_test_split(X1, y, test_size=0.2, stratify=y)
#model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=1000, early_stopping_rounds=200)
#
#cal_result = calculate_metrics(model, X_train, y_train, X_valid, y_valid)
#
#X = train.drop(['ID_code', 'target'], axis=1)
#X_test = test.drop(['ID_code'], axis=1)
#
#columns = top_features = [i for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature if 'BIAS' not in i][:20]
#for col1 in columns:
#    for col2 in columns:
#        X[col1 + '_' + col2] = X[col1] * X[col2]
#        X_test[col1 + '_' + col2] = X_test[col1] * X_test[col2]
#        
#        
###Scaling, Notice scaling severely decreases score
#X = train.drop(['ID_code', 'target'], axis=1)
#X_test = test.drop(['ID_code'], axis=1)
#scaler = StandardScaler()
#X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
#X_test[X_train.columns] = scaler.transform(X_test[X_train.columns])
#
#oof_lgb, prediction_lgb, scores = train_model(np.round(X, 4), np.round(X_test, 4), y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
#sub = pd.read_csv('data/sample_submission.csv')
#sub['target'] = prediction_lgb
#sub.to_csv('lgb_rounded.csv', index=False)