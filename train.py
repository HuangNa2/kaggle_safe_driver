#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nathan
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import autoencoder

pd.set_option('display.max_columns', None)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

all_cols = list(train.columns.values)
feat_cols = [x for x in all_cols if x not in ['id','target']]

x_train = train[feat_cols]
x_test = test[feat_cols]
y_train = train['target']

encoder = autoencoder.train_autoencoder(x_test,x_train)   
x_test_encode  = autoencoder.encode_features(x_test  , encoder)
x_train_encode = autoencoder.encode_features(x_train , encoder)

x_train = pd.concat([x_train,x_train_encode], axis = 1)
x_test = pd.concat([x_test,x_test_encode], axis = 1)

params = {'max_depth':7, 
        'eta':0.05, 
        'silent':1, 
        'objective':'binary:logistic', 
        'alpha' :4, 
        'lambda':10,
        'min_child_weight' : 1,
        'gamma' : 2,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'num_parallel_tree' : 3,
        'eval_metric' : 'auc'}

        
clf = xgb.cv(params, xgb.DMatrix(x_train,y_train), 300, nfold=5,
       metrics={'auc'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

bst_rounds = clf['test-auc-mean'].idxmax(axis = 1)

np.random.seed(0)
model = xgb.train(params, xgb.DMatrix(x_train,y_train), bst_rounds, metrics={'auc'}, verbose_eval = 1)

final_submission = pd.read_csv('./sample_submission.csv')
test['target'] = model.predict(xgb.DMatrix(test.ix[:,test.columns != 'id']))

final_submission = pd.merge(final_submission[['id']],test[['id','target']], how='left')

final_submission.to_csv('./submission_scores.csv',index = False)


## END 

#test = []
#str_list = ['bin','car','calc']
#
#for string in str_list:
#    filtered = [x for x in cols if string in x]
#    test.extend(filtered)
#
#train.groupby(['target']).size()

#pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)


#
#
#def add_feats(data):
#    data['num_na'] = (data == -1).sum(axis = 1)
#    data['high_na'] = data['num_na'].map(lambda x : 1 if x > 4 else 0)
#    data['ps_car_13_ps_reg_03'] = data[['ps_car_13','ps_reg_03']].sum(axis = 1)
#    data['ps_reg_mult'] = data[['ps_reg_01','ps_reg_02','ps_reg_03']].prod(axis=1)
#    data['ps_ind_bin_sum'] = data[['ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin']].sum(axis = 1)
#    return(data)
#
#x_train = add_feats(x_train)
#x_test = add_feats(x_test)
