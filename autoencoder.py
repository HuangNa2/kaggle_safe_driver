#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:53:30 2017

@author: nathan
"""

import pandas as pd
from numpy.random import seed
from keras.layers import Input, Dense
from keras.models import Model
#from keras.callbacks import TensorBoard


#pd.set_option('display.max_columns', None)
#
#train = pd.read_csv('./train.csv')
#test = pd.read_csv('./test.csv')
#
#all_cols = list(train.columns.values)
#feats_cols = [x for x in all_cols if x not in ['id','target']]
#
#x_test_data = test[feats_cols]
#x_train_data = train[feats_cols]

def train_autoencoder(train,test):
    train = train.as_matrix()
    test = test.as_matrix()
    
    seed(38) # Seedmaster
    iL = Input(shape = (57,))
    
    encodedL = Dense(40, activation = 'relu') (iL)
    encodedL = Dense(25, activation = 'relu') (encodedL)
    encodedL = Dense(18, activation = 'relu') (encodedL)
    
    decodedL = Dense(25, activation = 'relu') (encodedL)
    decodedL = Dense(40, activation = 'relu') (decodedL)
    decodedL = Dense(57, activation = 'linear') (decodedL)
    
    autoencoder = Model(iL,decodedL)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    autoencoder.fit(train, train,
                    epochs=15,
                    batch_size=256,
                    shuffle=False,
                    validation_data=(test, test))
                    #,
                    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    
    encoder = Model(iL,encodedL)
    return(encoder)

def encode_features(data, model_encoder):
    data = data.as_matrix()
    encoded_feats = pd.DataFrame(model_encoder.predict(data))

    # Some neurons were 0, dirty ass dropping
    all0 = (encoded_feats != 0).sum(axis = 0)
    encoded_feats = encoded_feats.drop(all0[all0 == 0].index, axis = 1)
    
    # Rename cols
    encoded_feats.columns = ['encoded_var{0}'.format(x+1) for x in range(0,encoded_feats.shape[1])]
                    
    return(encoded_feats)
    

                         