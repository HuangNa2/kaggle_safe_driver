#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:36:02 2017

@author: nathan
"""

import os
import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv('./train.csv')

cols = list(train.columns.values)

test = []
str_list = ['bin','car','calc']

for string in str_list:
    filtered = [x for x in cols if string in x]
    test.extend(filtered)
