#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:56:26 2018

@author: loandnt
"""


# coding: utf-8

# ### Read data

import pandas as pd

movie_info = pd.read_csv('/home/loandnt/Documents/seminar/movie_rating/pre-processed/movie_all.csv').iloc[:, 1:]

movie_info.head()

movie_info.columns
# #### train test seperate

feature_cols = ['category_Animation', 'category_Children\'s', 'category_Comedy',
       'category_Crime', 'category_Documentary', 'category_Drama',
       'category_Fantasy', 'category_Film-Noir', 'category_Horror',
       'category_Musical', 'category_Mystery', 'category_Romance',
       'category_Sci-Fi', 'category_Thriller', 'category_War',
       'category_Western']

x_train = movie_info[feature_cols].loc[:int(0.8 * len(movie_info))]
print len(x_train)
x_train.head()

y_train = movie_info[['mean']].loc[:int(0.8 * len(movie_info))]
print len(y_train)
y_train.head()


x_test = movie_info[feature_cols].loc[int(0.8 * len(movie_info)):]
print len(x_test)
x_test.head()

y_test = movie_info[['mean']].loc[int(0.8 * len(movie_info)):]
print len(y_test)
y_test.head()
"""
x_train.to_csv('../../data/input_formated/v0/x_train.csv')
y_train.to_csv('../../data/input_formated/v0/y_train.csv')
x_test.to_csv('../../data/input_formated/v0/x_test.csv')
y_test.to_csv('../../data/input_formated/v0/y_test.csv')
"""
# predict = ann(Variable(torch.from_numpy(np.array(x_test, dtype=np.float32))))
# criterion = nn.MSELoss()
# # targets = Variable(torch.from_numpy(np.array(y_test, dtype=np.float32)))
# loss = criterion(predict, targets)
# print loss
