#!/usr/bin/python
#coding:utf-8
import numpy as np
import logging
import mylog
import mykmeans as ml
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def str2num(s):
  a = ['very_low', 'Low', 'Middle', 'High']
  for i in range(0, len(a)):
    if a[i] == s:
      return float(i)
if __name__ == '__main__':
  filename = './data/data_user_modeling.txt'
  train_data = np.loadtxt(filename, delimiter = ',', converters = {5:str2num})
  logger.debug(train_data)
  logger.debug(train_data.shape)

  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]
  logger.debug(train_x)
  logger.debug(train_y)

  param = {}
  param['use_random_for_k'] = 1
  param['k'] = [i for i in range(0, 258, 1)]
  param['n_clusters'] = 4
  param['max_iter'] = 100
  kmeans = ml.Kmeans(param)
  kmeans.Fit(train_x)
#  logger.debug(kmeans)
  pred = kmeans.Predict(train_x)
  logger.info('train_y:%s', train_y)
  logger.info('   pred:%s', pred)
#  logger.info('k-means准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
