#!/usr/bin/python
#coding:utf-8
import numpy as np
import logging
import mylog
import mykmeans as ml
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
  filename = './data/movement_libras.data'
  train_data = np.loadtxt(filename, delimiter = ',')
  logger.debug(train_data)
  logger.debug(train_data.shape)

  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]
  logger.debug(train_x)
  logger.debug(train_y)

  param = {}
  param['n_clusters'] = 15
  param['max_iter'] = 100
  kmeans = ml.Kmeans(param)
  kmeans.Fit(train_x)
  logger.debug(kmeans)
  pred = kmeans.Predict(train_x)
  logger.info('train_y:%s', train_y)
  logger.info('   pred:%s', pred)
  logger.info('k-means准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
