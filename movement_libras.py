#!/usr/bin/python
#coding:utf-8
import numpy as np
import logging
import mylog
import mykmeans as ml
import myplot
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

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
  param['use_random_for_k'] = 1
  param['k'] = [1,25, 49,73,97,121,145,169,193,217,241,265,289,313,337]
  param['n_clusters'] = 15
  param['max_iter'] = 100
  kmeans = ml.Kmeans(param)
#  kmeans.BisectingFit(train_x)
  kmeans.Fit(train_x)
  logger.info('SSE:%f', kmeans.CalSSE())
#  logger.debug(kmeans)
#  dia = kmeans.CalAverageDiameter()
#  logger.info('diameter:%f', dia)
#  pred = kmeans.Predict(train_x)
#  logger.info('train_y:%s', train_y)
#  logger.info('   pred:%s', pred)
#  logger.info('k-means准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
  ml.PickingRightK(train_x, param)

  #第三问
#  import myplot
#  myplot.Figure()
#  ml.FitMulti(train_x, param, 100)
#  ml.BisectingFitMulti(train_x, param, 100)
#  myplot.Legend(['k-means','bisecting'])
#  myplot.Title('libras movement')
#  myplot.Show()

