#!/usr/bin/python
#coding:utf-8
import numpy as np
import logging
import mylog
import mykmeans as ml
import myplot
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def PickingRightK(x):
  param = {}
  param['use_random_for_k'] = 1
#  param['k'] = []
  param['n_clusters'] = 5
  param['max_iter'] = 100
  ks = [1] + [i for i in range(2, 30, 2)]
  dias = []
  sses = []
  kmeans = ml.Kmeans(param)
  for k in ks:
    kmeans.SetK(k)
    kmeans.Fit(train_x)
    dia = kmeans.CalAverageDiameter()
    dias.append(dia)
    sse = kmeans.CalSSE()
    sses.append(sse)
  #plot
  print sses
  print dias
  myplot.Plot2DLine(ks, dias, 'K','average diameter','average diameter and K')
  myplot.Plot2DLine(ks, sses, 'K', 'SSE', 'SSE and K')
  myplot.Show()
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
  param['n_clusters'] = 8
  param['max_iter'] = 100
  kmeans = ml.Kmeans(param)
#  kmeans.BisectingFit(train_x)
  kmeans.Fit(train_x)
#  logger.debug(kmeans)
  dia = kmeans.CalAverageDiameter()
  logger.info('diameter:%f', dia)
#  pred = kmeans.Predict(train_x)
#  logger.info('train_y:%s', train_y)
#  logger.info('   pred:%s', pred)
#  logger.info('k-means准确率:%f', 1.0*sum(pred == train_y)/len(train_y))

  PickingRightK(train_x)
