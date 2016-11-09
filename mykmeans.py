#!/usr/bin/python
#coding:utf-8

import numpy as np
import os
import sys
import mylog
import logging
import copy
import time
#np.random.seed(time.time())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Kmeans(object):
  def __init__(self, param = {}):
    self.param = copy.deepcopy(param)
    self.clusters = []

  def __str__(self):
    kmeans_str = '共有簇的的个数:' + str(len(self.clusters))
    for i in range(0, len(self.clusters)):
      kmeans_str += '\n第' + str(i) + '个簇:' + str(self.clusters[i])
    return kmeans_str
  def Fit(self, x):
    #clusters
    try:
      self.InitClusters(x.shape[1])
    except:
      logger.warn('数据集只有一个属性:%s', x.shape)
      self.InitClusters(1)
    logger.debug('共有簇的的个数:%d', len(self.clusters))
    #iter
    for i in range(0, self.param['max_iter']):
      self.ClearClusters()
      for j in range(0, len(x)):
        point = Point(x[j])
        self.DispatchPoint(point)
      #重新计算质心
      self.CalCentroids()
      #检查是否有空簇
      self.CheckCentroids()
      logger.info('iteration:%d SSE:%f', i, self.CalSSE())
  def Predict(self, x):
    pred = np.array([-1] * len(x), dtype = np.int)
    for i in range(0, len(x)):
      cluster = self.FindNearestCluster(Point(x[i]))
      pred[i] = cluster.GetLabel()
    return pred
  def CheckCentroids(self):
    for cluster in self.clusters:
      if cluster.GetPointsNum() <= 0:
        p = self.GetPointFromMaxSSECluster()
        cluster.SetCentroid(p)
  def GetPointFromMaxSSECluster(self):
    max_sse_cluster = None
    max_sse = sys.float_info.min
    for cluster in self.clusters:
      sse = cluster.CalSSE()
      if sse > max_sse:
        max_sse = sse
        max_sse_cluster = cluster
    #随机选其中一个点
    idx = np.random.randint(max_sse_cluster.GetPointsNum())
    return max_sse_cluster.GetPoint(idx)
  def InitClusters(self, size):
    #clusters
    logger.debug('n_clusters:%d', self.param['n_clusters'])
    for i in range(0, self.param['n_clusters']):
      logger.debug('初始化cluster:%d', i)
      cluster = Cluster(i, size)
      cluster.RandCentroid()
      self.clusters.append(cluster)
    return self.clusters
  def ClearClusters(self):
    for cluster in self.clusters:
      cluster.Clear()
  def CalCentroids(self):
    for cluster in self.clusters:
      cluster.CalCentriod()
  def DispatchPoint(self, p):
    cluster = self.FindNearestCluster(p)
    cluster.AddPoint(p)

  def FindNearestCluster(self, p):
    nearest = sys.float_info.max
    nearest_cluster = None
    for cluster in self.clusters:
      dist =  Kmeans.EuclideanDistance(p, cluster.GetCentroid())
      if nearest > dist:
#        logger.debug('dist:%f', dist)
        nearest = dist
        nearest_cluster = cluster
#    logger.debug('nearest dist:%f, nearest_cluster:%d', nearest, nearest_cluster.GetLabel())
    assert(not nearest_cluster == None)
    assert(nearest >= 0.0)
    return nearest_cluster
     
  @classmethod
  def EuclideanDistance(self, p1, p2):
    return Kmeans.Distance(2, p1.GetVector(), p2.GetVector())
  @classmethod
  def Distance(self, p, x1, x2):
    vector_x1 = np.array(x1)
    vector_x2 = np.array(x2)
    return np.sum(np.abs(vector_x1 - vector_x2) ** p)**(1.0/p)

  def GetClusterCenters(self):
    return self.clusters
  def CalSSE(self):
    sse = 0.0
    for cluster in self.clusters:
      sse+=cluster.CalSSE()
    return sse

class Cluster(object):
  def __init__(self, label, point_size):
    self.points = []
    self.centroid = None
    self.point_size = point_size
    self.label = label
  def __str__(self):
    return 'centroid:' + str(self.centroid) + '.点个数:' + str(len(self.points)) + '.'
  def GetPoint(self, i):
    return self.points[i]
  def GetPointsNum(self):
    return len(self.points)
  def GetLabel(self):
    return self.label
  def Clear(self):
    self.points = []
  def GetCentroid(self):
    return self.centroid
  def SetCentroid(self, p):
    self.centroid = p
    return self.centroid
  def AddPoint(self, p):
    if not p == None:
      self.points.append(p)
  def RandCentroid(self):
    self.centroid = Point.NewRandomPoint(self.point_size)
    return self.centroid
  def CalCentriod(self):
    self.centroid = self.CalMuPoint()
    return self.centroid
  def CalMuPoint(self):
    p = Point.NewOriginPoint(self.point_size)
    if len(self.points) <= 0:
#      logger.warn('簇中没有点')
      return None
    for point in self.points:
#      logger.debug('before:%s', p)
      p += point
#      logger.debug(' after:%s', p)
#    logger.debug('簇中点个数:%d', len(self.points))
    p = p * (1.0 / len(self.points))
    return p
  def CalSSE(self):
    sse = 0.0
    for point in self.points:
      sse += Kmeans.EuclideanDistance(point, self.centroid) ** 2
    self.sse = sse
    return self.sse

class Point(object):
  def __init__(self, x):
    self.x = np.array(x)
    try:
      self.point_size = self.x.shape[1]
    except:
      self.poing_size = 1
  def __mul__(self, k):
    return Point(self.x * (1.0*k))

  def __str__(self):
    return str(self.x)
  def __eq__(self, p):
    if p == None:
      return False
    return list(self.x) == list(p.x)
  def GetVector(self):
    return self.x
  def GetDimension(self):
    return len(self.x)
  @classmethod
  def NewRandomPoint(self, size):
    return Point(np.random.random_sample(size))
  @classmethod
  def NewPoint(self, x):
    return Point(x)
  @classmethod
  def NewOriginPoint(self, size):
    return Point([0.0] * size)
  def __add__(self, p):
    return Point(self.x + p.x)

if __name__== '__main__':
#  dist = Distance(2, [1, 1], [2, 2])
#  logger.debug(dist)
  param = {}
  param['n_clusters'] = 10
  param['max_iter'] = 100
  kmeans = Kmeans(param)
  kmeans.Fit(x)
  pred = kmeans.Predict(x)
  logger.info('pred:%s', pred)

