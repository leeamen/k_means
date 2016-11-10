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

def sort_key(x):
  return x.GetPointsNum()

class Kmeans(object):
  def __init__(self, param = {}):
    self.param = copy.deepcopy(param)
    self.clusters = []

  def __str__(self):
    kmeans_str = '共有簇的的个数:' + str(len(self.clusters))
    for i in range(0, len(self.clusters)):
      kmeans_str += '\n第' + str(i) + '个簇:' + str(self.clusters[i])
    return kmeans_str
  def SetK(self, k):
    self.param['n_clusters'] = k
  def InitOneCluster(self, x):
    cluster = Cluster(0, x.shape[1])
    self.clusters = []
    self.clusters.append(cluster)
    for i in range(0, len(x)):
      cluster.AddPoint(Point(x[i]))
  #2分kmeans
  def BisectingFit(self, x):
    self.InitOneCluster(x)
    n = self.param['n_clusters']
    i = 1
    while i < n:
      self.BisectingCluster()
      logger.info('第%d次二分 SSE:%f', i, self.CalSSE())
      i+=1
  def BisectingCluster(self):
    self.clusters.sort(key = sort_key, reverse = True)
#    sum = 0
#    for cluster in self.clusters:
#      sum += cluster.GetPointsNum()
#      logger.debug('点个数:%d', cluster.GetPointsNum())
#    logger.debug('sum:%d', sum)
#    logger.debug('簇个数:%d', len(self.clusters))
    cluster = self.clusters.pop(0)
    backup_clusters = self.clusters
    self.clusters = []
    points = cluster.GetPoints()
#    logger.debug('点个数:%d', len(points))
    for i in range(0, 2):
      new_cluster = Cluster(len(backup_clusters)+i, cluster.GetPointSize())
      new_cluster.RandCentroid()
      self.clusters.append(new_cluster)
    #iteration
    last_sse = 0.0
    for i in range(0, self.param['max_iter']):
      self.ClearClusters()
      for j in range(0, len(points)):
        self.DispatchPoint(points[j])
      #计算新的中心
      self.CalCentroids()
      #检查是否有空簇
      self.CheckCentroids()
      this_sse = self.CalSSE()
#      logger.info('Bisecting iteration:%d SSE:%f', i, this_sse)
      if last_sse == this_sse :
#        logger.info('Bisecting iteration over!')
        break
      last_sse = this_sse
    self.clusters = backup_clusters + self.clusters

  def Fit(self, x):
    #clusters
    self.InitClusters(x)
#    logger.debug('共有簇的的个数:%d', len(self.clusters))
    #iter
    last_sse = 0.0
    for i in range(0, self.param['max_iter']):
      self.ClearClusters()
      for j in range(0, len(x)):
        point = Point(x[j])
        self.DispatchPoint(point)
      #重新计算质心
      self.CalCentroids()
      #检查是否有空簇
      self.CheckCentroids()
      this_sse = self.CalSSE()
      logger.info('iteration:%d SSE:%f', i, this_sse)
      if last_sse == this_sse:
        logger.info('iteration over!')
        break
      last_sse = this_sse
  def Predict(self, x):
    pred = np.array([-1] * len(x), dtype = np.int)
    for i in range(0, len(x)):
      cluster = self.FindNearestCluster(Point(x[i]))
      pred[i] = cluster.GetLabel()
    return pred
  def CalAverageDiameter(self):
    ave_dia = 0.0
    for i in range(0, len(self.clusters)):
      cluster = self.clusters[i]
      max_dia = 0.0
      for j in range(0, cluster.GetPointsNum()):
        for k in range(j+1, cluster.GetPointsNum()):
          dist = self.EuclideanDistance(cluster.GetPoint(j), cluster.GetPoint(k))
          if max_dia < dist:
            max_dia = dist
      ave_dia+=max_dia
#      logger.debug('max_dia:%f', max_dia)
    ave_dia /= float(len(self.clusters))
#    logger.info('ave_dia:%f', ave_dia)
    return ave_dia
  def CheckCentroids(self):
    for cluster in self.clusters:
      if cluster.GetPointsNum() <= 0:
        p = self.GetPointFromMaxSSECluster()
        cluster.SetCentroid(p)
  def GetPointFromMaxSSECluster(self):
    max_sse_cluster = None
    max_sse = -1.0#sys.float_info.min
#    logger.debug('here.%d',len(self.clusters))
    for cluster in self.clusters:
      sse = cluster.CalSSE()
#      logger.debug(sse)
      if sse > max_sse:
        max_sse = sse
        max_sse_cluster = cluster
    #随机选其中一个点
#    logger.debug(max_sse_cluster.GetPointsNum())
    idx = np.random.randint(max_sse_cluster.GetPointsNum())
    return max_sse_cluster.GetPoint(idx)
  def InitClusters(self, x):
    #clusters
    self.clusters = []
#    logger.debug('n_clusters:%d', self.param['n_clusters'])
    for i in range(0, self.param['n_clusters']):
#      logger.debug('初始化cluster:%d', i)
      cluster = Cluster(i, x.shape[1])
      if self.param.has_key('use_random_for_k') is True:
        cluster.RandCentroid()
      elif self.param.has_key('k') is True:
        cluster.SetCentroid(Point(x[self.param['k'][i]]))
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
    assert(not p is None)
    nearest = sys.float_info.max
    nearest_cluster = None
    for cluster in self.clusters:
      assert(not cluster is None)
      dist =  Kmeans.EuclideanDistance(p, cluster.GetCentroid())
      if nearest > dist:
#        logger.debug('dist:%f', dist)
        nearest = dist
        nearest_cluster = cluster
#    logger.debug('nearest dist:%f, nearest_cluster:%d', nearest, nearest_cluster.GetLabel())
    assert(not nearest_cluster is None)
    assert(nearest >= 0.0)
    return nearest_cluster
     
  @classmethod
  def EuclideanDistance(self, p1, p2):
    assert(not p1 is None)
    assert(not p2 is None)
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
    self.sse = -1
  def __str__(self):
    return 'centroid:' + str(self.centroid) + '.点个数:' + str(len(self.points)) + '.'
  def GetPointSize(self):
    return self.point_size
  def GetPoint(self, i):
    return self.points[i]
  def GetPoints(self):
    return self.points
  def GetPointsNum(self):
    return len(self.points)
  def GetLabel(self):
    return self.label
  def Clear(self):
    self.points = []
    self.sse = -1
    self.has_sse = False
  def GetCentroid(self):
    return self.centroid
  def SetCentroid(self, p):
    self.centroid = p
    return self.centroid
  def AddPoint(self, p):
    if p is not None:
      self.points.append(p)
  def RandCentroid(self):
    self.centroid = Point.NewRandomPoint(self.point_size)
    return self.centroid
  def CalCentriod(self):
    centroid = self.CalMuPoint()
    if centroid is not None:
      self.centroid = centroid
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
    if self.has_sse is True:
      return self.sse
    if self.centroid is None:
      logger.debug(self.sse)
      logger.debug(len(self.points))
      logger.error('error,centroid没有计算,%s', str(self.centroid))
      exit()
    sse = 0.0
    for point in self.points:
      if self.centroid is None:
        logger.error('error,centroid没有计算,%s', str(self.centroid))
        exit()
      sse = sse + Kmeans.EuclideanDistance(point, self.centroid) ** 2
    self.sse = sse
    self.has_sse = True
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
    
#  def __eq__(self, p):
  def Equal(self, p):
    if p is None:
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

