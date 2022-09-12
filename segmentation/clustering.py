import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt




def organize_by_lines(data, n_points):
  res_dict={}
  
  for item in data:
    try:

        res_dict[item[1]].append(item[2])
    except KeyError:
        res_dict[item[1]] = [item[2]]
  return res_dict

"""
def get_data_organized(data, n_points):
  lines = {}
  for i in range(0, len(data)):
    organized_b_lines = organize_by_lines2(data[i], n_points);
    try:
      lines[i].append(organized_b_lines);
    except KeyError:
      lines[i] = (organized_b_lines);

  return lines
"""
def get_avg_5(data):
  aux = {}
  colors = list(data.keys())
  for line in range(0, len(colors)):
    avg_values = []
    for frame in range(0, len(data[colors[0]])):
      if line == 0 or line == 1:
        if line == 0:
          prev = data[colors[len(colors)-1]][frame]
        else:
          prev = data[colors[0]][frame]
        prev2 = data[colors[len(colors)-2+line]][frame]
        avg_value = (prev2 +prev + data[colors[line]][frame] + data[colors[line+1]][frame]+data[colors[line+2]][frame])/5
      elif line == (len(colors)-1) or line == (len(colors)-2):
        next = data[colors[0]][frame]
        if line == len(colors)-2:
          next2 = data[colors[len(colors)-1]][frame]
        else: 
          next2 = data[colors[1]][frame]
        avg_value = (next + next2 + data[colors[line]][frame] + data[colors[line-1]][frame]+data[colors[line-2]][frame])/5
      else:
        avg_value =(data[colors[line-1]][frame]+ data[colors[line-2]][frame] + data[colors[line]][frame] + data[colors[line+1]][frame] + data[colors[line+2]][frame])/5
      try:
        aux[colors[line]].append(avg_value)
      except KeyError:
        aux[colors[line]] = [avg_value]
  return aux


def get_avg_values(data):
  avg_values = []
  aux = {}
  for key in data:
    avg = sum(list(data[key]))/len(data[key])
    try:
      aux[key].append(avg)
    except KeyError:
      aux[key] = [avg]
  return aux


def avg_zero(data, avg_values):
  aux = {}
  aux = data
  
  aux2 = {}
  for key in aux:
    avg_value = avg_values[key][0]
    
    for i in range(0, len(data[key])):
      value = aux[key][i]-avg_value
      #aux2[:] = [value-avg_values[key][0] for value in aux2]
      try:
        aux2[key].append(value)
      except KeyError:
        aux2[key] = []
        aux2[key].append(value)
  return aux2


def get_array_cluster(data):
  array_values = []
  for key in data:    
    array_values.append(data[key])   
  return np.array(array_values)

def clustering(array_to_cluster, max_iter,n_init, n_clusters):
  kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, n_init=n_init, random_state=0)
  pred_y = kmeans.fit_predict(array_to_cluster)
  print(pred_y)
  print(kmeans.cluster_centers_)
  return pred_y

def data_to_cluster(data,n_points, n_clusters, max_iter,n_init):
    organized_by_lines = organize_by_lines(data,n_points)
    data_avg_5 = get_avg_5(organized_by_lines)
    avg_values_5 = get_avg_values(data_avg_5)
    data_avg_zero = avg_zero(data_avg_5,avg_values_5)
    array_to_cluster = get_array_cluster(data_avg_zero)
    pred_y = clustering(array_to_cluster,max_iter,n_init,n_clusters)
    return pred_y