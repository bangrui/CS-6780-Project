import copy
import numpy as np
from math import sqrt
from numpy.linalg import inv

text_file1 = open("norm_mean.txt","r")
norm_mean = [float(d) for d in text_file1.read().split()]
text_file2 = open("norm_cov.txt","r")
norm_cov = [float(d) for d in text_file2.read().split()]
norm_cov = np.reshape(norm_cov,(len(norm_mean),len(norm_mean)))
text_file3 = open("RN.txt","r")
feature = [float(d) for d in text_file3.read().split()]
feature = np.reshape(feature,(len(feature) / len(norm_mean),len(norm_mean)))

np.random.seed(1)

num_of_feature = feature.shape[0]
simu_result = []
repli_times = 10
time_period = 2275
sigma = 0.8

user_pref = np.random.multivariate_normal(norm_mean,norm_cov,repli_times)

print user_pref

for j in range(repli_times): 
  temp_norm_mean = copy.copy(norm_mean)
  temp_norm_cov = copy.copy(norm_cov)
  result = 0
  for t in range(time_period):
    ucb = []
    for i in range(num_of_feature):
      ucb.append(np.dot(temp_norm_mean,feature[i]) + 1.96 * sqrt(np.dot(np.dot(feature[i],temp_norm_cov),feature[i])))
    arm_id = ucb.index(max(ucb))
    reward = np.dot(user_pref[j],feature[arm_id]) + np.random.normal(0,sigma)
    result = result + reward
    new_norm_cov = inv(inv(temp_norm_cov) + np.mat(np.reshape(feature[arm_id],(20,1)) * np.mat(feature[arm_id])) / (sigma * sigma)) 
    temp_norm_mean = np.reshape(np.dot(new_norm_cov,np.dot(inv(temp_norm_cov),np.reshape(temp_norm_mean,(20,1)))+np.reshape(np.dot(feature[arm_id],reward / (sigma * sigma)),(20,1))),(1,20))
    temp_norm_cov = copy.copy(new_norm_cov)
  simu_result.append(result)

mean = np.mean(simu_result)
var = np.var(simu_result)
lower_CI = mean - 1.96 * sqrt(var / repli_times)
upper_CI = mean + 1.96 * sqrt(var / repli_times)
print lower_CI,upper_CI
print 'The simulation average is %6.3f.' %(mean)
print 'The 95 percent CI is [%6.3f,%6.3f].' %(lower_CI,upper_CI)
