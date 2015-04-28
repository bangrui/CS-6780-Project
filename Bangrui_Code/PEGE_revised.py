import copy
import numpy as np
import random
from math import sqrt

text_file1 = open("norm_mean.txt","r")
norm_mean = [float(d) for d in text_file1.read().split()]
text_file2 = open("norm_cov.txt","r")
norm_cov = [float(d) for d in text_file2.read().split()]
norm_cov = np.reshape(norm_cov,(len(norm_mean),len(norm_mean)))
dim = len(norm_mean)
text_file3 = open("RN.txt","r")
feature = [float(d) for d in text_file3.read().split()]
feature = np.reshape(feature,(len(feature) / dim,dim))

while True:
  temp = feature[random.sample(range(len(feature)),dim)]
  arm = np.dot(np.transpose(temp),temp)
  if min(np.linalg.eig(arm)[0]) > 0.05:
    break

exp_arm = copy.copy(temp)
exp_arm = feature[[0,1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,20,28,29,42]]


simu_result = []
repli_times = 100
time_period = 610
sigma = 0.8

count = 0
while True:
  if 20 * count + count * (count + 1) / 2 > time_period:
    break
  else:
    count = count + 1
count = count - 1
reminder = time_period - (count * 20 + count * (count + 1) / 2)

np.random.seed(1)
user_pref = np.random.multivariate_normal(norm_mean,norm_cov,repli_times)

for j in range(repli_times):
  result = 0
  reward = []
  arm_forward = []
  period = 0
  arm_id = -1
  for i in range(count):
  ## Exploration period
    for k in range(dim):
      arm_forward.append(exp_arm[k])
      temp = np.dot(user_pref[j],exp_arm[k]) + np.random.normal(0,sigma)
      reward.append(temp)
      result = result + temp
  ## Exploitation period
    period = period + 1
    for t in range(period):
      theta = np.asarray(np.linalg.lstsq(arm_forward,reward)[0])
      temp_reward = [np.dot(feature[t],theta) for t in range(len(feature))]
      arm_id = temp_reward.index(max(temp_reward))
      temp = np.dot(user_pref[j],feature[arm_id]) + np.random.normal(0,sigma)
      arm_forward.append(feature[arm_id])
      reward.append(temp)
      result = result + temp 
  for i in range(reminder):
    theta = np.asarray(np.linalg.lstsq(arm_forward,reward)[0])
    temp_reward = [np.dot(feature[t],theta) for t in range(len(feature))]
    arm_id = temp_reward.index(max(temp_reward))
    temp = np.dot(user_pref[j],feature[arm_id]) + np.random.normal(0,sigma)
    arm_forward.append(feature[arm_id])
    reward.append(temp)
    result = result + temp 
  simu_result.append(result)

mean = np.mean(simu_result)
var = np.var(simu_result)
lower_CI = mean - 1.96 * sqrt(var / repli_times)
upper_CI = mean + 1.96 * sqrt(var / repli_times)
print lower_CI,upper_CI
print 'The simulation average is %6.3f.' %(mean)
print 'The 95 percent CI is [%6.3f,%6.3f].' %(lower_CI,upper_CI)
