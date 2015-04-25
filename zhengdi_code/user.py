import sys
import numpy as np


class user:

    def __init__(self, mean, cov, sigma):
        self.mean = mean
        self.cov = cov
        self.mu0 = np.matrix(mean)
        self.S0 = np.matrix(cov)
        self.sigma = sigma

    def generate_user(self):
        self.theta = np.random.multivariate_normal(self.mean, self.cov)
        #print 'User:', self.theta

    def rate(self, feature):
        #print 'feature:', len(feature)
        #print 'theta:', len(self.theta)
        #np.random.seed(1)
        reward = np.dot(feature, self.theta) + np.random.normal(0, self.sigma)
        return reward

    def maxrate(self, featurelist):
        m = 0
        for f in featurelist:
            temp = np.dot(self.theta, np.array(f))
            if temp > m:
                m = temp
        return m
