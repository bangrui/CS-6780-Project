import sys
import numpy as np
from math import e

def ALG():
    return 'EGA'

class server:
    def __init__(self, mean_prior, cov_prior, sigma, featurelist, nite):
        self.mu0 = np.matrix(mean_prior)
        self.mu = self.mu0
        self.S0 = np.matrix(cov_prior)
        self.S = self.S0
        self.featurelist = featurelist
        self.sigma = sigma
        
        self.Sinv = np.linalg.inv(self.S)
        self.Sinvmu = self.Sinv * self.mu.T
        
        self.nite = nite
        self.D = 4
        self.narms = len(featurelist)
        self.weights = [1.0/self.narms for x in range(self.narms)]
        self.gamma = min(1.0, np.sqrt(self.narms * np.log(self.narms * 1.0) / (e-1) / self.D / nite))
        #print 'gamma:', self.gamma
        #self.eta = np.sqrt(8 * np.log(self.narms * 1.0) / self.nite)
        self.eta = 0.5




    def recommand(self):
        self.prob = [ (1-self.gamma) * x + self.gamma / self.narms for x in self.weights]
        #print 'prob =', self.prob
        cs = np.cumsum(self.prob)
        #np.random.seed(1)
        idx = sum(cs < np.random.rand())
        recommand = self.featurelist[idx]
        #print 'recommand:', self.featurelist.index(recommand)
        return recommand


    def update(self, recommand_feature, rate_score):
        C = self.D
        loss = (C - rate_score)
        idx = self.featurelist.index(recommand_feature)
        #print idx, loss, self.weights
        self.weights[idx] *= np.exp(-self.eta * loss / self.prob[idx])
        s = sum(self.weights)
        self.weights = [x/s for x in self.weights]
