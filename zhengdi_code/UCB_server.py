import sys
import numpy as np

def ALG():
    return 'UCB'

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

        #print 'mu0:', self.mu0
        #print 'S0', self.S0

    def recommand1(self):
        bound = 0
        mm = 0
        vv = 0
        best_feature = []
        
        for feature in self.featurelist:
            feature = np.matrix(feature)
            m = float(self.mu * feature.T)
            v = float(feature * self.S * feature.T)
            b = m + 1.96 * np.sqrt(v)
            if b > bound:
                bound = b
                mm = m
                vv = v
                best_feature = feature
        idx = self.featurelist.index(best_feature.tolist()[0])
        #print idx, 'bound', bound, 'mean', mm, 'var', v
        return best_feature.tolist()[0]

    def recommand2(self):
        distribution = []
        for feature in self.featurelist:
            feature = np.matrix(feature)
            m = float(self.mu * feature.T)
            v = float(feature * self.S * feature.T)
            b = m + 1.96*np.sqrt(v)
            distribution.append(b)

        s = sum(distribution)
        weights = [x/s for x in distribution]
        cs = np.cumsum(weights)
        idx = sum(cs < np.random.rand())
        recommand = featurelist[idx]

        return recommand

    def recommand(self):
        recommand0 = self.recommand1()
        return recommand0

    def update(self, recommand_feature, rate_score):
        matrix_recommand_feature = np.matrix(recommand_feature)
        sigma2 = self.sigma * self.sigma
        #print 'Sinv', self.Sinv
        #print 'XTX', matrix_recommand_feature.T * matrix_recommand_feature / sigma2
        self.Sinv += matrix_recommand_feature.T * matrix_recommand_feature / sigma2
        
        self.Sinvmu += matrix_recommand_feature.T * rate_score / sigma2
        self.S = np.linalg.inv(self.Sinv)
        self.mu = self.S * self.Sinvmu
        self.mu = self.mu.T
       
        #self.mu = (np.linalg.solve(self.Sinv, self.Sinvmu)).T
        idx = self.featurelist.index(recommand_feature)
        #print idx, rate_score
