import sys
import numpy as np

def ALG():
    return 'PEGE'


def collect_base(list):
    """ list is n by m
        need to find m bases
    """
    n = len(list)
    m = len(list[0])
    base = []
    r = 0
    idx = 0
    while r < m and idx < n:
        base.append(list[idx])
        R = np.linalg.matrix_rank(np.matrix(base))
        idx += 1
        if R == r:
            base = base[0:-1]
        else:
            r = R
            #print idx
    if r == m:
        return base
    else:
        print 'the rank is not full'
        return base

class server:
    def __init__(self, mean_prior, cov_prior, sigma, featurelist, nite):
        self.n_arms = len(featurelist)
        self.r = len(featurelist[0])
        self.step_in_cycle = 0
        self.cycle = 1
        #self.Z0 = np.matrix(mean_prior)
        self.sum_bY = np.matrix([0.0 for x in range(self.r)])
        self.Z = np.matrix([0.0 for x in range(self.r)])
        
        self.base = collect_base(featurelist)
        self.b = np.matrix(self.base) # each row is a featue vector
        self.bb = self.b.T * self.b
        self.bbinv = np.linalg.inv(self.bb)
        self.lambdamin = min(np.linalg.eigvalsh(self.bb))
        #print 'lambda:', self.lambdamin
        
        self.featurelist = featurelist
    

    def recommand(self):
        #exploration in r periods:
        if self.step_in_cycle < self.r:
            recommand_feature = self.b[self.step_in_cycle]
            recommand_feature = recommand_feature.tolist()
            recommand_feature = recommand_feature[0]
        
        #exploitation in c periods:
        else:
            idx = 0
            m = 0
            for i in range(self.n_arms):
                f = np.matrix(self.featurelist[i])
                temp = float(f * self.Z.T)
                if temp > m:
                    m = temp
                    idx = i
            recommand_feature = self.featurelist[idx]
            
        return recommand_feature

    def update(self, recommand_feature, rate_score):
        idx = self.featurelist.index(recommand_feature)
        #print 'cycle', self.cycle, 'id:', idx, 'rating:', rate_score
        if self.step_in_cycle < self.r:
            self.sum_bY += self.b[self.step_in_cycle] * float(rate_score)
            
            if self.step_in_cycle == self.r - 1:
                self.Z = self.sum_bY * self.bbinv / self.cycle
    
        self.step_in_cycle += 1

        if self.step_in_cycle == self.r + self.cycle:
            self.step_in_cycle = 0
            self.cycle += 1
            #print 'Z:', self.Z
