import numpy as np
import data_process as data
import user
#import EGA_server as server
#import UCB_server as server
import PEGE_server as server


def run(n_replications, n_steps, mean, cov, sigma, featurelist):
    total_reward = 0
    var = 0
    total_regret = 0
    for sample in range(n_replications):
        User = user.user(mean, cov, sigma)
        User.generate_user()
        Server = server.server(mean, cov, sigma, featurelist, n_steps)
        
        maxrate = User.maxrate(featurelist)
        reward = 0
        for i in range(n_steps):
            restaurant = Server.recommand()
            rate = User.rate(restaurant)
            reward += rate
            Server.update(restaurant, rate)

        reward = reward * 1.0 / n_steps
        #print 'rating', reward * n_steps

        total_reward += reward
        var += reward * reward
        total_regret += maxrate - reward

    mean_reward = total_reward * 1.0 / n_replications
    var = var * 1.0 / n_replications - mean_reward * mean_reward
    CI = [mean_reward - 1.96 * np.sqrt(var/n_replications), mean_reward + 1.96 * np.sqrt(var/n_replications)]
    mean_regret = total_regret * 1.0 / n_replications
    return mean_regret, mean_reward, CI




if __name__ == '__main__':
    
    option = 2 # 1: generate new; 2: read from files
    n_categories = 20
    n_reviews = 20
    featurelist, mean, cov = data.process(option, n_categories, n_reviews)
    print 'data processed'
    sigma = 0.8
    n_steps = 2275
    
    np.random.seed(1)
    
    #U = user.user(mean, cov, sigma)
    #S = server.server(mean, cov, sigma, featurelist, n_steps)
    
    #restaurant = S.recommand()
    #print restaurant
    print server.ALG()
    
    
    n_replications = 100
    mean_regret, mean_reward, CI = run(n_replications, n_steps, mean, cov, sigma, featurelist)
    print 'regret', mean_regret
    print 'reward', mean_reward * n_steps
    print 'reward CI', CI[0]*n_steps, CI[1]*n_steps
