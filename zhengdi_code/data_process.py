import json
import collections
import numpy as np
from sklearn import datasets, linear_model

use_intercept = False

def restaurant_process(n_top):
    data = []
    with open('yelp_restaurant_objects.json') as f:
        for line in f:
            data.append(json.loads(line))

# create list of categories:
    category = []
    category2 = []
    for restaurant in data:
        category2.append(restaurant['categories'])
        category2[-1].remove('Restaurants')
        category.extend(category2[-1])

# find the top n_top categories:
    counter = collections.Counter(category)
    top = counter.most_common(n_top)
    
    categorylist = []
    for c in top:
        categorylist.append(c[0])

    featurelist = []
    restaurant_feature = {}
    
    for restaurant in data:
        restaurant_id = str(restaurant['business_id'])
        
        # generate the binary feature vector for this restaurant:
        feature = [0 for x in range(n_top)]
        for c in restaurant['categories']:
            if c in categorylist:
                i = categorylist.index(c)
                feature[i] = 1
    
        if not feature in featurelist:
            featurelist.append(feature)

        restaurant_feature[restaurant_id] = feature
    
    np.savetxt('feature.txt', featurelist, delimiter = ' ', fmt = '%s', newline = '\r\n')
    return featurelist, restaurant_feature


def read_features():
    f = open('feature.txt', 'r')
    featurelist = [[float(x) for x in line.split()] for line in f]
    return featurelist


def review_process(n_review):
    # collect users that give >= n_review reviews:
    data = json.load(open('restaurant_review.json'))
    
    user_id = []
    for review in data:
        user_id.append(review['user_id'])
    
    counter = collections.Counter(user_id)
    new_data = {}
    for user in counter:
        if counter[user] > n_review:
            new_data[str(user)] = []

    for review in data:
        user = str(review['user_id'])
        if user in new_data:
            new_data[user].append(review)

    new_user_id = new_data.keys()
    
    return new_data, new_user_id


def ridge(valpha, userlist, reviews, featurelist, restaurant_feature, n_categories):
    coef = []
    
    i = 0
    
    for user in userlist:
        rate = []
        feature_vector = []
        
        for review in reviews[str(user)]:
            rate.append(int(review['stars']))
            restaurant_id = str(review['business_id'])
            feature = restaurant_feature[restaurant_id]
            feature_vector.append(feature)
        
        rate = np.array(rate)
        feature_vector = np.array(feature_vector)
        clf = linear_model.Ridge(fit_intercept = use_intercept)
        clf.set_params(alpha=valpha)
        clf.fit(feature_vector, rate)
        beta_hat = clf.coef_
        """
        if use_intercept:
            beta_hat = np.append(beta_hat, [clf.intercept_])
        """
        coef.append(beta_hat)
        
    
    coef = np.array(coef)
    mean = np.mean(coef, axis = 0)
    cov = np.cov(coef.T)

    np.savetxt('mean.txt', mean, delimiter = ' ')
    np.savetxt('cov.txt', cov, delimiter = ' ', fmt = '%s', newline = '\r\n')
    return mean, cov


def import_2D_data(filename):
    f = open(filename, 'r')
    data = [[float(x) for x in line.split()] for line in f]
    return data


def import_1D_data(filename):
    f = open(filename, 'r')
    data = []
    for x in f.readlines():
        data.append(float(x))
    return data


def process(option, n_categories = 20, n_reviews = 20):
    if option == 1: # compute new data
        
        featurelist, restaurant_feature = restaurant_process(n_categories)
        np.savetxt('feature.txt', featurelist, delimiter = ' ', fmt = '%s', newline = '\r\n')
    
        reviews, user = review_process(n_reviews)
    
        alpha = 0.1
        mean, cov = ridge(alpha, user, reviews, featurelist, restaurant_feature, n_categories)
        np.savetxt('mean.txt', mean, delimiter = ' ')
        np.savetxt('cov.txt', cov, delimiter = ' ', fmt = '%s', newline = '\r\n')

    
    else: # import from files
        featurelist = import_2D_data('feature.txt')
        mean = import_1D_data('mean.txt')
        cov = import_2D_data('cov.txt')

    return featurelist, mean, cov

