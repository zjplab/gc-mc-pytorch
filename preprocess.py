import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import torch
# For automatic dataset downloading
from urllib2 import urlopen
from zipfile import ZipFile
from StringIO import StringIO
import os.path
from os import rename, system

def download_dataset(dataset, data_dir):
    """ Downloads dataset if files are not present. 
    -----
    dataset: ml_100k | ml_1m | ml_10m(not recommended)
    files
    """
    dict={"ml_100k":['/u.data', '/u.item', '/u.user'], \
        'ml_1m':['/ratings.dat', '/movies.dat', '/users.dat']}
    files=dict[dataset]
    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)
        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'data/' + dataset
        elif dataset == 'ml_10m':
            target_dir = 'data/' + 'ml_10m'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(StringIO(request.read())) as zip_ref:
            zip_ref.extractall('data/')
        rename("./data/"+dataset.replace('_', '-'), target_dir)


def preprocess(dataset):
    if dataset=="ml_1m":
        ratings = pd.read_csv('./data/ml_1m/ratings.dat', sep = '::', header = None,\
             engine = 'python', encoding = 'latin-1')
        total_length = len(ratings)
        ratings = ratings.sample(frac=1)

        len_train = int(total_length*0.85)
        len_val   = int(total_length*0.9)

        rating_train = ratings[:len_train]
        rating_val   = ratings[len_train:len_val]
        rating_test  = ratings[len_val:]

        num_users  = 6040
        num_items = 3953
        rating_cnt = 5

        for i, ratings in enumerate([rating_train, rating_val, rating_test]):
            rating_mtx = torch.zeros(rating_cnt, num_users, num_items)
    
            for index, row in ratings.iterrows():
                u = row[0]-1
                v = row[1]-1
                r = row[2]-1
                
                rating_mtx[r, u, v] = 1
            torch.save(rating_mtx, './data/rating_%d.pkl'%i)

        users_headers = ['user id', 'gender', 'age', 'occupation', 'zip code']
        users_df = pd.read_csv('./data/ml_1m/users.dat', sep = '::', header = None, \
            names = users_headers, engine = 'python', encoding = 'latin-1')
        movie_headers = ['movie id', 'movie title', 'genre']
        movie_df = pd.read_csv('./data/ml_1m/movies.dat', sep = '::', header = None, names = movie_headers, \
             engine = 'python', encoding = 'latin-1')

        occupation = set(users_df['occupation'].values.tolist())
        age_dict = {1:0., 18:1., 25:2., 35:3., 45:4., 50:5., 56:6.}
        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']-1
            # age
            u_features[u_id, 0] = age_dict[row['age']]
            # gender
            u_features[u_id, 1] = gender_dict[row['gender']]
            # occupation
            u_features[u_id, occupation_dict[row['occupation']]] = 1.
        torch.save(torch.from_numpy(u_features), './data/ml_1m/u_features.pkl')

        
        genre_dict = {'Action':0, 'Adventure':1, 'Animation':2, "Children's":3, 'Comedy':4,
                    'Crime':5, 'Documentary':6, 'Drama':7, 'Fantasy':8, 'Film-Noir':9, 'Horror':10,
                    'Musical':11, 'Mystery':12, 'Romance':13, 'Sci-Fi':14, 'Thriller':15,
                    'War':16, 'Western':17}
        num_genres = len(genre_dict)

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df['genre'].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            for j in [genre_dict[g] for g in g_vec.split('|')]:
                v_features[movie_id-1][j] = 1

        torch.save(torch.from_numpy(v_features), './data/ml_1m/v_features.pkl')

    elif dataset=="ml_100k":
        
        train = pd.read_csv('./data/ml_100k/u1.base', sep = '\t', header = None, engine = 'python', \
            encoding = 'latin-1')
        test  = pd.read_csv('./data/ml_100k/u1.test', sep = '\t', header = None, engine = 'python', \
            encoding = 'latin-1')


        train_length = len(train)
        train = train.sample(frac=1)

        len_train = int(train_length*0.9)

        rating_train = train[:len_train]
        rating_val   = train[len_train:]
        rating_test  = test

        num_users = 943
        num_items = 1682
        rating_cnt= 5


        for i, ratings in enumerate([rating_train, rating_val, rating_test]):
            rating_mtx = torch.zeros(rating_cnt, num_users, num_items)
            
            for index, row in ratings.iterrows():
                u = row[0]-1
                v = row[1]-1
                r = row[2]-1
                
                rating_mtx[r, u, v] = 1
            torch.save(rating_mtx, './data/ml_100k/rating_%d.pkl'%i)



        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv('./data/ml_100k/u.user', sep = '|', header = None, \
            names = users_headers, engine = 'python', encoding = 'latin-1')
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                        'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
        movie_df = pd.read_csv('./data/ml_100k/u.item', sep = '|', header = None, \
            names = movie_headers, engine = 'python', encoding = 'latin-1')



        occupation = set(users_df['occupation'].values.tolist())
        age = users_df['age'].values
        age_max = age.max()
        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']-1
            # age
            u_features[u_id, 0] = row['age'] / np.float(age_max)
            # gender
            u_features[u_id, 1] = gender_dict[row['gender']]
            # occupation
            u_features[u_id, occupation_dict[row['occupation']]] = 1.
        torch.save(torch.from_numpy(u_features), './data/ml_100k/u_features.pkl')



        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            v_features[movie_id-1] = g_vec
        torch.save(torch.from_numpy(v_features), './data/ml_100k/v_features.pkl')


