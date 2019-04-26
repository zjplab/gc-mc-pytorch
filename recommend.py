import torch
import numpy as np
import pickle as pkl
import pandas as pd
import itertools

def recommend(dataset:str, user:int):
    mhat=torch.load("./"+dataset+"_mhat.pt").cpu().numpy()
    with open("./data/"+dataset+"/train_numpy.pkl", "rb") as f:
        train=pkl.load(f)
    with open("./data/"+dataset+"/valid_numpy.pkl", "rb") as f:
        valid=pkl.load(f)
    with open("./data/"+dataset+"/test_numpy.pkl", "rb") as f:
        test=pkl.load(f)
    train=train+valid
    
    if dataset=="ml_100k":
        sep = r'|'
        movie_file = "./data/ml_100k/u.item"
        movie_headers = ['movie id', 'title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python', usecols=["movie id", "title"]
                               )
    elif dataset=="ml_1m":
        sep = r'\:\:'
        movies_file = "./data/ml_1m/movies.dat"
        movies_headers = ['movie_id', 'title', 'genre']
        movie_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python', usecols=["movie_id", "title"])
                                
    non_zero=np.nonzero(train[user,:])
    indices=np.argsort(mhat[user,:])
    indices=np.setdiff1d(indices, non_zero, assume_unique=True)
    names=movie_df.title[ indices[:-11:-1]]
    for n,name in enumerate(names):
        print("Top {} Movie Name: {}\n".format(n, name))

    return