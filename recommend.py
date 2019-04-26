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
        print("Top {} Movie Name: {}\t Your estimated score:{}\n".format(n, name, mhat[user, indices[-(n+1)]]))

    return

def recommend_metric(dataset:str):
    mhat=torch.load("./"+dataset+"_mhat.pt").cpu().numpy()
    with open("./data/"+dataset+"/train_numpy.pkl", "rb") as f:
        train=pkl.load(f)
    with open("./data/"+dataset+"/valid_numpy.pkl", "rb") as f:
        valid=pkl.load(f)
    with open("./data/"+dataset+"/test_numpy.pkl", "rb") as f:
        test=pkl.load(f)
    train=train+valid
    
    num_user, num_iter=train.shape
    precision=.0
    recall=.0
    f_measure=.0
    ndcg=.0
    map_score=.0
    count=0
    for i in range(num_user):
        if len(np.nonzero(test[i,:])[0])>10:
            count+=1
            test10=np.argsort(test[i, :])[:-11:-1]
            rec10=np.argsort(mhat[i, :])[:-11:-1]
            #precision
            tmp_precision=len(np.intersect1d(test10, rec10))/len(rec10)
            precision+=tmp_precision

            #recall
            tmp_recall=len(np.intersect1d(test10, rec10))/len(test10)
            recall+=tmp_recall

            #f1 score
            try:
                f_measure+=2*(tmp_precision*tmp_recall)/(tmp_precision + tmp_recall)
            else:
                f_measure+=1

            #discounted cumulative gain
            tmp_dcg=.0
            dcg_count=0
            for index, item in enumerate(rec10):
                if item in test10:
                    dcg_count+=1
                    tmp_dcg+=1/np.log2(index+1)
            tmp_idcg=np.sum([1/np.log2(i+1) for i in range(dcg_count)])
            ndcg+=tmp_dcg/tmp_idcg

            #mean avg precision
            tmp_map_score=.0
            map_score_count=0
            for index, item in enumerate(rec10):
                if item in test10:
                    map_score_count+=1
                    tmp_map_score+=map_score_count/(index+1)
            tmp_map_score/=map_score_count
            map_score+=tmp_map_score

    #avg over
    precision/=count
    recall/=count
    f_measure/=count
    ndcg/=count
    map_score/=count

    print("precision :{}\n \
        recall: {}\n \
            f1 score: {}\n \
                Normalized Discounted Cumulative Gain: {}\n \
                    Mean Average Precision :{}".format(precision, recall,\
                        f_measure, ndcg, map_score))
    return 