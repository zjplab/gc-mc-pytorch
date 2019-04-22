import torch 
import pickle as pkl
import numpy as np

def tensor2numpy(filepath:str, savefolder="./data/ml_100k/"):
    rating2=torch.load(filepath)
    arr=rating2.permute(1,2,0).cpu().numpy()
    u,v,R=arr.shape
    output=np.zeros(shape=(u,v))
    for i in range(u):
        for j in range(v):
            for r in range(R):
                if arr[i, j, r]==1:
                    output[i,j]=r+1
    try:
        with open(savefolder+"test_rating_numpy.pkl", "wb") as f:
            pkl.dump(output, f)
    except:
        print("Dump error")
    return 