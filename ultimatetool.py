import torch 
import pickle as pkl
import numpy as np

def tensor2dump(filename:str, dataset:str, savename:str):
    rating2=torch.load("./data/"+dataset+"/"+filename)
    omg = torch.sum(rating2, 0).detach()
    y = torch.max(rating2, 0)[1].float() + 1.
    y= torch.mul(omg, y)
    y=y.cpu().numpy()
    try:
        with open("./data/"+dataset+"/"+savename, "wb") as f:
            pkl.dump(y, f)
    except:
        print("Dump error")
    return 