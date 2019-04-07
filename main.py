# Importing the libraries
import os
import numpy as np
import pandas as pd
from random import sample
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import BatchSampler, SequentialSampler
import pickle
from model import *
import argparse
from data_loader import get_loader


###### Arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument('--mode', type=str, default="train", help='train / test')
parser.add_argument('--data_type', type=str, default="ml_100k")
parser.add_argument('--model-path', type=str, default="./models")
parser.add_argument('--data-path', type=str, default="./data")
parser.add_argument('--data-shuffle', type=bool, default=True)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--val-step', type=int, default=5)
parser.add_argument('--test-epoch', type=int, default=50)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--neg-cnt', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')

parser.add_argument('--emb-dim', type=int, default=32)
parser.add_argument('--hidden', default=[64,32,16, 8])
parser.add_argument('--nb', type=int, default=2)

parser.add_argument('--train_path', '-train',type=str, default='/rating_0.pkl')#train.pkl')
parser.add_argument('--val_path','-val', type=str, default='/rating_1.pkl')#val.pkl')
parser.add_argument('--test_path', '-test',type=str, default='/rating_2.pkl')#test.pkl')

args = parser.parse_args()

############
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the datas
num_users, num_items, num_classes, num_side_features, num_features,\
u_features, v_features, u_features_side, v_features_side = get_loader(args.data_type)

u_features = torch.from_numpy(u_features).to(device).float()
v_features = torch.from_numpy(v_features).to(device).float()
u_features_side = torch.from_numpy(u_features_side).to(device)
v_features_side = torch.from_numpy(v_features_side).to(device)
rating_train = torch.load('./data/'+args.data_type+args.train_path).to(device)
rating_val = torch.load('./data/'+args.data_type+args.val_path).to(device)
rating_test = torch.load('./data/'+args.data_type+args.test_path).to(device)

# Creating the architecture of the Neural Network
model = GAE(num_users, num_items, num_classes,
            num_side_features, args.nb,
            u_features, v_features, u_features_side, v_features_side,
            num_users+num_items, args.emb_dim, args.hidden, args.dropout)
if torch.cuda.is_available():
    model.cuda()
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=[args.beta1, args.beta2])

best_epoch = 0
best_loss  = 9999.

def reset_grad():
    """Reset the gradient buffers."""
    optimizer.zero_grad()

def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()

        train_loss = 0.
        train_rmse = 0.
        for s, u in enumerate(BatchSampler(SequentialSampler(sample(range(num_users), num_users)),
                              batch_size=num_users, drop_last=False)):
                              #batch_size=args.batch_size, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)

            for t, v in enumerate(BatchSampler(SequentialSampler(sample(range(num_items), num_items)),
                                  batch_size=num_items, drop_last=False)):
                                  #batch_size=args.batch_size, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)
                if len(torch.nonzero(torch.index_select(torch.index_select(rating_train, 1, u), 2, v))) == 0:
                    continue

                m_hat, loss_ce, loss_rmse = model(u, v, rating_train)

                reset_grad()
                loss_ce.backward()
                optimizer.step()

                train_loss += loss_ce.item()
                train_rmse += loss_rmse.item()

        log = 'epoch: '+str(epoch+1)+' loss_ce: '  +str(train_loss/(s+1)/(t+1)) \
                                    +' loss_rmse: '+str(train_rmse/(s+1)/(t+1))
        print(log)

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                u = torch.from_numpy(np.array(range(num_users))).to(device)
                v = torch.from_numpy(np.array(range(num_items))).to(device)
                m_hat, loss_ce, loss_rmse = model(u, v, rating_val)

            print('[val loss] : '+str(loss_ce.item())+
                  ' [val rmse] : '+str(loss_rmse.item()))
            if best_loss > loss_rmse.item():
                best_loss = loss_rmse.item()
                best_epoch= epoch+1
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path,
                          'model-%d.pkl'%(best_epoch))))
    model.eval()
    with torch.no_grad():
        u = torch.from_numpy(np.array(range(num_users))).to(device)
        v = torch.from_numpy(np.array(range(num_items))).to(device)
        m_hat, loss_ce, loss_rmse = model(u, v, rating_test)

    print('[test loss] : '+str(loss_ce.item()) +
          ' [test rmse] : '+str(loss_rmse.item()))

def predict():
    model.load_state_dict(torch.load(os.path.join(args.model_path,
                          'model-%d.pkl'%(best_epoch))))
    model.eval()
    with torch.no_grad():
        u = torch.from_numpy(np.array(range(num_users))).to(device)
        v = torch.from_numpy(np.array(range(num_items))).to(device)
        _,_,_, mhat = model.predict(u, v, rating_train)
    return mhat

if __name__ == '__main__':
    dataset=args.data_type
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
    mhat=predict()
    torch.save(mhat, 'mhat.pt')