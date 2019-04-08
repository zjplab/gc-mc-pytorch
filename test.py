import argparse
import data_loader
#from data_loader import get_loader

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



print(args.num_epochs)