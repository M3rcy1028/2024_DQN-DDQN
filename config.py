import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--model', type=str, default='reinforce_cartpole/model.pt')
parser.add_argument('--runtype', type=str, default='train_run', choices=('train', 'run', 'train_run'))

parser.add_argument('--algorithm', type=str, default='dqn13', choices=('dqn13', 'dqn15', 'ddqn'))
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--maxreward', type=int, default=500)
parser.add_argument('--update', type=int, default=40)
args = parser.parse_args()