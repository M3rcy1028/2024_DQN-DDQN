from config import args
from environment import init_env
from model import NN
from runner import Runner
import torch
import time
import gym
import os
# set the cuda visible devices
# cuda:0 uses GPU if GPU is available else uses CPU 
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.device)
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[Device]\tDevice selected: ", device)

    env = init_env()
    nn = NN(env.observation_space.shape[0], env.action_space.n).to(device)
    
    # Load model if required
    if args.load: 
        nn.load_state_dict(torch.load(args.model))

    loss = torch.nn.MSELoss()
    runner = Runner(nn, loss, device, env, args.algorithm, lr=args.lr, gamma=args.gamma, target_update=args.update, maxreward=args.maxreward, logs=f"dqn_cartpole/{time.time()}")
    
    if "train" in args.runtype:
        print("[Train]\tTraining Beginning ...")
        runner.train(args.episodes)

        if args.plot:
            print("[Plot]\tPlotting Training Curves ...")
            runner.plot()

    if args.save: 
        print("[Save]\tSaving Model ...")
        runner.save()

    if "run" in args.runtype:
        print("[Run]\tRunning Simulation ...")
        runner.run()

    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()
