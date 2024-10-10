import gym

def init_env():
    # Initialize the CartPole environment
    env = gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
    return env