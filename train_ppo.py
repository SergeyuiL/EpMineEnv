import gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from gym.envs.registration import register
from envs.SingleAgent.mine_toy import EpMineEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, f"models/PPO")
loggir_dir = os.path.join(current_dir, f"logs/PPO")
train_test_dir = os.path.join(current_dir, f"train_test/PPO")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(loggir_dir, exist_ok=True)
os.makedirs(train_test_dir, exist_ok=True)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    register(
        id='EpMineEnv-v0',
        entry_point='envs.SingleAgent.mine_toy:EpMineEnv',
        max_episode_steps=1000,
    )

    env_id = "EpMineEnv-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
    # env = gym.make("EpMineEnv-v0")
    model = PPO("CnnPolicy",
                 env,
                 tensorboard_log=loggir_dir,
                 verbose=1)
    
    TimeSteps = 1e6
    
    for i in range(1,50):
        model.learn(total_timesteps=TimeSteps)
        save_dir = model_dir + "/PPO " + str(i*TimeSteps) + "_steps"
        model.save(save_dir)        
    obs = env.reset()

    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
