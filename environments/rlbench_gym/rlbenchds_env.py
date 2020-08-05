import numpy as np
from gym import spaces
import torch as th
from environments.srl_env import SRLGymEnv
from state_representation.episode_saver import EpisodeSaver
from gym.utils import seeding

def getGlobals():
    """
    :return: (dict)
    """
    return globals()

class RLBenchDSEnv():
    """
    wrapped_env is the rlbench env, passed in initialized. This class just calls actions on that env.
    """

    def __init__(self,name="RLBenchDS-unset", wrapped_env=None, max_steps_per_epoch=100, path=None, **_kwargs):
        #Passed in in dataset_generator.py kwargs.
        self.env = wrapped_env
        self.max_steps_per_epoch = max_steps_per_epoch
        print("Recording data...")
        try:
            env_name = self.env.task.get_name()
        except:
            env_name = self.env.unwrapped.spec.id
        print(env_name)
        self.saver = EpisodeSaver(name, env_name=env_name, path = path)

        self.action_space = self.env.action_space

    def seed(self, seed=None):
        """
        Seed random generator
        :param seed: (int)
        :return: ([int])
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, t=0):
        """
        :action: (int)
        :return: (observation, int reward, bool done, dict extras)
        """
        self.action = action

        self.observation, reward, done, _ = self.env.step(action)
        ground_truth, obs = self.observation['state'], self.observation['observation']
        done = done or t == self.max_steps_per_epoch
        self.saver.step(obs, action, reward, done, ground_truth)
        return self.observation, reward, done, {}   

    @staticmethod
    def getGroundTruthDim():
        """
        :return: (int)
        """
        return None

    def getGroundTruth(self):
        """
        Don't need this method 
        :return: (numpy array)
        """
        return None


    def reset(self):
        """
        Reset the environment
        :return: (numpy ndarray) first observation of the env
        """
        self.observation = self.env.reset()
        ground_truth, obs = self.observation['state'], self.observation['observation']
        if self.saver is not None:
            self.saver.reset(obs, ground_truth)
        return self.observation



    def render(self, mode='rgb_array'):
        """
        :param mode: (str)
        :return: (numpy array) BGR image
        """
        return None
