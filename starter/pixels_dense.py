import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpislonStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.conv_networks import CNN, FlattenCNN, TanhCNNPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from torch import nn as nn
import gym
import visualpriors
import torch.utils.model_zoo
import sys
from rlkit.envs.wrappers import TransformObservationWrapper
import numpy as np
import rlbench.gym
import gym.wrappers as wrappers
from rlbench import VisualRandomizationConfig



def experiment(variant):
    img_size = 64
    train_top10 = VisualRandomizationConfig(
        image_directory='./experiment_textures/train/top10',
            whitelist = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible'],
            apply_arm = False,
            apply_gripper = False,
            apply_floor = True
    )
    expl_env = gym.make('reach_target_easy-vision-v0', sparse=False, img_size=img_size, force_randomly_place=True, force_change_position=False, blank=True)
    expl_env = wrappers.FlattenDictWrapper(
        expl_env, dict_keys=['observation']
    )
    t_fn = variant["t_fn"]
    expl_env = TransformObservationWrapper(expl_env, t_fn)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    conv_args = {
            "input_width": 64,
            "input_height": 64,
            "input_channels": 3,
            "kernel_sizes": [4,4,3],
            "n_channels": [32,64,64],
            "strides": [2,1,1],
            "paddings": [0,0,0],
            "hidden_sizes": [1024,512],
            "batch_norm_conv":False,
            "batch_norm_fc":False,
            'init_w':1e-4,
            "hidden_init":nn.init.orthogonal_,
            "hidden_activation":nn.ReLU(),
    }

    qf1 = FlattenCNN(
        output_size=1,
        added_fc_input_size=action_dim,
        **variant['qf_kwargs'],
        **conv_args
    )
    qf2 = FlattenCNN(
        output_size=1,
        added_fc_input_size=action_dim,
        **variant['qf_kwargs'],
        **conv_args
    )
    target_qf1 = FlattenCNN(
        output_size=1,
        added_fc_input_size=action_dim,
        **variant['qf_kwargs'],
        **conv_args
    )
    target_qf2 = FlattenCNN(
        output_size=1,
        added_fc_input_size=action_dim,
        **variant['qf_kwargs'],
        **conv_args
    )
    policy = TanhCNNPolicy(
        output_size=action_dim,
        **variant['policy_kwargs'],
        **conv_args
    )
    target_policy = TanhCNNPolicy(
        output_size=action_dim,
        **variant['policy_kwargs'],
        **conv_args
    )
    # es = GaussianStrategy(
    #     action_space=expl_env.action_space,
    #     max_sigma=0.3,
    #     min_sigma=0.1,  # Constant sigma
    # )
        
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space, 
        epsilon=0.3,
        max_sigma=0.0, min_sigma=0.0, #constant sigma 0
        decay_period=1000000
    )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=None,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=None,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def set_seed(seed):
    """
    Set the seed for all the possible random number generators.
    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)



if __name__ == "__main__":
    lr = 1e-5
    INPUT_SHAPE = (3,256,256)
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=1000000,
            num_eval_steps_per_epoch=0, 
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=50,
            batch_size=128,
            random_before_training=True
        ),
        trainer_kwargs=dict(
            policy_learning_rate=lr,
            qf_learning_rate=lr,
            discount=0.95,
            target_policy_noise=0.2,
            policy_and_target_update_period=2
        ),
        replay_buffer_size=int(1e6),
    )
    def t_fn(obs):
        obs = obs * 2 - 1
        return obs
    variant["t_fn"] = t_fn
    ptu.set_gpu_mode(True)
    special_name = ""
    setup_logger('scratch-%s-%s' % (special_name, lr), variant=variant, snapshot_mode='gap_and_last', snapshot_gap=100)
    experiment(variant)
