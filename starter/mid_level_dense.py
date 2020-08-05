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
from visualpriors.transforms import VisualPriorPredictedLabel
import argparse


def experiment(variant):
    expl_env = envs[variant['env']](variant['dr'])
    expl_env = wrappers.FlattenDictWrapper(
        expl_env, dict_keys=['observation']
    )
    t_fn = variant["t_fn"]
    expl_env = TransformObservationWrapper(expl_env, t_fn)
    action_dim = expl_env.action_space.low.size
    conv_args = {
            "input_width": 16,
            "input_height": 16,
            "input_channels": 8,
            "kernel_sizes": [4],
            "n_channels": [32],
            "strides": [4],
            "paddings": [0],
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
    if variant['noise'] == "eps":
        es = GaussianAndEpislonStrategy(
            action_space=expl_env.action_space, 
            epsilon=0.3,
            max_sigma=0.0, min_sigma=0.0, #constant sigma 0
            decay_period=1000000
        )
    elif variant['noise'] == "gaussian":
        es = GaussianStrategy(
            action_space=expl_env.action_space,
            max_sigma=0.3,
            min_sigma=0.1,
            decay_period = 1000000  
        )
    else:
        print("unsupported param for --noise")
        assert False
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
    parser = argparse.ArgumentParser(description='Train fine tuned features')

    parser.add_argument('--feature_task', type=str,required=True, help='Feature task to test. One of sobel, normal, segment_semantic, or depth')
    parser.add_argument('--models_path', type=str,required=True, help='path to root of model. Expects path/normal to contain normal logs.')
    parser.add_argument('--env', type=str,required=True, help='Env to run on. one of the keys in envs.')
    parser.add_argument('--lr', type=float, default=1e-4,help='policy/q learning rate')
    parser.add_argument('--dr', action='store_true', default=False, help="Include this flag to use the chosen environment with domain randomization")
    parser.add_argument('--noise', type=str, default="eps", help="eps for 0.3 eps greedy, gaussian for gaussian noise ")
    parser.add_argument('--max_path_length', type=int, required=True, help="#steps / rollout ")
    parser.add_argument('--special_name', type=str, required=True, help="name of exp")
    parser.add_argument('--not_special_p', type=float, default=0, help="percent episodes to start from above")
    parser.add_argument('--ground_p', type=float, default=0.5, help="percent episodes to have ground goals")
    parser.add_argument('--special_is_grip', type=int, default=0, help="whether to grip or hover. 0 for hover, 1 for grip. ")
    args = parser.parse_args()

    # Environments
    def dr_flag(dr):
        if dr:
            return '-random'
        return ''
    img_size = 256
    rand_config = VisualRandomizationConfig(
        image_directory='path/to/experiment_textures/train/top10',
            whitelist = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible'],
            apply_floor=True
    )
    reach = lambda dr: gym.make('reach_target_easy-vision%s-v0' % dr_flag(dr), rand_config=rand_config, img_size=img_size, force_randomly_place=True, force_change_position=False)
    push_rotate = lambda dr: gym.make('slide_block_to_target-vision%s-v0' % dr_flag(dr), fixed_grip=0, rand_config=rand_config, img_size=img_size, force_randomly_place=False, force_change_position=False)
    pick_and_place = lambda dr: gym.make('pick_and_lift-vision%s-v0' % dr_flag(dr), rand_config=rand_config, force_randomly_place=False, force_change_position=False, img_size=img_size,  not_special_p=args.not_special_p, ground_p = args.ground_p, special_is_grip=args.special_is_grip)

    envs = {
        'reach': reach,
        'push_rotate': push_rotate,
        'pick_and_place': pick_and_place,
    }

    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=5000000, 
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=args.max_path_length,
            batch_size=128,
            random_before_training = True
        ),
        trainer_kwargs=dict(
            policy_learning_rate=args.lr,
            qf_learning_rate=args.lr,
            discount=0.95,
            target_policy_noise=0.2,
            policy_and_target_update_period=2
        ),
        replay_buffer_size=int(1e6),
        env=args.env,
        dr = args.dr,
        feature_task=args.feature_task,
        noise=args.noise
    )


    map_task_to_taskonomy = {
        "normal" : "normal",
        "sobel": "edge_texture",
        "depth": "depth_euclidean",
        "segment_img": "segment_semantic",
        "sobel_3d": "edge_occlusion",
        "autoencoder": "autoencoding",
        "denoising": "denoising"
    }
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    taskonomy_feature = map_task_to_taskonomy[args.feature_task]
    VisualPriorPredictedLabel._load_unloaded_nets([taskonomy_feature])
    VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature] = VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature].to(default_device)
    net = VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature] 

    if args.feature_task == "segment_img":
        num_max = 5
        net.decoder.decoder_output[0] = torch.nn.Conv2d(16, num_max, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        net.decoder.decoder_output[1] = torch.nn.Identity()
        net.cuda()

    net.to(default_device)
    net.encoder.eval()
    net.encoder.normalize_outputs=True
    net.load_state_dict(torch.load(args.models_path + "/" + args.feature_task + "/bestModel.pth"))
    net.eval()

    def t_fn(obs):
        #Expects obs to come between 0 and 1
        INPUT_SHAPE = (3,256,256)
        obs = obs * 2 - 1
        obs = torch.tensor(obs, device=default_device).float().reshape([1, *INPUT_SHAPE])
        with torch.no_grad():
            obs = net.encoder(obs)
        return np.array(obs.cpu()).flatten()

    variant["t_fn"] = t_fn
    ptu.set_gpu_mode(True)
    setup_logger('{}-Dense_MLF-{}{}-{}-{}-{}'.format(args.special_name, args.env,dr_flag(args.dr), args.feature_task, args.noise, args.lr), variant=variant, snapshot_mode='gap_and_last', snapshot_gap=100)
    experiment(variant)
