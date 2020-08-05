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
from rlkit.envs.vec_envs import DummyVecEnv, SubprocVecEnv

import rlbench.gym
import gym
import rlkit.torch.pytorch_util as ptu
# from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpislonStrategy
from rlkit.launchers.launcher_util import setup_logger
# from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import gym.wrappers as wrappers

from rlkit.data_management.obs_dict_replay_buffer import imgObsDictRelabelingBuffer
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import FlattenMlp
# from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
# from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

def experiment(variant):
    expl_env = envs[variant['env']](variant['dr'])
    expl_env = TransformObservationWrapper(expl_env, variant['main_t_fn'])

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = "achieved_goal"
    replay_buffer = imgObsDictRelabelingBuffer(
        env=expl_env,
        rerendering_env = rerendering_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        t_fn=variant['t_fn'],
        **variant['replay_buffer_kwargs']
    )
    # obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    if variant['mlf']:
        if variant['alt'] == 'both':
            conv_args = {
                    "input_width": 16,
                    "input_height": 16,
                    "input_channels": 16,
                    "kernel_sizes": [4],
                    "n_channels": [32],
                    "strides": [4],
                    "paddings": [0],
                    "hidden_sizes": [1024,512],
                    # "added_fc_input_size": action_dim,
                    "batch_norm_conv":False,
                    "batch_norm_fc":False,
                    'init_w':1e-4,
                    "hidden_init":nn.init.orthogonal_,
                    "hidden_activation":nn.ReLU(),
            }
        else:
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
    else:
        if variant['alt'] == 'both':
            conv_args = {
                    "input_width": 64,
                    "input_height": 64,
                    "input_channels": 6,
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
        else:
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
    # eval_path_collector = MdpPathCollector(
    #     eval_env,
    #     policy,
    # )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    eval_path_collector = GoalConditionedPathCollector(
        expl_env,
        exploration_policy.policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
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
    # trainer = HERTrainer(trainer)
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
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000, help="#steps before starting to train ")
    parser.add_argument('--special_name', type=str, default="", help="extra descrpition to run. usually for reruns ")
    parser.add_argument('--mlf', type=int, default=1, help="Whether or not to use mid level features.")
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help="Size of replay buffer.")
    parser.add_argument('--trial', type=int, default=0, help="random seed trial")
    parser.add_argument('--not_special_p', type=float, default=0, help="percent episodes to start from above")
    parser.add_argument('--ground_p', type=float, default=0.5, help="percent episodes to have ground goals")
    parser.add_argument('--special_is_grip', type=int, default=0, help="whether to grip or hover. 0 for hover, 1 for grip. ")
    parser.add_argument('--alt', default=False, help="True to replace with altview. Both for both. False for frontcam. ")
    parser.add_argument('--num_cpus', default=16, help="Number of envs to use.")
    parser.add_argument('--checkpoint', type=int, default=-1, help="to select a checkpointed model. -999 for random, -9 for init taskonomy, -1 for best, +int % 20 for checkpoints.")
    parser.add_argument('--blank', action="store_true", default=False, help="Whether to use envs with pixel observations as all 0s. ")
    parser.add_argument("--readout", action="store_true", default=False, help="Whether to use readouts directly to train. ")    
    args = parser.parse_args()
    

    # Environments
    def dr_flag(dr):
        if dr:
            return '-random'
        return ''
    if not (args.readout or args.mlf):
        img_size = 64

    else:
        img_size = 256
    print("img size", img_size)
    rand_config = VisualRandomizationConfig(
        image_directory="path/to/experiment_textures/train/top10",
            whitelist = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible'],
            apply_floor=True
    )
    
    reach = lambda dr: gym.make('reach_target_easy-vision%s-v0' % dr_flag(dr), sparse=True, rand_config=rand_config, img_size=img_size, force_randomly_place=True, force_change_position=False, altview=args.alt, blank=args.blank)
    push_rotate = lambda dr: gym.make('slide_block_to_target-vision%s-v0' % dr_flag(dr), sparse=True, fixed_grip=0, rand_config=rand_config, img_size=img_size, force_randomly_place=False, force_change_position=False, altview=args.alt, blank=args.blank)
    push_free = None
    pick_and_place = lambda dr: gym.make('pick_and_lift-vision%s-v0' % dr_flag(dr),rand_config=rand_config, sparse=True, force_randomly_place=False, img_size=img_size,force_change_position=False, not_special_p=args.not_special_p, ground_p = args.ground_p, special_is_grip=args.special_is_grip, altview=args.alt, blank=args.blank)
    procedural = lambda dr: gym.make('empty_container-vision%s-v0' % dr_flag(dr),img_size=img_size, sparse=True, not_special_p=args.not_special_p, ground_p = args.ground_p, special_is_grip=args.special_is_grip, force_randomly_place=False, force_change_position=False, procedural_ind=2, altview=args.alt)
    procedural_set = lambda dr: gym.make('empty_container-vision%s-v0' % dr_flag(dr),img_size=img_size, sparse=True, not_special_p=args.not_special_p, ground_p = args.ground_p, special_is_grip=args.special_is_grip, force_randomly_place=False, force_change_position=False, procedural_ind=2, procedural_mode="random",  procedural_set=[0,2,3,4,5,6,8,10,11,12], altview=args.alt)

    
    fetch_push = lambda dr: gym.make('FetchPushImage-v1', img_size=256)
    fetch_reach = lambda dr: gym.make('FetchReachImage-v1', img_size=256)
    
    envs = {
        'reach': reach,
        'push_rotate': push_rotate,
        'push_free': push_free,
        'pick_and_place': pick_and_place,
        'procedural': procedural,
        'procedural_set' : procedural_set,
        'fetch_push': fetch_push,
        'fetch_reach': fetch_reach
    }
    def env_fn():
        return envs[args.env](args.dr)
    num_cpus = int(args.num_cpus)
    rerendering_env = SubprocVecEnv([env_fn for _ in range(num_cpus)])
    import time
    time.sleep(1)

    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=1000000, # 3e6 train steps
            num_eval_steps_per_epoch=0, # 2 rollouts per epoch
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=args.min_num_steps_before_training,
            max_path_length=args.max_path_length,
            batch_size=128,
            random_before_training = True,
            num_epochs_per_eval=0
        ),
        trainer_kwargs=dict(
            policy_learning_rate=args.lr,
            qf_learning_rate=args.lr,
            discount=0.95,
            target_policy_noise=0.2,
            policy_and_target_update_period=2
        ),
        replay_buffer_kwargs=dict(
            max_size=args.replay_buffer_size,
            k=4,
        ),        
        env=args.env,
        dr = args.dr,
        feature_task=args.feature_task,
        noise=args.noise,
        mlf = args.mlf,
        trial = args.trial,
        not_special_p=args.not_special_p,
        ground_p = args.ground_p,
        special_is_grip=args.special_is_grip,
        alt = args.alt,
        checkpoint=args.checkpoint,
        blank = args.blank,
        readout = args.readout
    )


    map_task_to_taskonomy = {
        "normal" : "normal",
        "sobel": "edge_texture",
        "depth": "depth_euclidean",
        "segment_semantic": "segment_semantic",
        "segment_img": "segment_semantic",
        "sobel_3d": "edge_occlusion",
        "autoencoder": "autoencoding",
        "denoise": "denoising"
    }
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.mlf or args.readout:
        taskonomy_feature = map_task_to_taskonomy[args.feature_task]
        VisualPriorPredictedLabel._load_unloaded_nets([taskonomy_feature])
        VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature] = VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature].to(default_device)
        net = VisualPriorPredictedLabel.feature_task_to_net[taskonomy_feature] 

        if args.feature_task == "segment_semantic":
            #set in RLBench
            num_max = 30
            net.decoder.decoder_output[0] = torch.nn.Conv2d(16, num_max, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.decoder.decoder_output[1] = torch.nn.Identity()
            net.cuda()
        if args.feature_task == "segment_img":
            num_max = 5
            net.decoder.decoder_output[0] = torch.nn.Conv2d(16, num_max, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.decoder.decoder_output[1] = torch.nn.Identity()
            net.cuda()

        net.to(default_device)
        if args.checkpoint == -9:
            pass
        elif args.checkpoint == -999:
            def weight_reset(m):
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
            net.apply(weight_reset)
            print("Reset parameters.")
        elif args.checkpoint == -1:
            net.load_state_dict(torch.load(args.models_path + "/" + args.feature_task + "/bestModel.pth"))
        else:
            net.load_state_dict(torch.load(args.models_path + "/" + args.feature_task + "/epoch_{}.pth".format(args.checkpoint)))
        if args.readout:
            net.eval()
        else:
            net.encoder.eval()
            net.encoder.normalize_outputs=True
            net.eval()

    # single view cases
    if not args.alt=="both":
        def t_fn(obs):
            # Expects obs to come between 0 and 1
            INPUT_SHAPE = (3,256,256)
            obs = obs * 2 - 1
            obs = torch.tensor(obs, device=default_device).float().reshape([1, *INPUT_SHAPE])
            with torch.no_grad():
                obs = net.encoder(obs)
                return np.array(obs.cpu()).flatten()
            return obs
        def main_t_fn(obs):
            o = obs['observation']
            #Expects obs to come between 0 and 1
            INPUT_SHAPE = (3,256,256)
            o = o * 2 - 1
            o = torch.tensor(o, device=default_device).float().reshape([1, *INPUT_SHAPE])
            with torch.no_grad():
                o = net.encoder(o)
            obs['observation'] = np.array(o.cpu()).flatten()
            return obs
    # alt view is both
    else:
        def t_fn(obs):
            # Expects obs to come between 0 and 1
            INPUT_SHAPE = (3,256,256)
            obs = obs * 2 - 1
            front, alt = obs
            front = torch.tensor(front, device=default_device).float().reshape([1, *INPUT_SHAPE])
            alt = torch.tensor(alt, device=default_device).float().reshape([1, *INPUT_SHAPE])
            with torch.no_grad():
                front = net.encoder(front).cpu()
                alt = net.encoder(alt).cpu()
                return np.concatenate([front, alt], axis=1).flatten()

        def main_t_fn(obs):
            o = obs['observation']
            #Expects obs to come between 0 and 1
            INPUT_SHAPE = (3,256,256)
            o = o * 2 - 1
            front, alt = o
            front = torch.tensor(front, device=default_device).float().reshape([1, *INPUT_SHAPE])
            alt = torch.tensor(alt, device=default_device).float().reshape([1, *INPUT_SHAPE])
            with torch.no_grad():
                front = net.encoder(front).cpu()
                alt = net.encoder(alt).cpu()
            obs['observation'] = np.concatenate([front, alt], axis=1).flatten()
            return obs
    if args.readout:
        def t_fn(obs):
            INPUT_SHAPE = (3,256,256)
            obs = obs*2 - 1
            default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            obs = torch.tensor(obs, device=default_device).float().reshape([1, *INPUT_SHAPE])
            obs = net(obs)
            input_size = 256
            output_size = 64
            bin_size = input_size // output_size
            small_image = np.array(obs.detach().cpu()).reshape((3, output_size, bin_size, 
                                                output_size, bin_size)).max(4).max(2).flatten()
            # small_image = small_image.astype(np.uint8)
            return small_image

        def main_t_fn(o):
            obs = o['observation']
            INPUT_SHAPE = (3,256,256)
            obs = obs*2 - 1
            default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            obs = torch.tensor(obs, device=default_device).float().reshape([1, *INPUT_SHAPE])
            obs = net(obs)
            input_size = 256
            output_size = 64
            bin_size = input_size // output_size
            small_image = np.array(obs.detach().cpu()).reshape((3, output_size, bin_size, 
                                                output_size, bin_size)).max(4).max(2).flatten()
            # small_image = small_image.astype(np.uint8)
            o['observation'] = small_image 
            return o

    if args.mlf or args.readout:
        variant["t_fn"] = t_fn
        variant["main_t_fn"] = main_t_fn
    else:
        variant['t_fn'] = lambda x : x * 2 - 1
        def main_t_fn(obs):
            obs['observation'] = obs['observation'] * 2 - 1
            return obs
        variant['main_t_fn'] = main_t_fn
    ptu.set_gpu_mode(True)

    if args.mlf:
        mlf_label = "mlf"
    else:
        mlf_label = "pixels"
    setup_logger('trial{}-{}-HER-{}{}{}-{}-{}-{}'.format(args.trial, args.special_name, mlf_label, args.env,dr_flag(args.dr), args.feature_task, args.noise, args.lr), variant=variant, snapshot_mode='gap_and_last', snapshot_gap=100)
    experiment(variant)
