import rlbench.gym
import gym
import rlkit.torch.pytorch_util as ptu
# from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpislonStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import gym.wrappers as wrappers

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer, imgObsDictRelabelingBuffer
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import FlattenMlp

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
import argparse

ptu.set_gpu_mode(True)

def experiment(variant):
    expl_env = gym.make('pick_and_lift-state-v0',sparse=True, not_special_p=0.5, ground_p = 0, special_is_grip=True, img_size=256, force_randomly_place=False, force_change_position=False)

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'

    achieved_goal_key = "achieved_goal"
    replay_buffer = ObsDictRelabelingBuffer(
        env=expl_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    action_dim = expl_env.action_space.low.size
    goal_dim = expl_env.observation_space.spaces['desired_goal'].low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.3,
        min_sigma=0.1,
        decay_period=1000000  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    trainer = TD3Trainer(
        # env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    
    trainer = HERTrainer(trainer)
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=None,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=None,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='State her experiments')
    parser.add_argument('--special_name', type=str, default="", help="optional name of experiment")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    
    variant = dict(
        algorithm='HER-TD3',
        version='normal',
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=1000000,
            num_expl_steps_per_train_loop=100,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=50,
            random_before_training = True
        ),
        td3_trainer_kwargs=dict(
            discount=0.95,
            policy_learning_rate=args.lr,
            qf_learning_rate=args.lr,
            target_policy_noise=0.2,
            policy_and_target_update_period=2
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256,256,256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256,256,256],
        ),
    )
    setup_logger('{}-stateHER'.format(args.special_name), variant=variant, snapshot_mode='gap_and_last', snapshot_gap=100)
    experiment(variant)
