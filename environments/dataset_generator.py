from __future__ import division, absolute_import, print_function

import argparse
import glob
import multiprocessing
import os
import shutil
import time
# import dmc2gym
import numpy as np
# from stable_baselines import PPO2
# from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.policies import CnnPolicy
from environments import ThreadingType
# from environments.registry import registered_env
# from real_robots.constants import USING_OMNIROBOT
from srl_zoo.utils import printRed, printYellow

from environments.rlbench_gym.rlbenchds_env import RLBenchDSEnv


import gym
import rlbench.gym
from rlbench import VisualRandomizationConfig
rand_config = VisualRandomizationConfig(
    image_directory='/anonymized/experiment_textures/train/top10',
        whitelist = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible'],
        apply_floor=True
)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow
img_size = 288

def dr_flag(dr):
    if dr:
        return '-random'
    return ''

reach = lambda dr,ss, alt: gym.make('reach_target_easy-visiondepthmask%s-v0' % dr_flag(dr), altview=alt,rand_config=rand_config, fixed_grip=0, img_size=img_size, force_randomly_place=True, force_change_position=False)
push_rotate = lambda dr, ss, alt: gym.make('slide_block_to_target-visiondepthmask%s-v0' % dr_flag(dr),altview=alt, fixed_grip=0, rand_config=rand_config, img_size=img_size, force_randomly_place=False, force_change_position=False)
push_free = lambda dr,ss, alt: gym.make('slide_block_to_target-visiondepthmask%s-v0' % dr_flag(dr),altview=alt, fixed_grip=0, rand_config=rand_config,img_size=img_size, force_randomly_place=False, force_change_position=False)
pick_and_lift = lambda dr, ss, alt: gym.make('pick_and_lift-visiondepthmask%s-v0' % dr_flag(dr),altview=alt,rand_config=rand_config,sparse=True, not_special_p=0.5, ground_p = 0.5, special_is_grip=True, img_size=img_size, force_randomly_place=False, force_change_position=False)
pick_and_lift_nograb = lambda dr, ss, alt: gym.make('pick_and_lift-visiondepthmask%s-v0' % dr_flag(dr),altview=alt, rand_config=rand_config,sparse=True, not_special_p=0.5, ground_p = 0.5, special_is_grip=False, img_size=img_size, force_randomly_place=False, force_change_position=False)
procedural = lambda dr, ss, alt: gym.make('empty_container-visiondepthmask%s-v0'% dr_flag(dr),altview=alt, img_size=img_size,  not_special_p=0.5, ground_p = 0.5, special_is_grip=False, force_randomly_place=False, force_change_position=False, procedural_ind=0, procedural_mode='increase')

envs = {
    'reach': reach,
    'push_rotate': push_rotate,
    'push_free': push_free,
    'pick_and_lift': pick_and_lift,
    'pick_and_lift_nograb': pick_and_lift_nograb,
    'procedural': procedural
}

def convertImagePath(args, path, record_id_start):
    """
    Used to convert an image path, from one location, to another
    :param args: (ArgumentParser object)
    :param path: (str)
    :param record_id_start: (int) where does the current part start counting its records
    :return:
    """
    image_name = path.split("/")[-1]
    # get record id for output, by adding the current offset with the record_id
    # of the folder
    new_record_id = record_id_start + int(path.split("/")[-2].split("_")[-1])
    return args.name + "/record_{:03d}".format(new_record_id) + "/" + image_name


def env_thread(args, thread_num, partition=True, use_ppo2=False):
    """
    Run a session of an environment
    :param args: (ArgumentParser object)
    :param thread_num: (int) The thread ID of the environment session
    :param partition: (bool) If the output should be in multiple parts (default=True)
    :param use_ppo2: (bool) Use ppo2 to generate the dataset
    """
    env_kwargs = {
        "max_steps_per_epoch": args.max_steps_per_epoch
    }

    if partition:
        env_kwargs["name"] = args.name + "_part-" + str(thread_num)
    else:
        env_kwargs["name"] = args.name
    env_kwargs['path'] = args.save_path

    env_kwargs["wrapped_env"] = envs[args.env](args.dr, args.special_start, args.alt)
    env_class = RLBenchDSEnv
    env = env_class(**env_kwargs)

    frames = 0
    start_time = time.time()
    # divide evenly, then do an extra one for only some of them in order to get the right count
    for i_episode in range(args.num_episode // args.num_cpu + 1 * (args.num_episode % args.num_cpu > thread_num)):
        # seed + position in this slice + size of slice (with reminder if uneven partitions)
        seed = args.seed + i_episode + args.num_episode // args.num_cpu * thread_num + \
               (thread_num if thread_num <= args.num_episode % args.num_cpu else args.num_episode % args.num_cpu)

        env.seed(seed)
        env.action_space.seed(seed)  # this is for the sample() function from gym.space
        obs = env.reset()
        done = False
        t = 0
        episode_toward_target_on = False
        while not done:
            action = [env.action_space.sample()]

            action_to_step = action[0]

            frames += 1
            t += 1

            _, _, done, _ = env.step(action_to_step,t)

            done == done or t == args.max_steps_per_epoch
            if done:
                print("Episode finished after {} timesteps".format(t + 1))

        if thread_num == 0:
            print("{:.2f} FPS".format(frames * args.num_cpu / (time.time() - start_time)))


def main():
    parser = argparse.ArgumentParser(description='Deteministic dataset generator for SRL training ' +
                                                 '(can be used for environment testing)')
    parser.add_argument('--num-cpu', type=int, default=1, help='number of cpu to run on')
    parser.add_argument('--num-episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('--max_steps_per_epoch', type=int, default=200,
                        help='max num steps per epoch')

    #CUSTOM ARGS. want to udpate eventually, i.e., specify a specific path for dr
    parser.add_argument('--dr', action='store_true', default=False, help="Include this flag to use the chosen environment with domain randomization")
    parser.add_argument('--alt', action='store_true', default=False, help="Include this flag to use the chosen environment with alternate view")
    parser.add_argument('--special_start', action='store_true', default=False, help="Include this flag to use the chosen environment with the special start")

    parser.add_argument('--save-path', type=str, default='robotics-rl-srl/data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='UNSETNAME', help='Folder name for the output')
    parser.add_argument('--env', type=str, default='push_rotate', help='The environment wanted',
                        choices=list(envs.keys()))



    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no-record-data', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the save, even if it overrides something else,' +
                             ' including partial parts if they exist')

    #TODO: Change this argument to be for the diff types of tasks                        
    parser.add_argument('--multi-view', action='store_true', default=False, help='Set a second camera to the scene')

    parser.add_argument('--reward-dist', action='store_true', default=False,
                        help='Prints out the reward distribution when the dataset generation is finished')
    parser.add_argument('--run-ppo2', action='store_true', default=False,
                        help='runs a ppo2 agent instead of a random agent')
    parser.add_argument('--ppo2-timesteps', type=int, default=1000,
                        help='number of timesteps to run PPO2 on before generating the dataset')

    args = parser.parse_args()

    assert (args.num_cpu > 0), "Error: number of cpu must be positive and non zero"

    assert (args.num_episode > 0), "Error: number of episodes must be positive and non zero"
    # assert not(registered_env[args.env][3] is ThreadingType.NONE and args.num_cpu != 1), \
        # "Error: cannot have more than 1 CPU for the environment {}".format(args.env)

    if args.num_cpu > args.num_episode:
        args.num_cpu = args.num_episode
        printYellow("num_cpu cannot be greater than num_episode, defaulting to {} cpus.".format(args.num_cpu))

    # this is done so seed 0 and 1 are different and not simply offset of the same datasets.
    args.seed = np.random.RandomState(args.seed).randint(int(1e10))

    # File exists, need to deal with it
    if not args.no_record_data and os.path.exists(args.save_path + args.name):
        assert args.force, "Error: save directory '{}' already exists".format(args.save_path + args.name)

        shutil.rmtree(args.save_path + args.name)
        for part in glob.glob(args.save_path + args.name + "_part-[0-9]*"):
            shutil.rmtree(part)
    if not args.no_record_data:
        # create the output
        os.makedirs(args.save_path + args.name, exist_ok=True)

    if args.num_cpu == 1:
        env_thread(args, 0, partition=False, use_ppo2=args.run_ppo2)
    else:
        # try and divide into multiple processes, with an environment each
        try:
            jobs = []
            for i in range(args.num_cpu):
                process = multiprocessing.Process(target=env_thread, args=(args, i, True, args.run_ppo2))
                jobs.append(process)

            for j in jobs:
                j.start()

            try:
                for j in jobs:
                    j.join()
            except Exception as e:
                printRed("Error: unable to join thread")
                raise e

        except Exception as e:
            printRed("Error: unable to start thread")
            raise e

    if not args.no_record_data and args.num_cpu > 1:
        # sleep 1 second, to avoid congruency issues from multiprocess (eg., files still writing)
        time.sleep(1)
        # get all the parts
        file_parts = sorted(glob.glob(args.save_path + args.name + "_part-[0-9]*"), key=lambda a: int(a.split("-")[-1]))

        # move the config files from any as they are identical
        os.rename(file_parts[0] + "/dataset_config.json", args.save_path + args.name + "/dataset_config.json")
        os.rename(file_parts[0] + "/env_globals.json", args.save_path + args.name + "/env_globals.json")

        ground_truth = None
        preprocessed_data = None

        # used to convert the part record_id to the fused record_id
        record_id = 0
        for part in file_parts:
            # sort the record names alphabetically, then numerically
            records = sorted(glob.glob(part + "/record_[0-9]*"), key=lambda a: int(a.split("_")[-1]))

            record_id_start = record_id
            for record in records:
                os.renames(record, args.save_path + args.name + "/record_{:03d}".format(record_id))
                record_id += 1

            # fuse the npz files together, in the right order
            if ground_truth is None:
                # init
                ground_truth = {}
                preprocessed_data = {}
                ground_truth_load = np.load(part + "/ground_truth.npz")
                preprocessed_data_load = np.load(part + "/preprocessed_data.npz")

                for arr in ground_truth_load.files:
                    if arr == "images_path":
                        ground_truth[arr] = np.array(
                            [convertImagePath(args, path, record_id_start) for path in ground_truth_load[arr]])
                    else:
                        ground_truth[arr] = ground_truth_load[arr]
                for arr in preprocessed_data_load.files:
                    preprocessed_data[arr] = preprocessed_data_load[arr]

            else:
                ground_truth_load = np.load(part + "/ground_truth.npz")
                preprocessed_data_load = np.load(part + "/preprocessed_data.npz")

                for arr in ground_truth_load.files:
                    if arr == "images_path":
                        sanitised_paths = np.array(
                            [convertImagePath(args, path, record_id_start) for path in ground_truth_load[arr]])
                        ground_truth[arr] = np.concatenate((ground_truth[arr], sanitised_paths))
                    else:
                        ground_truth[arr] = np.concatenate((ground_truth[arr], ground_truth_load[arr]))
                for arr in preprocessed_data_load.files:
                    preprocessed_data[arr] = np.concatenate((preprocessed_data[arr], preprocessed_data_load[arr]))

            # remove the current part folder
            shutil.rmtree(part)

        # save the fused outputs
        np.savez(args.save_path + args.name + "/ground_truth.npz", **ground_truth)
        np.savez(args.save_path + args.name + "/preprocessed_data.npz", **preprocessed_data)

    if args.reward_dist:
        rewards, counts = np.unique(np.load(args.save_path + args.name + "/preprocessed_data.npz")['rewards'],
                                    return_counts=True)
        counts = ["{:.2f}%".format(val * 100) for val in counts / np.sum(counts)]
        print("reward distribution:")
        [print(" ", reward, count) for reward, count in list(zip(rewards, counts))]


if __name__ == '__main__':
    main()
