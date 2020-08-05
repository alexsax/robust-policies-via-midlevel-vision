import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np

# Randomness imports
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench.backend.utils import rgb_handles_to_mask

class RLBenchEnv(gym.GoalEnv):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human']}
    def __init__(self, task_class, observation_mode='state', randomization_mode="none", 
        rand_config=None, img_size=256, special_start=[], fixed_grip=-1, force_randomly_place=False, force_change_position=False, sparse=False, not_special_p = 0, ground_p = 0, special_is_grip=False, altview=False, procedural_ind=0, procedural_mode='same', procedural_set = [], is_mesh_obs=False, blank=False, COLOR_TUPLE=[1,0,0]):
        # blank is 0s agent. 
        self.blank = blank
        #altview is whether to have second camera angle or not. True/False, "both" to concatentae the observations. 
        self.altview=altview
        self.img_size=img_size
        self.sparse = sparse
        self.task_class = task_class
        self._observation_mode = observation_mode
        self._randomization_mode = randomization_mode
        #special start is a list of actions to take at the beginning.
        self.special_start = special_start
        #fixed_grip temp hack for keeping the gripper a certain way. Change later. 0 for fixed closed, 0.1 for fixed open, -1 for not fixed
        self.fixed_grip = fixed_grip
        #to force the task to be randomly placed
        self.force_randomly_place = force_randomly_place
        #force the task to change position in addition to rotation
        self.force_change_position = force_change_position



        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision' or observation_mode=="visiondepth" or observation_mode=="visiondepthmask":
            # obs_config.set_all(True)
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)

        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_NOQ)
        print("using delta pose pan")

        if randomization_mode == "random":
            objs = ['target', 'boundary', 'Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible']
            if rand_config is None:
                assert False
            self.env = DomainRandomizationEnvironment(
                action_mode, obs_config=obs_config, headless=True,
                randomize_every=RandomizeEvery.EPISODE, frequency=1,
                visual_randomization_config=rand_config
            )
        else:
            self.env = Environment(
                action_mode, obs_config=obs_config, headless=True
            )
        self.env.launch()
        self.task = self.env.get_task(task_class)

        # Probability. Currently used for probability that pick and lift task will start off gripper at a certain location (should probs be called non_special p)
        self.task._task.not_special_p = not_special_p
        # Probability that ground goal.
        self.task._task.ground_p = ground_p 
        # For the "special" case, whether to grip the object or just hover above it. 
        self.task._task.special_is_grip = special_is_grip
        # for procedural env
        self.task._task.procedural_ind = procedural_ind
        # procedural mode: same, increase, or random. 
        self.task._task.procedural_mode = procedural_mode
        # ideally a list-like object, dictates the indices to sample from each episode. 
        self.task._task.procedural_set = procedural_set
        # if state obs is mesh obs
        self.task._task.is_mesh_obs = is_mesh_obs

        self.task._task.sparse = sparse
        self.task._task.COLOR_TUPLE = COLOR_TUPLE
        
        _, obs = self.task.reset()

        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_pose = cam_placeholder.get_pose().copy()
        #custom pose
        cam_pose = [ 1.59999931,  0. ,         2.27999949 , 0.65328157, -0.65328145, -0.27059814, 0.27059814]
        cam_pose[0] = 1
        cam_pose[2] = 1.5
        self.frontcam = VisionSensor.create([img_size, img_size])
        self.frontcam.set_pose(cam_pose)

        self.frontcam.set_render_mode(RenderMode.OPENGL)
        self.frontcam.set_perspective_angle(60)
        self.frontcam.set_explicit_handling(1)

        self.frontcam_mask = VisionSensor.create([img_size, img_size])
        self.frontcam_mask.set_pose(cam_pose)
        self.frontcam_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
        self.frontcam_mask.set_perspective_angle(60)
        self.frontcam_mask.set_explicit_handling(1)

        if altview:
            alt_pose = [0.25    , -0.65    ,  1.5,   0, 0.93879825 ,0.34169483 , 0]
            self.altcam = VisionSensor.create([img_size, img_size])
            self.altcam.set_pose(alt_pose)
            self.altcam.set_render_mode(RenderMode.OPENGL)
            self.altcam.set_perspective_angle(60)
            self.altcam.set_explicit_handling(1)
            
            self.altcam_mask = VisionSensor.create([img_size, img_size])
            self.altcam_mask.set_pose(alt_pose)
            self.altcam_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
            self.altcam_mask.set_perspective_angle(60)
            self.altcam_mask.set_explicit_handling(1)


        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_mode.action_size,))

        if observation_mode == 'state':
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=self.task._task.get_state_obs().shape),
                "achieved_goal": spaces.Box(
                low=-np.inf, high=np.inf, shape=self.task._task.get_achieved_goal().shape),
                'desired_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=self.task._task.get_desired_goal().shape)
            })
        # Use the frontvision cam
        elif observation_mode == 'vision':
            self.frontcam.handle_explicitly()
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "observation": spaces.Box(
                    low=0, high=1, shape=self.frontcam.capture_rgb().transpose(2,0,1).flatten().shape,
                    )
                })
            if altview == "both":
                example = self.frontcam.capture_rgb().transpose(2,0,1).flatten()
                self.observation_space = spaces.Dict({
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=obs.get_low_dim_data().shape),
                    "observation": spaces.Box(
                        low=0, high=1, shape=np.array([example,example]).shape,
                        )
                    })
        elif observation_mode == 'visiondepth':

            self.frontcam.handle_explicitly()
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "observation": spaces.Box(
                    #thinking about not flattening the shape, because primarily used for dataset and not for training
                    low=0, high=1, shape=np.array([self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_depth()[None,...]]).shape,
                    )
                })
        elif observation_mode == 'visiondepthmask':

            self.frontcam.handle_explicitly()
            self.frontcam_mask.handle_explicitly()
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "observation": spaces.Box(
                    #thinking about not flattening the shape, because primarily used for dataset and not for training
                    low=0, high=1, shape=np.array([self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_depth()[None,...], rgb_handles_to_mask(self.frontcam_mask.capture_rgb())]).shape,
                    )
                })
            if altview == "both":
                self.observation_space = spaces.Dict({
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=obs.get_low_dim_data().shape),
                    "observation": spaces.Box(
                        #thinking about not flattening the shape, because primarily used for dataset and not for training
                        low=0, high=1, shape=np.array([self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_depth()[None,...], rgb_handles_to_mask(self.frontcam_mask.capture_rgb())]).shape,
                        )
                    })

        self._gym_cam = None


    # GoalEnv 
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.task._task.compute_reward(achieved_goal, desired_goal, info)
        return reward

    def _extract_obs(self, obs):
        if self._observation_mode == 'state':
            return {
                'observation': self.task._task.get_state_obs(),
                'achieved_goal': self.task._task.get_achieved_goal(),
                'desired_goal':self.task._task.get_desired_goal(),
            }
        elif self._observation_mode == 'vision':
            self.frontcam.handle_explicitly()
            if self.altview:
                self.altcam.handle_explicitly()
                second_view = self.altcam.capture_rgb().transpose(2,0,1).flatten()
                if self.altview == "both":
                    return {
                        'achieved_goal': self.task._task.get_achieved_goal(),
                        'desired_goal':self.task._task.get_desired_goal(),
                        'save_state': self.task._task.get_save_state(),
                        "observation": np.array([self.frontcam.capture_rgb().transpose(2,0,1).flatten(), second_view])
                    } 
                return {
                    'achieved_goal': self.task._task.get_achieved_goal(),
                    'desired_goal':self.task._task.get_desired_goal(),
                    'save_state': self.task._task.get_save_state(),
                    "observation": self.altcam.capture_rgb().transpose(2,0,1).flatten()
                }
            if self.blank:
                return {
                    'achieved_goal': self.task._task.get_achieved_goal(),
                    'desired_goal':self.task._task.get_desired_goal(),
                    'save_state': self.task._task.get_save_state(),
                    "observation": np.zeros(self.observation_space['observation'].shape)
                }                
            return {
                'achieved_goal': self.task._task.get_achieved_goal(),
                'desired_goal':self.task._task.get_desired_goal(),
                'save_state': self.task._task.get_save_state(),
                "observation": self.frontcam.capture_rgb().transpose(2,0,1).flatten(),
            }
        elif self._observation_mode == 'visiondepth':
            self.frontcam.handle_explicitly()
            return {
                "state": obs.get_low_dim_data(),
                "observation": [self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_depth()[None,...]]
            }
        elif self._observation_mode == 'visiondepthmask':
            self.frontcam.handle_explicitly()
            self.frontcam_mask.handle_explicitly()

            mask = np.array(self.frontcam_mask.capture_rgb())

            if self.altview:
                self.altcam.handle_explicitly()
                self.altcam_mask.handle_explicitly()
                altmask = np.array(self.altcam_mask.capture_rgb())
                second_view = self.altcam.capture_rgb().transpose(2,0,1)

                if self.altview == "both":
                    assert False
                    # no use case for this
                    return {
                        "state": obs.get_low_dim_data(),
                        "observation": [self.frontcam.capture_rgb().transpose(2,0,1), self.altcam.capture_rgb().transpose(2,0,1).flatten()]
                    } 
                return {
                    "state": obs.get_low_dim_data(),
                    "observation": [self.altcam.capture_rgb().transpose(2,0,1), self.altcam.capture_depth()[None,...], altmask]
                }


            return {
                "state": obs.get_low_dim_data(),
                "observation": [self.frontcam.capture_rgb().transpose(2,0,1), self.frontcam.capture_depth()[None,...], mask]
            }

    def render(self, mode='human'):
        self.frontcam.handle_explicitly()
        return self.frontcam.capture_rgb()

    def reset(self):
        _, obs = self.task.reset(force_randomly_place=self.force_randomly_place, force_change_position=self.force_change_position)
        import time
        if len(self.special_start) != 0:
            for a in self.special_start:
                self.step(a)
        # Doesn't actually use obs. 
        return self._extract_obs(obs)

    def step(self, action):
        if self.fixed_grip >= 0:
            action[3] = self.fixed_grip
        obs, reward, terminate = self.task.step(action)
        reward, done = self.task._task.get_reward_and_done(sparse=self.sparse)
        return self._extract_obs(obs), reward, done, {"is_success": int(done)}

    def close(self):
        self.env.shutdown()

    def rerender(self, state):
        self.task._task.restore_full_state(state)
        return self.render().transpose(2,0,1).flatten()


    def resample_step(self, state, action, sampled_goal):
        self.task._task.restore_save_state(state)
        Shape('target').set_position(sampled_goal)

        if self.altview == "both":
            self.frontcam.handle_explicitly()
            self.altcam.handle_explicitly()
            o_before = np.array([self.frontcam.capture_rgb().transpose(2,0,1).flatten(), self.altcam.capture_rgb().transpose(2,0,1).flatten()])
        elif self.altview:
            self.altcam.handle_explicitly()
            o_before =  self.altcam.capture_rgb().transpose(2,0,1).flatten()
        else:
            if self.blank:
                o_before = np.zeros(self.observation_space['observation'].shape)
            else:
                o_before = self.render().transpose(2,0,1).flatten()
        o, r, d, _ = self.step(action)
        o_after = o['observation']
        return o_before, r, d, o_after