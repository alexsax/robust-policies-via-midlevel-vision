from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def ingest(self, demos):
        pass

    def act(self, obs):
        return (np.random.normal(0.0, 0.1, size=(self.action_size,))).tolist()


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera.rgb = True
obs_config.right_shoulder_camera.rgb = True

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

task = env.get_task(ReachTarget)
demos = task.get_demos(2, live_demos=live_demos)

agent = Agent(action_mode.action_size)
agent.ingest(demos)

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
