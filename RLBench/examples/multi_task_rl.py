from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MT30_V1
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        return (np.random.normal(0.0, 0.1, size=(self.action_size,))).tolist()


obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera.rgb = True
obs_config.right_shoulder_camera.rgb = True

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

agent = Agent(action_mode.action_size)

train_tasks = MT30_V1['train']

training_cycles_per_task = 3
training_steps_per_task = 80
episode_length = 40

for _ in range(training_cycles_per_task):

    task_to_train = np.random.choice(train_tasks, 1)[0]
    task = env.get_task(task_to_train)
    task.sample_variation()  # random variation

    for i in range(training_steps_per_task):
        if i % episode_length == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
            print(descriptions)
        action = agent.act(obs)
        obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
