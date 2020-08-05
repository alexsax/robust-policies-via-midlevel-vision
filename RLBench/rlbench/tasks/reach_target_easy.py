from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects.dummy import Dummy
from pyrep.const import RenderMode


class ReachTargetEasy(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

        self.thresh = 0.04

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        joint_pos = [-5.435585626401007e-05, 0, 3.4842159948311746e-05, -2.453382730484009, 0.00016987835988402367, 3.0057222843170166, 0.785248875617981]
        gripper_pos = [0.0004903227090835571, 0.0007378421723842621]
        self.robot.arm.set_joint_positions(joint_pos)
        self.robot.gripper.set_joint_positions(gripper_pos)
        return ['reach the %s target' % color_name]

    def get_desired_goal(self):
        return self.target.get_position()
    def get_achieved_goal(self):
        return self.robot.arm.get_tip().get_position()

    def get_reward_and_done(self, sparse = False):
        dist = np.linalg.norm(np.array(self.robot.arm.get_tip().get_position()) - np.array(self.target.get_position()))
        if sparse:
            return -(dist > self.thresh).astype(np.float32), dist < self.thresh
        return -dist, dist < self.thresh 
 

    #For HER
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(np.array(desired_goal) - np.array(achieved_goal))
        return -(dist > self.thresh).astype(np.float32)
    def compute_reward_and_done(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(np.array(desired_goal) - np.array(achieved_goal))
        return -(dist > self.thresh).astype(np.float32), dist < self.thresh

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        return np.array(self.target.get_position())

    def get_state_obs(self): 
        joint_positions = self.robot.arm.get_joint_positions()
        joint_vel = self.robot.arm.get_joint_velocities()
        gripper_position = self.robot.arm.get_tip().get_position()
        gripper_vel = np.array(self.robot.arm.get_tip().get_velocity()).flatten()
        if self.sparse:
            target = self.get_desired_goal()
            return  np.concatenate([joint_positions, gripper_position, joint_vel, gripper_vel, target])
        return np.concatenate([joint_positions, gripper_position, joint_vel, gripper_vel])

    def get_save_state(self):
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_positions = self.robot.gripper.get_joint_positions()
        target_position = self.target.get_position()        
        return [joint_positions, gripper_positions, target_position]
    def restore_save_state(self, state):
        joint_positions, gripper_positions , target_position = state
        self.robot.arm.set_joint_positions(joint_positions)
        self.robot.gripper.set_joint_positions(gripper_positions)
        self.target.set_position(target_position)        
