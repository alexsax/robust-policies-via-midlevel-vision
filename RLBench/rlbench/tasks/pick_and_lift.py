from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors
import random


class PickAndLift(Task):

    def init_task(self) -> None:
        self.block = Shape('pick_and_lift_target')
        self.register_graspable_objects([self.block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.block),
            DetectedCondition(self.block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])

        self.target = Shape('target')

    def base_rotation_bounds(self):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.block.set_color(block_rgb)

        self.boundary.clear()
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))

        if random.random() < self.ground_p:
            pose = self.target.get_pose()
            pose[2] = 7.75003076e-01
            self.target.set_pose(pose)

        for block in [self.block]: #+ self.distractors:
            self.boundary.sample(block, min_distance=0.1, min_rotation=(0,0,0), max_rotation=(0,0,0))
        def _path_action(pos):
            x, y, z, qx, qy, qz, qw = self.robot.arm.get_tip().get_pose()
            try:
                path = self.robot.arm.get_path(
                    pos[:3], quaternion=[qx, qy, qz, qw], ignore_collisions=True)
                done = False
                observations = []
                while not done:
                    done = path.step()
                    self.pyrep.step()
            except Exception as e:
                print(e)
                print("***!!!! Failed to start pick and lift from demonstration !!!!***")
                print("pos", pos)
                print("current xyz", x, y, z)

        if random.random() < self.not_special_p:
            joint_pos = [-1.8593069398775697e-05, -0.09871861338615417, 4.5094158849678934e-05, -2.7951412200927734, -3.730739263119176e-05, 2.8503623008728027, 0.7854341268539429]
            gripper_pos = [0.04000139236450195, 0.04000053554773331]
            self.robot.arm.set_joint_positions(joint_pos)
            self.robot.gripper.set_joint_positions(gripper_pos)
            self.robot.gripper.actuate(1, 1)
            return

        if not self.special_is_grip:
            _path_action(self.block.get_position() + np.array([0,0,0.04]))
        else:
            _path_action(self.block.get_position())
            done = False
            while not done:
                done = self.robot.gripper.actuate(0, velocity=1)
                self.pyrep.step()
                self.step()
            for g_obj in self.get_graspable_objects():
                self.robot.gripper.grasp(g_obj)
        return ""
    def get_reward(self):
        dist = np.linalg.norm(np.array(self.target.get_position()) - np.array(self.block.get_position()))
        thresh = 0.04
        if sparse:
            return -(dist > thresh).astype(np.float32), dist < thresh
        #epsilon reward
        reward = -dist
        if dist < thresh:
            reward += 0.1
        return reward
    def variation_count(self) -> int:
        return len(colors)


    def get_reward_and_done(self, sparse=False):

        dist = np.linalg.norm(np.array(self.target.get_position()) - np.array(self.block.get_position()))
        thresh = 0.04
        if sparse:
            return -(dist > thresh).astype(np.float32), dist < thresh
        reward = -dist
        if dist < thresh:
            reward += 0.1
        return reward, dist < thresh

    #For HER
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(np.array(desired_goal) - np.array(achieved_goal))
        thresh = 0.04
        return -(dist > thresh).astype(np.float32)
    def compute_reward_and_done(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(np.array(desired_goal) - np.array(achieved_goal))
        thresh = 0.04
        return -(dist > thresh).astype(np.float32), dist < thresh

    def variation_count(self) -> int:
        return 1
    def get_desired_goal(self):
        return self.target.get_position()

    def get_achieved_goal(self):
        return self.block.get_position()

        
    def get_state_obs(self): 
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_position = self.robot.arm.get_tip().get_position()
        gripper_open = [self.robot.gripper.get_open_amount()[0] > 0.9]
        gripper_vel = np.array(self.robot.arm.get_tip().get_velocity()).flatten()
        joint_vel = self.robot.arm.get_joint_velocities()
        block = self.block.get_position()
        return np.concatenate([joint_positions, gripper_position, gripper_open, block, gripper_vel, joint_vel])

    def get_save_state(self):
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_positions = self.robot.gripper.get_joint_positions()
        gripper_open = self.robot.gripper.get_open_amount()[0] > 0.9
        block_position = self.block.get_position()
        target_position = self.target.get_position()        
        return [joint_positions, gripper_positions, gripper_open, block_position, target_position]
    def restore_save_state(self, state):
        joint_positions, gripper_positions, gripper_open, block_position, target_position = state
        self.robot.arm.set_joint_positions(joint_positions)
        self.robot.gripper.set_joint_positions(gripper_positions)
        self.robot.gripper.actuate(gripper_open, 1)
        self.block.set_position(block_position)
        self.target.set_position(target_position)      