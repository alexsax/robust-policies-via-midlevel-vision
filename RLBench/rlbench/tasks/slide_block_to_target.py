from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
import numpy as np
from rlbench.backend.spawn_boundary import SpawnBoundary

class SlideBlockToTarget(Task):
 
    def init_task(self) -> None:
        self.block = Shape('block')
        self.success_sensor = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.block, self.success_sensor)])
        self.target = Shape('target')
        self.workspace = Shape('workspace')
        self.target_boundary = SpawnBoundary([Shape('target_boundary')])
    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index

        self.target_boundary.clear()
        self.target_boundary.sample(self.block, change_position=False, min_rotation=(0,0,0), max_rotation=(0,0,0)) #just add the block to the boundary.
        notdoneyet = True
        while notdoneyet:
            self.target_boundary.sample(self.target, min_distance=0.05, min_rotation=(0,0,0), max_rotation=(0,0,0))
            notdoneyet = self.get_reward_and_done()[1]
        joint_pos = [-5.435585626401007e-05, 0.39650705456733704, 3.4842159948311746e-05, -2.453382730484009, 0.00016987835988402367, 3.0057222843170166, 0.785248875617981]
        gripper_pos = [0.0004903227090835571, 0.0007378421723842621]
        self.robot.arm.set_joint_positions(joint_pos)
        self.robot.gripper.set_joint_positions(gripper_pos)
        self.robot.gripper.actuate(0, 1)
        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def get_reward_and_done(self, sparse=False):

        dist = np.linalg.norm(np.array(self.target.get_position()) - np.array(self.block.get_position()))
        thresh = 0.04
        if sparse:
            return -(dist > thresh).astype(np.float32), dist < thresh
        #epsilon reward
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
        joint_vel = self.robot.arm.get_joint_velocities()
        gripper_position = self.robot.arm.get_tip().get_position()
        gripper_vel = np.array(self.robot.arm.get_tip().get_velocity()).flatten()
        block = self.block.get_pose()
        return np.concatenate([joint_positions, gripper_position, block, joint_vel, gripper_vel])
    def get_save_state(self):
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_positions = self.robot.gripper.get_joint_positions()
        block_position = self.block.get_position()
        target_position = self.target.get_position()        
        return [joint_positions, gripper_positions, block_position, target_position]
    def restore_save_state(self, state):
        joint_positions, gripper_positions , block_position, target_position = state
        self.robot.arm.set_joint_positions(joint_positions)
        self.robot.gripper.set_joint_positions(gripper_positions)
        self.block.set_position(block_position)
        self.target.set_position(target_position)        
