"""
Procedural objects supplied from:
https://sites.google.com/site/brainrobotdata/home/models
"""

from typing import List
import random
import numpy as np
import os
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape

from rlbench.backend.conditions import ConditionSet, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
# from rlbench.backend.task_utils import sample_procedural_objects
from rlbench.const import colors

assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../assets/procedural_objects')
sorted_dir = sorted(os.listdir(assets_dir))


def sample_procedural_objects(task_base, num_samples, mass=0.1):
    # assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                           '../assets/procedural_objects')
    samples = np.random.choice(
        assets_dir, num_samples, replace=False)
    created = []
    for s in samples:
        respondable = os.path.join(assets_dir, s, s + '_coll.obj')
        visual = os.path.join(assets_dir, s, s + '.obj')
        resp = Shape.import_mesh(respondable, scaling_factor=0.005)
        vis = Shape.import_mesh(visual, scaling_factor=0.005)
        resp.set_renderable(False)
        vis.set_renderable(True)
        vis.set_parent(resp)
        vis.set_dynamic(False)
        vis.set_respondable(False)
        resp.set_dynamic(True)
        resp.set_mass(mass)
        resp.set_respondable(True)
        resp.set_model(True)
        resp.set_parent(task_base)
        created.append(resp)
    return created

def get_procedural_objects_by_index(task_base, indices, mass=0.1):
    samples = np.array(assets_dir)[indices]
    created = []
    for s in samples:
        respondable = os.path.join(assets_dir, s, s + '_coll.obj')
        visual = os.path.join(assets_dir, s, s + '.obj')
        resp = Shape.import_mesh(respondable, scaling_factor=0.005)
        vis = Shape.import_mesh(visual, scaling_factor=0.005)
        resp.set_renderable(False)
        vis.set_renderable(True)
        vis.set_parent(resp)
        vis.set_dynamic(False)
        vis.set_respondable(False)
        resp.set_dynamic(True)
        resp.set_mass(mass)
        resp.set_respondable(True)
        resp.set_model(True)
        resp.set_parent(task_base)
        created.append(resp)
    return created

def get_one_procedural_object_by_index(task_base, index, mass=0.1):
    s =  sorted_dir[index]
    respondable = os.path.join(assets_dir, s, s + '_coll.obj')
    visual = os.path.join(assets_dir, s, s + '.obj')
    resp = Shape.import_mesh(respondable, scaling_factor=0.005)
    vis = Shape.import_mesh(visual, scaling_factor=0.005)
    resp.set_renderable(False)
    vis.set_renderable(True)
    vis.set_parent(resp)
    vis.set_dynamic(False)
    vis.set_respondable(False)
    resp.set_dynamic(True)
    resp.set_mass(mass)
    resp.set_respondable(True)
    resp.set_model(True)
    resp.set_parent(task_base)
    return resp, vis


COLOR_TUPLE = [1,0,0]
class EmptyContainer(Task):

    def init_task(self) -> None:
        self.spawn_boundary = SpawnBoundary([Shape('spawn_boundary')])
        self.target=Shape('target')

        self.bin_objects = []
        self.block, self.block_visual = get_one_procedural_object_by_index(self.get_base(), self.procedural_ind)
        self.block_pose = self.block.get_pose()
        self.block_visual.set_color(self.COLOR_TUPLE)
        self.register_graspable_objects([self.block])
        self.ind = 0

    def init_episode(self, index: int) -> List[str]:
        if self.procedural_mode == "increase" :
            if self.procedural_set:
                self.block.remove()
                self.block, self.block_visual = get_one_procedural_object_by_index(self.get_base(), self.procedural_set[self.ind])
                self.block_pose = self.block.get_pose()
                self.block_visual.set_color(self.COLOR_TUPLE)
                self.register_graspable_objects([self.block])
                self.ind = (self.ind + 1) % len(self.procedural_set)
            else:
                self.block.remove()
                self.block, self.block_visual = get_one_procedural_object_by_index(self.get_base(), self.procedural_ind)
                self.block_pose = self.block.get_pose()
                self.block_visual.set_color(self.COLOR_TUPLE)
                self.register_graspable_objects([self.block])
                self.procedural_ind += 1
        elif self.procedural_mode == "random":
            assert self.procedural_set
            self.block.remove()
            self.block, self.block_visual = get_one_procedural_object_by_index(self.get_base(), random.choice(self.procedural_set))
            self.block_pose = self.block.get_pose()
            self.block_visual.set_color(self.COLOR_TUPLE)
            self.register_graspable_objects([self.block])
        self.spawn_boundary.clear()
        #Fix the target to spawn on a certain height.
        self.spawn_boundary.sample(self.target)
        pose = self.target.get_pose()
        if random.random() < self.ground_p:
            pose[2] = 7.75003076e-01
        else:   
            pose[2] = 8.83000314e-01
        self.target.set_pose(pose)

        self.block.set_pose(self.block_pose)
        self.block.set_position([0,0,0.76], reset_dynamics=False)
        self.spawn_boundary.sample(
            self.block, ignore_collisions=True, min_distance=0.05, min_rotation=(0,0,0), max_rotation=(0,0,0))


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
                print("***!!!! Failed to start from demonstration !!!!***")
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
            block_pos = self.block.get_position()
            _path_action(block_pos)
            done = False
            while not done:
                done = self.robot.gripper.actuate(0, velocity=1)
                self.pyrep.step()
                self.step()
            for g_obj in self.get_graspable_objects():
                self.robot.gripper.grasp(g_obj)

                
        return ''
    def variation_count(self) -> int:
        return len(colors)

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
        # return self.block.get_position()

        
    def get_state_obs(self): 
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_position = self.robot.arm.get_tip().get_position()
        gripper_open = [self.robot.gripper.get_open_amount()[0] > 0.9]
        gripper_vel = np.array(self.robot.arm.get_tip().get_velocity()).flatten()
        joint_vel = self.robot.arm.get_joint_velocities()
        if hasattr(self, "is_mesh_obs") and self.is_mesh_obs:
            vertices = np.zeros((600, 3))
            v = self.block.get_mesh_data()[0]
            vertices[:v.shape[0]] = v
            vertices -= self.block.get_position()
            block = vertices.flatten()
        else:
            block = self.block.get_pose()
        return np.concatenate([joint_positions, gripper_position, gripper_open, block, gripper_vel, joint_vel])

    def get_save_state(self):
        joint_positions = self.robot.arm.get_joint_positions()
        gripper_positions = self.robot.gripper.get_joint_positions()
        gripper_open = self.robot.gripper.get_open_amount()[0] > 0.9
        block_pose = self.block.get_pose()
        target_position = self.target.get_position()        
        return [joint_positions, gripper_positions, gripper_open, block_pose, target_position]
    def restore_save_state(self, state):
        joint_positions, gripper_positions, gripper_open, block_pose, target_position = state
        self.robot.arm.set_joint_positions(joint_positions)
        self.robot.gripper.set_joint_positions(gripper_positions)
        self.robot.gripper.actuate(gripper_open, 1)
        self.block.set_pose(block_pose)
        self.target.set_position(target_position)      