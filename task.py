import copy

import numpy as np

from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Base reward
        reward = 0

        # Heavy penalization for hitting ground
        if self.sim.pose[2] <= 0:
            reward -= 2
            self.sim.done = True

        previous_distance = np.linalg.norm(self.previous_pose[:3] - self.target_pos)
        current_distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)

        # Bonus reward if drone is close enough
        if current_distance < 1:
            reward += 1
            self.sim.done = True

        # Reward/Penalty for distance change
        reward += previous_distance - current_distance

        return reward, self.sim.done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        self.previous_pose = copy.copy(self.sim.pose)
        pose_all = []
        self.rotor_speeds = rotor_speeds
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            new_reward, new_done = self.get_reward()
            reward += new_reward
            if new_done:
                done = True
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
